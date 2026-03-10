import os
from dotenv import load_dotenv
load_dotenv(verbose=True)
import time
import enum
import json
import re
from typing import TypeVar, cast, Optional, List
from langchain_core.prompts import PromptTemplate
from patchright.async_api import ElementHandle, Page
from pydantic import BaseModel, Field, create_model
import requests
import urllib.parse
from urllib.parse import quote, urlsplit, urlunsplit, urlparse, parse_qs
import asyncio
from browser_use import Agent
from browser_use.agent.views import ActionModel, ActionResult
from browser_use import Browser
from typing import Literal, Optional
from browser_use.utils import time_execution_sync
import inspect
from typing import Any
from types import SimpleNamespace
from browser_use.agent.views import ActionModel

Context = TypeVar('Context')

class NoParams(BaseModel):
    pass

class SearchQuery(BaseModel):
    query: str

class OrcidQuery(BaseModel):
    query: str

class ClickElementAction(BaseModel):
    index: int

class InputTextAction(BaseModel):
    index: int
    text: str
    has_sensitive_data: bool = False

class ScrollAction(BaseModel):
    amount: Optional[int] = None

class SendKeysAction(BaseModel):
    keys: str | List[str]

class SwitchTabAction(BaseModel):
    page_id: int

class OpenTabAction(BaseModel):
    url: str

class CloseTabAction(BaseModel):
    page_id: int

class GoToUrlAction(BaseModel):
    url: str

class DragDropAction(BaseModel):
    # pick whatever you actually support in your impl:
    element_source: Optional[str] = None
    element_target: Optional[str] = None
    coord_source_x: Optional[int] = None
    coord_source_y: Optional[int] = None
    coord_target_x: Optional[int] = None
    coord_target_y: Optional[int] = None
    steps: int = 10
    delay_ms: int = 5

class ZoomToPointParams(BaseModel):
    # Choose ONE way to specify the target:
    selector: Optional[str] = None         # e.g. "canvas", "img[alt*='sign']"
    text: Optional[str] = None             # visible text near the target
    x: Optional[float] = None              # normalized [0..1] viewport x
    y: Optional[float] = None              # normalized [0..1] viewport y

    steps: int = 6                         # how many zoom increments
    strategy: Literal["wheel","double_click","controls","auto"] = "auto"
    center_first: bool = True              # click once to set focus before zoom
    step_wait_ms: int = 180                # delay between increments

class FindArchiveURLAction(BaseModel):
    url: str
    date: str

class PDFAction(BaseModel):
    action: str # download, scroll_down, scroll_up, jump
    # download
    pdf_url: str | None
    save_name: str | None
    # jump
    page_number: int | None
    # scroll
    pixels: int | None
    # search
    search_text: str | None

class FindInPageAction(BaseModel):
    query: str
    match_case: bool = False
    whole_word: bool = False
    max_snippets: int = 100
    highlight: bool = False  # set True if you want <mark> injection

class VideoAction(BaseModel):
    action: str # jump
    # jump
    video_url: str | None
    time: int | None

class EnsureEarliestSVParams(BaseModel):
    max_clicks: int = 80   # how many “Previous/Earlier” clicks to try
    try_drag: bool = True  # also drag slider fully left as a fallback

class TimelineArrowParams(BaseModel):
    # 'left' ≈ earlier imagery, 'right' ≈ later imagery
    direction: Literal["left", "right", "earlier", "later", "prev", "next"] = "left"
    steps: int = 5                 # how many arrow clicks
    wait_ms: int = 250             # pause between clicks
    ensure_timeline_open: bool = True  # try to open the time widget first

class DoneParams(BaseModel):
    text: str
    success: bool = True

class SearchQuery(BaseModel):
    query: str

class KeysParam(BaseModel):
    keys: str | list[str]

def _maybe_await(x):
    if inspect.isawaitable(x):
        return x
    async def _wrap():
        return x
    return _wrap()

class _BrowserAdapter:
    def __init__(self, session_or_browser):
        self._s = session_or_browser

    async def _get_ctx(self):
        s = self._s

        # 1) Already-initialized context locations
        for attr in ("playwright_context", "context"):
            ctx = getattr(s, attr, None)
            if ctx:
                return ctx

        # 2) If the session holds a sub-session with the context
        sess = getattr(s, "session", None)
        if sess:
            for attr in ("playwright_context", "context"):
                ctx = getattr(sess, attr, None)
                if ctx:
                    return ctx

        # 3) Late-init hooks in browser_use 0.7.x
        #    Try methods that prepare/return a session/context.
        if hasattr(s, "get_session"):
            try:
                sess = await s.get_session()
                for attr in ("playwright_context", "context"):
                    ctx = getattr(sess, attr, None)
                    if ctx:
                        return ctx
            except Exception:
                pass

        if hasattr(s, "ensure_playwright"):
            try:
                await s.ensure_playwright()
                for attr in ("playwright_context", "context"):
                    ctx = getattr(s, attr, None)
                    if ctx:
                        return ctx
            except Exception:
                pass

        # 4) Fall back: if there’s a Playwright browser, create a context
        for candidate in (getattr(s, "playwright_browser", None),
                          getattr(getattr(s, "session", None), "playwright_browser", None)):
            if candidate:
                return await candidate.new_context()

        raise AttributeError("No Playwright context on BrowserSession")

    async def get_current_page(self):
        ctx = await self._get_ctx()
        pages = ctx.pages
        if pages:
            return pages[-1]
        # ensure at least one page exists
        page = await ctx.new_page()
        return page

    async def create_new_tab(self, url: str | None = None):
        page = await self.get_current_page()
        ctx = page.context
        new_page = await ctx.new_page()
        if url:
            await browser.navigate_to(url)
            # await new_page.goto(url, wait_until="domcontentloaded")
        return new_page

    async def switch_to_tab(self, index: int):
        ctx = await self._get_ctx()
        pages = ctx.pages
        if not pages:
            await ctx.new_page()
            pages = ctx.pages
        if index < 0:
            index = len(pages) + index
        index = max(0, min(index, len(pages)-1))
        return pages[index]

    async def go_back(self):
        page = await self.get_current_page()
        try:
            await page.go_back()
        except Exception:
            pass

    async def get_session(self):
        # for places where your tools still ask for session
        return getattr(self._s, "session", self._s)

class Controller():
    def __init__(
            self,
            exclude_actions: list[str] = [],
            output_model: type[BaseModel] | None = None,
            http_save_path: str = None,
            http_server_port: int = 8080,
    ):
        self.http_save_path = http_save_path
        self.http_server_port = http_server_port

        # self.registry = Registry[Context](exclude_actions)

        self._actions: dict[str, Any] = {}          # name -> callable
        self._action_specs: dict[str, dict[str, Any]] = {}
        self._registry_actions: dict[str, SimpleNamespace] = {}
        self._exclude = set(exclude_actions or [])
        self._context_params = {"browser", "page_extraction_llm", "sensitive_data", "available_file_paths", "context", "has_sensitive_data"}

        def _compute_param_model(func: Any, action_name: str, override: type[BaseModel] | None = None):
            sig = inspect.signature(func)
            user_params = []
            for param in sig.parameters.values():
                if param.name in self._context_params:
                    continue
                if param.name == "_":
                    continue
                user_params.append(param)

            takes_model = False
            param_model: type[BaseModel]

            if override is not None:
                param_model = override
                takes_model = issubclass(override, BaseModel)
            elif user_params and isinstance(user_params[0].annotation, type) and issubclass(user_params[0].annotation, BaseModel):
                param_model = user_params[0].annotation
                takes_model = True
            else:
                fields = {}
                for param in user_params:
                    ann = param.annotation if param.annotation is not inspect._empty else Any
                    default = param.default if param.default is not inspect._empty else ...
                    fields[param.name] = (ann, default)
                param_model = create_model(f"{action_name}_params", __base__=BaseModel, **fields)
            return param_model, takes_model, [p.name for p in user_params]

        def register_tool(func: Any, name: str | None = None, description: str | None = None, param_model: type[BaseModel] | None = None):
            n = name or getattr(func, "name", func.__name__)
            if n in self._exclude:
                return
            desc = description or getattr(func, "description", getattr(func, "__doc__", "") or "")
            param_model_res, takes_model, user_param_names = _compute_param_model(func, n, param_model)
            setattr(func, "description", desc)
            self._actions[n] = func
            self._action_specs[n] = {
                "func": func,
                "description": desc,
                "param_model": param_model_res,
                "takes_model": takes_model,
                "user_param_names": user_param_names,
            }
            self._registry_actions[n] = SimpleNamespace(
                name=n,
                description=desc,
                function=func,
                param_model=param_model_res,
                domains=None,
                page_filter=None,
            )
            setattr(self, n, func)

        self.register_tool = register_tool

        def _auto_register(ns: dict[str, Any]):
            """Auto-register coroutine functions we just defined inside __init__."""
            for n, obj in ns.items():
                if n.startswith("_"):
                    continue
                if inspect.iscoroutinefunction(obj):
                    register_tool(obj, name=n)

        """Register all default browser actions"""

        if output_model is not None:
            # Create a new model that extends the output model with success parameter
            class ExtendedOutputModel(BaseModel):  # type: ignore
                success: bool = True
                data: output_model  # type: ignore

            async def done(params: DoneParams):
                """Finish the task and return the final answer.

                Params (object):
                - text: str — final answer to return.
                - success: bool (default: True) — whether the task succeeded.

                Returns: ActionResult(is_done=True, extracted_content=<text>, success=<success>)"""

                # Exclude success from the output JSON since it's an internal parameter
                output_dict = params.data.model_dump()

                # Enums are not serializable, convert to string
                for key, value in output_dict.items():
                    if isinstance(value, enum.Enum):
                        output_dict[key] = value.value

                return ActionResult(is_done=True, success=params.success, extracted_content=params.text)
        else:

            async def done(params: DoneParams):
                """Finish the task and return the final answer.

                Params (object):
                - text: str — final answer to return.
                - success: bool (default: True) — whether the task succeeded.

                Returns: ActionResult(is_done=True, extracted_content=<text>, success=<success>)"""

                return ActionResult(is_done=True, success=params.success, extracted_content=params.text)

        #=== Custom code - Kim ===#

        async def take_screenshot(browser):
            """Save a screenshot of the current page to disk (PNG; filename based on URL).

                Params: {}

                Returns: ActionResult(extracted_content="Saved screenshot of <url> to ./<file>.png")"""

            page = await browser.get_current_page()

            # Generate a safe filename from the URL
            short_url = re.sub(r'^https?://(?:www\.)?|/$', '', page.urget_current_page_urll)
            slug = re.sub(r'[^a-zA-Z0-9]+', '-', short_url).strip('-').lower()
            sanitized_filename = f'{slug}.png'

            # Save screenshot
            await page.take_screenshot(path=sanitized_filename, full_page=True, quality=80)

            msg = f'Saved screenshot of {page.get_current_page_url} to ./{sanitized_filename}'
            # logger.info(msg)
            return ActionResult(
                extracted_content=msg,
                include_in_memory=True,
            )

        # Code to combat task id #3
        async def find_in_page(params: FindInPageAction, browser):
            """Find text on the current page and return match count + context snippets.

            Params (object):
            - query: str — text to search for.
            - match_case: bool (default: False)
            - whole_word: bool (default: False)
            - max_snippets: int (default: 100) — limit snippet count.
            - highlight: bool (default: False) — (info only; no DOM injection here)

            Returns: ActionResult(data={"count": int, "snippets": [str], "query": str})"""

            page = await browser.get_current_page()

            text = await page.inner_text("body")   # get visible text only
            query = params.query
            flags = 0 if params.match_case else re.IGNORECASE
            pattern = r"\b{}\b".format(re.escape(query)) if params.whole_word else re.escape(query)

            matches = list(re.finditer(pattern, text, flags))
            count = len(matches)

            snippets = []
            for m in matches[: params.max_snippets]:
                i = m.start()
                start = max(0, i - 40)
                end = min(len(text), i + len(query) + 40)
                snippets.append(text[start:i] + "[" + m.group() + "]" + text[i+len(query):end])

            msg = f"🔎 Found {count} match(es) for '{query}'"
            # logger.info(msg)
            return ActionResult(
                extracted_content=msg,
                data={"count": count, "snippets": snippets, "query": query},
                include_in_memory=True
            )

        async def zoom_to_point(params: ZoomToPointParams, browser):
            """Zoom into a target point on the page (e.g., maps/street view/canvas).

            Params (object) — choose ONE way to target:
            - selector: str | null — CSS/XPath-like selector to center/zoom.
            - text: str | null — visible text near the target to center/zoom.
            - x: float | null, y: float | null — normalized viewport coords [0..1].

            Optional:
            - steps: int (default: 6) — zoom increments.
            - strategy: "wheel" | "double_click" | "controls" | "auto" (default: "auto")
            - center_first: bool (default: True) — focus before zoom.
            - step_wait_ms: int (default: 180)

            Returns: ActionResult(data={"x": int, "y": int, "log": [(str, int)]})"""

            page = await browser.get_current_page()

            # --- Resolve target coordinates (cx, cy in page pixels) ---
            cx = cy = None
            if params.selector:
                try:
                    loc = page.locator(params.selector).first
                    await loc.wait_for(state="visible", timeout=1500)
                    box = await loc.bounding_box()
                    if box:
                        cx = int(box["x"] + box["width"]/2)
                        cy = int(box["y"] + box["height"]/2)
                except Exception:
                    pass

            if cx is None and params.text:
                try:
                    loc = page.get_by_text(params.text, exact=False).first
                    await loc.wait_for(state="visible", timeout=1500)
                    box = await loc.bounding_box()
                    if box:
                        cx = int(box["x"] + box["width"]/2)
                        cy = int(box["y"] + box["height"]/2)
                except Exception:
                    pass

            if cx is None and params.x is not None and params.y is not None:
                vw = await page.evaluate("() => window.innerWidth")
                vh = await page.evaluate("() => window.innerHeight")
                cx = int(max(0, min(1, params.x)) * vw)
                cy = int(max(0, min(1, params.y)) * vh)

            # Fallback: center of viewport
            if cx is None or cy is None:
                vw = await page.evaluate("() => window.innerWidth")
                vh = await page.evaluate("() => window.innerHeight")
                cx, cy = int(vw/2), int(vh/2)

            async def _focus_point():
                try:
                    await page.mouse.move(cx, cy)
                    await page.mouse.click(cx, cy)   # sets focus point in Street View
                    await page.wait_for_timeout(120)
                except Exception:
                    pass

            async def _zoom_wheel():
                count = 0
                for _ in range(max(1, params.steps)):
                    await page.mouse.move(cx, cy)
                    await page.mouse.wheel(0, -400)  # negative = zoom in
                    count += 1
                    await page.wait_for_timeout(params.step_wait_ms)
                return count

            async def _zoom_double_click():
                count = 0
                for _ in range(max(1, params.steps)):
                    await page.mouse.dblclick(cx, cy, delay=30)
                    count += 1
                    await page.wait_for_timeout(params.step_wait_ms)
                return count

            async def _zoom_controls():
                count = 0
                # controls usually live on the right; ensure they’re visible
                try:
                    vw = await page.evaluate("() => window.innerWidth")
                    vh = await page.evaluate("() => window.innerHeight")
                    await page.mouse.move(vw-40, vh/2)
                    await page.wait_for_timeout(120)
                except Exception:
                    pass
                for _ in range(max(1, params.steps)):
                    clicked = False
                    for sel in [
                        "button[aria-label*='Zoom in']",
                        "[role='button'][aria-label*='Zoom in']",
                        "div[aria-label*='Zoom in']",
                    ]:
                        try:
                            btn = page.locator(sel).first
                            if await btn.count() > 0 and await btn.is_visible():
                                await btn.click()
                                clicked = True
                                break
                        except Exception:
                            pass
                    if not clicked:
                        break
                    count += 1
                    await page.wait_for_timeout(params.step_wait_ms)
                return count

            if params.center_first:
                await _focus_point()

            strategy = params.strategy
            performed = 0
            logs = []

            async def run_strategy(name):
                nonlocal performed
                if name == "wheel":
                    c = await _zoom_wheel(); logs.append(("wheel", c)); performed += c
                elif name == "double_click":
                    c = await _zoom_double_click(); logs.append(("double_click", c)); performed += c
                elif name == "controls":
                    c = await _zoom_controls(); logs.append(("controls", c)); performed += c

            if strategy == "auto":
                # Try wheel → double-click → controls
                await run_strategy("wheel")
                if performed == 0:
                    await run_strategy("double_click")
                if performed == 0:
                    await run_strategy("controls")
            else:
                await run_strategy(strategy)
            
            msg = f"🔎 Zoomed in at ({cx},{cy}) using {', '.join(f'{n}×{c}' for n,c in logs if c)}."
            # logger.info(msg)
            return ActionResult(
                extracted_content=f"🔎 Zoomed in at ({cx},{cy}) using {', '.join(f'{n}×{c}' for n,c in logs if c)}.",
                data={"x": cx, "y": cy, "log": logs},
                include_in_memory=True,
            )

        """async def sv_click_timeline_arrow(params: TimelineArrowParams, browser):
            
            page = await browser.get_current_page()

            async def _open_timeline():
                # make sure the timeline/time widget is open (if available)
                for sel in [
                    "button[aria-label*='See more dates']",
                    "button[aria-label*='Time']",
                    "button[aria-label*='Timeline']",
                    "div[aria-label*='See more dates']",
                ]:
                    try:
                        loc = page.locator(sel).first
                        if await loc.count() > 0 and await loc.is_visible():
                            await loc.click()
                            await page.wait_for_timeout(250)
                            return
                    except Exception:
                        pass

            async def _reveal_arrows():
                # arrows sometimes appear on hover near the bottom timeline strip
                try:
                    vw = await page.evaluate("() => window.innerWidth")
                    vh = await page.evaluate("() => window.innerHeight")
                    await page.mouse.move(int(vw/2), max(0, vh - 40))
                    await page.wait_for_timeout(150)
                except Exception:
                    pass

            def _want_right():
                d = params.direction.lower()
                return d in ("right", "later", "next")

            def _selectors_for_direction():
                # try most specific first, then fallbacks
                if _want_right():
                    return [
                        "g-scrolling-carousel button[aria-label*='Next']",
                        "button[aria-label*='Next']",
                        "button[aria-label*='Later']",
                        "[role='button'][aria-label*='Next']",
                        "[role='button'][aria-label*='Later']",
                    ]
                else:
                    return [
                        "g-scrolling-carousel button[aria-label*='Previous']",
                        "button[aria-label*='Previous']",
                        "button[aria-label*='Earlier']",
                        "[role='button'][aria-label*='Previous']",
                        "[role='button'][aria-label*='Earlier']",
                    ]

            async def _read_date_label() -> str:
                try:
                    txt = await page.locator("[aria-live='polite']").first.text_content(timeout=600)
                    return (txt or "").strip()
                except Exception:
                    return ""

            # (1) optionally open the timeline
            if params.ensure_timeline_open:
                await _open_timeline()

            # (2) reveal the arrows if they’re hover-gated
            await _reveal_arrows()

            # (3) locate the arrow button we need
            arrow = None
            for sel in _selectors_for_direction():
                try:
                    loc = page.locator(sel).last if _want_right() else page.locator(sel).first
                    if await loc.count() > 0:
                        arrow = loc
                        break
                except Exception:
                    continue

            clicks = 0
            before = await _read_date_label()

            # (4) click the arrow repeatedly
            if arrow:
                for _ in range(max(1, params.steps)):
                    try:
                        await arrow.click()
                        clicks += 1
                        await page.wait_for_timeout(params.wait_ms)
                    except Exception:
                        # try re-revealing and continue
                        await _reveal_arrows()
                        try:
                            await arrow.click()
                            clicks += 1
                            await page.wait_for_timeout(params.wait_ms)
                        except Exception:
                            break
            else:
                # (5) fallback: focus slider and use keyboard arrows
                try:
                    slider = page.locator("[role='slider']").first
                    await slider.wait_for(state="visible", timeout=1200)
                    await slider.focus()
                    key = "ArrowRight" if _want_right() else "ArrowLeft"
                    for _ in range(max(1, params.steps)):
                        await page.keyboard.press(key)
                        clicks += 1
                        await page.wait_for_timeout(params.wait_ms)
                except Exception:
                    pass

            after = await _read_date_label()

            return ActionResult(
                extracted_content=(
                    f"➡️ Timeline arrow clicked {clicks}x towards "
                    f"{'RIGHT/later' if _want_right() else 'LEFT/earlier'}."
                    f" Date label: '{before}' → '{after}'"
                ),
                data={"clicks": clicks, "direction": "right" if _want_right() else "left", "before": before, "after": after},
                include_in_memory=True,
            )
"""
        # --- Street View helpers ---------------------------------------------
        async def _sv_open_timeline(self, page):
            for sel in [
                "button[aria-label*='See more dates']",
                "button[aria-label*='Time']",
                "button[aria-label*='Timeline']",
                "div[aria-label*='See more dates']",
            ]:
                try:
                    loc = page.locator(sel).first
                    if await loc.count() > 0:
                        await loc.click()
                        await page.wait_for_timeout(250)
                        return
                except Exception:
                    pass

        async def _sv_get_slider_vals(self, page):
            try:
                slider = page.locator("[role='slider']").first
                await slider.wait_for(state="visible", timeout=1200)
                vmin = await slider.get_attribute("aria-valuemin")
                vnow = await slider.get_attribute("aria-valuenow")
                vmax = await slider.get_attribute("aria-valuemax")
                to_i = lambda x: int(x) if x is not None else None
                return slider, to_i(vmin), to_i(vnow), to_i(vmax)
            except Exception:
                return None, None, None, None

        async def _sv_read_date_label(self, page) -> str:
            try:
                txt = await page.locator("[aria-live='polite']").first.text_content(timeout=600)
                return (txt or "").strip()
            except Exception:
                return ""

        async def _sv_click_previous_until_stable(self, page, max_clicks: int = 80):
            last = await self._sv_read_date_label(page)
            clicks = 0
            for _ in range(max_clicks):
                clicked = False
                for sel in ["button[aria-label*='Previous']", "button[aria-label*='Earlier']"]:
                    try:
                        btn = page.locator(sel).last
                        if await btn.count() > 0 and not await btn.is_disabled():
                            await btn.click()
                            clicked = True
                            clicks += 1
                            await page.wait_for_timeout(200)
                            break
                    except Exception:
                        pass
                cur = await self._sv_read_date_label(page)
                if not clicked or cur == last:
                    break
                last = cur
            return clicks, last

        async def _sv_drag_fully_left(self, page):
            try:
                slider = page.locator("[role='slider']").first
                box = await slider.bounding_box()
                if not box:
                    return False
                y = box["y"] + box["height"] / 2
                await page.mouse.move(box["x"] + box["width"] * 0.8, y)
                await page.mouse.down()
                await page.mouse.move(box["x"] - box["width"] * 2, y, steps=20)
                await page.mouse.up()
                await page.wait_for_timeout(250)
                return True
            except Exception:
                return False

        async def _sv_verify_earliest(self, page) -> dict:
            """
            Returns: {"earliest": bool, "label": str, "vmin": int|None, "vnow": int|None, "clicks": int}
            """
            await self._sv_open_timeline(page)

            # Try 'Home' (often jumps to earliest)
            try:
                slider = page.locator("[role='slider']").first
                await slider.wait_for(state="visible", timeout=1500)
                await slider.focus()
                await page.keyboard.press("Home")
                await page.wait_for_timeout(250)
            except Exception:
                pass

            clicks, label_after_clicks = await self._sv_click_previous_until_stable(page)

            slider, vmin, vnow, vmax = await self._sv_get_slider_vals(page)

            # If still not at earliest, drag the handle hard left
            if slider is not None and vmin is not None and vnow is not None and vnow > vmin:
                if await self._sv_drag_fully_left(page):
                    _, vmin, vnow, _ = await self._sv_get_slider_vals(page)

            label_final = await self._sv_read_date_label(page)
            earliest = (vmin is not None and vnow is not None and vnow == vmin)

            return {
                "earliest": earliest,
                "label": label_final or label_after_clicks,
                "vmin": vmin,
                "vnow": vnow,
                "clicks": clicks,
            }


        #=== End of Custom Code ===#

        async def _is_google_captcha(page) -> bool:
            html = (await page.content()).lower()
            if "detected unusual traffic" in html or "recaptcha" in html:
                return True
            # 更稳：检查 iframe
            for f in page.frames:
                u = (f.url or "").lower()
                if "recaptcha" in u or "bframe" in u:
                    return True
            return False
        
        async def search_duckduckgo(params: SearchQuery, browser):
            """Search DuckDuckGo for a query in the current tab.

            Params (object):
            - query: str

            Returns: ActionResult("Searched for \"<query>\" in DuckDuckGo")"""

            page = await browser.get_current_page()
            import urllib.parse
            q = urllib.parse.quote(params.query)
            await page.goto(f'https://duckduckgo.com/?q={q}')
            await page.wait_for_load_state()
            msg = f'🔍  Searched for "{params.query}" in DuckDuckGo'
            # logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)
        
        # Basic Navigation Actions
        async def search_google(params: SearchQuery, browser):
            """Search Google (English, non-redirect) and show results; auto-accept consent.
            Falls back to DuckDuckGo if CAPTCHA is detected.

            Params (object):
            - query: str

            Returns: ActionResult("Searched for \"<query>\" in Google" or CAPTCHA fallback note)"""

            page = await browser.get_current_page()

            # 1) 先去 NCR，避免地区重定向
            # await browser.navigate_to('https://www.google.com/ncr?hl=en')
            await page.goto('https://www.google.com/ncr?hl=en', wait_until='domcontentloaded')
            #await page.wait_for_load_state('networkidle')

            # 2) 若出现同意/隐私提示，自动接受
            try:
                if "consent" in page.url or await page.locator('form[action*="consent"]').count() > 0:
                    # 常见按钮文案覆盖
                    for btn_text in ["I agree", "Accept all", "Agree to all", "同意", "接受全部"]:
                        loc = page.get_by_role("button", name=btn_text, exact=False)
                        if await loc.count() > 0:
                            await loc.first.click()
                            await page.wait_for_load_state('networkidle')
                            break
            except Exception:
                pass

            # 3) 真正搜索（避免 udm=14 引发不稳定，强制英文）
            from urllib.parse import quote
            encoded = quote(params.query)
            await page.goto(f'https://www.google.com/search?q={encoded}&hl=en&num=10', wait_until='domcontentloaded')
            #await page.wait_for_load_state('networkidle')
            #await browser.navigate_to(f'https://www.google.com/search?q={encoded}&hl=en&num=10')

            # 4) 等搜索结果区域出现（多种选择器兜底）
            for sel in ['div#search', 'div[role="main"]', '#rso']:
                try:
                    await page.wait_for_selector(sel, timeout=8000)
                    break
                except Exception:
                    continue

            # 5) 稳定一下（给截图留点缓冲）
            # await page.wait_for_timeout(500)

            if await _is_google_captcha(page):
                msg = "🛑 Google CAPTCHA detected — falling back to DuckDuckGo."
                # logger.info(msg)
                # 直接改用 DuckDuckGo
                await search_duckduckgo(params, browser)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            
            msg = f'🔍  Searched for "{params.query}" in Google'
            # logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        async def open_orcid_if_present(params: OrcidQuery, browser):
            """Open an ORCID profile if an ORCID iD is present in the text.

                Params (object):
                - query: str — any text that may contain an ORCID iD (0000-0000-0000-0000).

                Returns: ActionResult("Opened ORCID profile: <url>" | "No ORCID id found in query.")"""

            page = await browser.get_current_page()
            import re
            m = re.search(r"\b(\d{4}-\d{4}-\d{4}-\d{3}[0-9X])\b", params.query)
            if m:
                orcid = m.group(1)
                url = f"https://orcid.org/{orcid}"
                await page.goto(url, wait_until='domcontentloaded')
                await page.wait_for_load_state('networkidle')
                msg = f"🔗  Opened ORCID profile: {url}"
                # logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            return ActionResult(extracted_content="No ORCID id found in query.", include_in_memory=False)

        async def go_to_url(params:GoToUrlAction, browser):
            """Navigate the current tab to a URL and wait for load.

            Params (object):
            - url: str

            Returns: ActionResult("Navigated to <url>")"""

            """page = await browser.get_current_page()
            await page.goto(params.url)
            await page.wait_for_load_state()"""
            await browser.navigate_to(params.url)
            msg = f'🔗  Navigated to {params.url}'
            # logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        async def find_archive_url(params: FindArchiveURLAction, browser):
            """Look up a Wayback Machine snapshot for a URL (optionally at a given date).

                Params (object):
                - url: str
                - date: str — YYYYMMDDhhmmss (Wayback timestamp) or empty to auto-closest.

                Returns: ActionResult("Found archive URL: <snapshot_url>" | "No archive URL found...")"""

            no_timestamp_url = f"https://archive.org/wayback/available?url={params.url}"
            archive_url = no_timestamp_url + f"&timestamp={params.date}"

            response = requests.get(archive_url).json()
            response_notimestamp = requests.get(no_timestamp_url).json()

            if "archived_snapshots" in response and "closest" in response["archived_snapshots"]:
                closest = response["archived_snapshots"]["closest"]
                # logger.info(f"Archive found! {closest}")

            elif "archived_snapshots" in response_notimestamp and "closest" in response_notimestamp[
                "archived_snapshots"]:
                closest = response_notimestamp["archived_snapshots"]["closest"]
                # logger.info(f"Archive found! {closest}")
            else:
                return ActionResult(
                    extracted_content = "❌  No archive URL found for the given URL and date.",
                    include_in_memory = True,
                )

            target_url = closest["url"]
            return ActionResult(
                extracted_content = f"🕰️  Found archive URL: {target_url}",
                include_in_memory = True,
            )

        async def go_back(_, browser):
            """Go back to the previous page in history.

            Params: {}

            Returns: ActionResult("Navigated back")"""

            await browser.go_back()
            msg = '🔙  Navigated back'
            # logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        # wait for x seconds
        async def wait(seconds: int = 3):
            """Pause the agent for a number of seconds.

            Params:
            - seconds: int (default: 3)

            Returns: ActionResult("Waiting for <seconds> seconds")"""

            msg = f'🕒  Waiting for {seconds} seconds'
            # logger.info(msg)
            await asyncio.sleep(seconds)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        # Element Interaction Actions
        async def click_element_by_index(params:ClickElementAction, browser):
            """Click an element by the agent's indexed selector map; handles new tabs/downloads.

                Params (object):
                - index: int — element index from the selector map.

                Returns: ActionResult with click/download message (switches to new tab if opened)"""

            session = await browser.get_session()

            if params.index not in await browser.get_selector_map():
                raise Exception(f'Element with index {params.index} does not exist - retry or use alternative actions')

            element_node = await browser.get_dom_element_by_index(params.index)
            initial_pages = len(session.context.pages)

            # if element has file uploader then dont click
            if await browser.is_file_uploader(element_node):
                msg = f'Index {params.index} - has an element which opens file upload dialog. To upload files please use a specific function to upload files '
                # logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

            msg = None

            try:
                download_path = await browser._click_element_node(element_node)
                if download_path:
                    msg = f'💾  Downloaded file to {download_path}'
                else:
                    msg = f'🖱️  Clicked button with index {params.index}: {element_node.get_all_text_till_next_clickable_element(max_depth=2)}'

                # logger.info(msg)
                # logger.debug(f'Element xpath: {element_node.xpath}')
                if len(session.context.pages) > initial_pages:
                    new_tab_msg = 'New tab opened - switching to it'
                    msg += f' - {new_tab_msg}'
                    # logger.info(new_tab_msg)
                    await browser.switch_to_tab(-1)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                # logger.warning(f'Element not clickable with index {params.index} - most likely the page changed')
                return ActionResult(error=str(e))

        async def input_text(params:InputTextAction, browser, has_sensitive_data: bool = False):
            """Type text into an element by index.

            Params (object):
            - index: int — element index from the selector map.
            - text: str — text to input.
            Optional:
            - has_sensitive_data: bool (default: False) — if True, omit raw text in logs.

            Returns: ActionResult("Input <text> into index <n>" or "Input sensitive data...")"""

            if params.index not in await browser.get_selector_map():
                raise Exception(f'Element index {params.index} does not exist - retry or use alternative actions')

            element_node = await browser.get_dom_element_by_index(params.index)
            await browser._input_text_element_node(element_node, params.text)
            if not has_sensitive_data:
                msg = f'⌨️  Input {params.text} into index {params.index}'
            else:
                msg = f'⌨️  Input sensitive data into index {params.index}'
            # logger.info(msg)
            # logger.debug(f'Element xpath: {element_node.xpath}')
            return ActionResult(extracted_content=msg, include_in_memory=True)

        # Save PDF

        async def save_pdf(browser):
            """Save the current page as a PDF (A4) to disk; filename based on URL.

            Params: {}

            Returns: ActionResult("Saving page with URL <url> as PDF to ./<file>.pdf")"""

            page = await browser.get_current_page()
            short_url = re.sub(r'^https?://(?:www\.)?|/$', '', page.url)
            slug = re.sub(r'[^a-zA-Z0-9]+', '-', short_url).strip('-').lower()
            sanitized_filename = f'{slug}.pdf'

            await page.emulate_media(media='screen')
            await page.pdf(path=sanitized_filename, format='A4', print_background=False)
            msg = f'Saving page with URL {page.url} as PDF to ./{sanitized_filename}'
            # logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)


        async def video_viewer(params: VideoAction, browser):
            """Open a local YouTube player and jump to a timestamp.

                Params (object):
                - action: "jump"
                - video_url: str — full YouTube URL or 11-char id.
                - time: int — seconds to seek to.

                Returns: ActionResult("Jumped to <t>s via local player. Current time: <t>s")"""

            page = await browser.get_current_page()

            action = params.action

            def extract_video_id(url: str) -> str | None:
                import re
                # 支持 watch?v=、shorts/、youtu.be/
                m = re.search(r"(?:v=|/shorts/|youtu\.be/)([0-9A-Za-z_-]{11})", url)
                return m.group(1) if m else (url if re.fullmatch(r"[0-9A-Za-z_-]{11}", url) else None)

            if action == "jump":

                if getattr(params, 'video_url', None) is None:
                    return ActionResult(
                        error="❌  Video URL is required for jump action.",
                        include_in_memory=True,
                    )

                if getattr(params, 'time', None) is None:
                    return ActionResult(
                        error="❌  Time is required for jump action.",
                        include_in_memory=True,
                    )

                # video_url = params.video_url
                # time = params.time

                # jumped_url = f"{video_url}&t={time}s"

                # await page.goto(jumped_url)
                # await page.wait_for_load_state("networkidle")
                # await page.evaluate("document.querySelector('video').play()")
                # await page.wait_for_timeout(100)
                # await page.evaluate("document.querySelector('video').pause()")
                # await page.wait_for_timeout(100)

                # current_time = await page.evaluate("document.querySelector('video').currentTime")
                # await page.wait_for_timeout(1000)

                # def extract_video_id(input_str):
                #     if re.fullmatch(r"[0-9A-Za-z_-]{11}", input_str):
                #         return input_str
                #     match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", input_str)
                #     return match.group(1) if match else None

                vid = extract_video_id(params.video_url)
                if not vid:
                    return ActionResult(error="❌  Could not parse YouTube video id from URL.", include_in_memory=True)
                
                local_player = f"http://localhost:{self.http_server_port}/video_viewer/player.html?v={vid}"
                
                browser.navigate_to(local_player)
                #await page.goto(local_player, wait_until="domcontentloaded")
                #await page.wait_for_load_state("networkidle")
                # await page.pause()
                
                # 等待 IFrame API ready（player.html 会把 window.playerReady 置 true）
                await page.wait_for_function("window.playerReady === true", timeout=30000)
                # 允许自动播放（先静音）
                await page.evaluate("() => { try { window.player.mute(); } catch(e){} }")
                
                # 跳转并轻触播放以解锁画面（再暂停，读取时间）
                t = int(params.time)
                await page.evaluate(f"() => window.ytCtl.seek({t});")
                await page.evaluate("() => window.ytCtl.play();")
                await page.wait_for_timeout(400)
                await page.evaluate("() => window.ytCtl.pause();")
                current = await page.evaluate("() => window.ytCtl.now()")
                # # find the iframe with player.html
                # player_frame = next(f for f in page.frames if local_video_server_url == f.url)
                # print(player_frame.name, player_frame.url)
                
                # await player_frame.wait_for_function("window.ytCtl && window.playerReady === true")
                # await asyncio.sleep(1)
                
                # # await player_frame.evaluate("() => window.ytCtl.play()")
                
                # await player_frame.evaluate(f"() => window.ytCtl.seek({time})")
                
                # current_time = await player_frame.evaluate("() => window.ytCtl.now()")

                # msg = f"🎥  Jumped to {time} seconds in the video with URL {video_url}. Current time is {current_time} seconds."
                msg = f"🎥  Jumped to {t}s via local player. Current time: {current:.2f}s"
                # logger.info(msg)

                return ActionResult(extracted_content=msg, include_in_memory=True)


        async def pdf_viewer(params: PDFAction, browser):
            """Work with PDFs via pdf.js/native viewers: download, scroll, jump, search.

                Params (object):
                - action: "download" | "scroll_down" | "scroll_up" | "jump" | "search"

                For action="download":
                - pdf_url: str (required)
                - save_name: str (optional; default "downloaded_file.pdf")

                For action="scroll_down" | "scroll_up":
                - pixels: int (optional) — if omitted, use viewer controls/PageUp/Down

                For action="jump":
                - page_number: int (required)

                For action="search":
                - search_text: str (required)

                Returns: ActionResult with a human-readable status message."""

            page = await browser.get_current_page()

            # -------- helpers --------
            def _download_pdf(url: str, save_path: str, timeout: int = 60):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # 若 URL 含空格，先做最小合法化
                if " " in url:
                    # 只对 path/query 进行编码，scheme+host 不动
                    parts = urlsplit(url)
                    url = urlunsplit((
                        parts.scheme,
                        parts.netloc,
                        quote(parts.path),
                        quote(parts.query, safe="=&?"),
                        parts.fragment
                    ))
                with requests.get(url, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    with open(save_path, "wb") as f:
                        for chunk in r.iter_content(8192):
                            if chunk:
                                f.write(chunk)
                return save_path

            async def _download_pdf_async(url: str, save_path: str):
                # ✨ 避免阻塞事件循环
                return await asyncio.to_thread(_download_pdf, url, save_path)
            
            def _extract_pdf_from_viewer_url(url: str) -> str | None:
                """从 pdf.js 的 viewer.html?file=... 里提取真实 PDF URL"""
                if "viewer.html" not in url:
                    return None
                q = urlparse(url).query
                params = parse_qs(q)
                file_vals = params.get("file")
                if not file_vals:
                    return None
                # 只取第一个，必要时再 unquote
                return file_vals[0]
    
            async def _open_local_pdfviewer(save_name: str) -> bool:
                # 本地 pdf.js viewer
                local_url = f"http://localhost:{self.http_server_port}/pdf_viewer/viewer.html?file=../local/{quote(save_name)}"
                try:
                    await page.goto(local_url, wait_until="domcontentloaded", timeout=20000)
                    await page.wait_for_selector("#pageNumber", timeout=20000)
                    return True
                except Exception:
                    return False

            async def _open_online_pdfviewer(pdf_url: str) -> bool:
                # 在线 pdf.js viewer
                remote = "https://mozilla.github.io/pdf.js/web/viewer.html?file=" + quote(pdf_url, safe='')
                try:
                    await page.goto(remote, wait_until="domcontentloaded", timeout=25000)
                    await page.wait_for_selector("#pageNumber", timeout=25000)
                    return True
                except Exception:
                    return False

            def _with_page_fragment(url: str, fragment: str) -> str:
                parts = list(urlsplit(url))
                # 保留已有片段的基础上设置/覆盖
                parts[4] = fragment
                return urlunsplit(parts)

            async def _is_pdfjs() -> bool:
                # pdf.js 里一定有 #pageNumber / #viewerContainer
                try:
                    await page.wait_for_selector("#pageNumber", timeout=1000)
                    return True
                except Exception:
                    return False

            async def _jump_pdfjs(page_no: int):
                await page.fill("#pageNumber", str(page_no))
                await page.keyboard.press("Enter")
                await asyncio.sleep(0.4)

            async def _jump_native(page_no: int):
                base = page.url.split("#")[0]
                await page.goto(_with_page_fragment(base, f"page={page_no}"), wait_until="load")

            # -------- main --------
            action = params.action

            # DOWNLOAD
            if action == "download":
                if not getattr(params, 'pdf_url', None):
                    return ActionResult(error="❌  PDF URL is required for download action.", include_in_memory=True)

                pdf_url = params.pdf_url
                save_name = params.save_name or "downloaded_file.pdf"
                save_path = os.path.join(self.http_save_path, save_name)

                try:
                    _download_pdf(pdf_url, save_path)
                except Exception as e:
                    return ActionResult(error=f"❌  PDF download failed: {e}", include_in_memory=True)

                # 优先尝试本地 pdf.js；不行则在线 pdf.js；再不行直接打开原 PDF
                opened = await _open_local_pdfviewer(save_name)
                mode = "local-pdfjs" if opened else None
                if not opened:
                    opened = await _open_online_pdfviewer(pdf_url)
                    mode = "online-pdfjs" if opened else None
                if not opened:
                    await page.goto(pdf_url, wait_until="load")
                    mode = "native"

                msg = f"📄  Downloaded PDF → {save_path}. Open mode: {mode}"
                # logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

            # SCROLL DOWN
            if action == "scroll_down":
                pixels = getattr(params, 'pixels', None)
                if await _is_pdfjs():
                    if pixels:
                        await page.evaluate(f"document.querySelector('#viewerContainer').scrollTop += {int(pixels)}")
                    else:
                        await page.click("#next")
                else:
                    # 原生预览器：用 PageDown 兜底
                    await page.keyboard.press("PageDown")
                await page.wait_for_load_state()
                return ActionResult(extracted_content="📄  Scrolled down.", include_in_memory=True)

            # SCROLL UP
            if action == "scroll_up":
                pixels = getattr(params, 'pixels', None)
                if await _is_pdfjs():
                    if pixels:
                        await page.evaluate(f"document.querySelector('#viewerContainer').scrollTop -= {int(pixels)}")
                    else:
                        await page.click("#previous")
                else:
                    await page.keyboard.press("PageUp")
                await page.wait_for_load_state()
                return ActionResult(extracted_content="📄  Scrolled up.", include_in_memory=True)

            # JUMP
            if action == "jump":
                if getattr(params, 'page_number', None) is None:
                    return ActionResult(error="❌  Page number is required for jump action.", include_in_memory=True)

                page_no = int(params.page_number)
                if await _is_pdfjs():
                    await _jump_pdfjs(page_no)
                    await page.wait_for_load_state()
                    msg = f"📄  Jumped to page {page_no} (pdf.js)."
                else:
                    await _jump_native(page_no)
                    msg = f"📄  Jumped to page {page_no} (native)."
                # logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

            # SEARCH
            if action == "search":
                if not getattr(params, 'search_text', None):
                    return ActionResult(error="❌  Search text is required for search action.", include_in_memory=True)

                query = params.search_text

                # 若非 pdf.js，自动切到在线 pdf.js 再搜
                if not await _is_pdfjs():
                    current_pdf = page.url.split("#")[0]
                    real_pdf = _extract_pdf_from_viewer_url(current_pdf)
                    if real_pdf:
                        switched = await _open_online_pdfviewer(real_pdf)
                    else:
                        #  不在 viewer.html，直接尝试把“当前 URL 当作 PDF”送入在线 pdf.js
                        switched = await _open_online_pdfviewer(current_pdf)
                    if not switched:
                        # 兜底：用浏览器原生查找（没有计数）
                        try:
                            await page.keyboard.press("Control+f")
                            # 常见的 find 框定位可能各浏览器不同，这里仅触发快捷键 + 回车
                            await page.keyboard.type(query)
                            await page.keyboard.press("Enter")
                            msg = f"🔎 Used browser native find for '{query}' (not PDF.js)."
                            # logger.info(msg)
                            return ActionResult(extracted_content=msg, include_in_memory=True)
                        except Exception as e:
                            return ActionResult(error=f"❌  Search requires pdf.js viewer and native find failed: {e}", include_in_memory=True)

                # pdf.js 搜索
                await page.click("#viewFind")
                await page.fill("#findInput", query)
                await page.keyboard.press("Enter")
                await page.wait_for_timeout(300)
                try:
                    count_text = await page.inner_text("#findResultsCount")
                except Exception:
                    count_text = ""
                m = re.search(r"of (\d+)", count_text)
                if not m:
                    msg = f"No matches found for '{query}'."
                else:
                    total = m.group(1)
                    await page.click("#findNext")
                    msg = f"📄  Found {total} matches for '{query}' and moved to first."
                # logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

            return ActionResult(error=f"Unknown pdf action: {action}", include_in_memory=True)

        # Tab Management Actions
        async def switch_tab(params:SwitchTabAction, browser):
            """Switch to an existing tab by page id (index).

            Params (object):
            - page_id: int — tab index (use -1 for last tab).

            Returns: ActionResult("Switched to tab <page_id>")"""

            await browser.switch_to_tab(params.page_id)
            # Wait for tab to be ready
            page = await browser.get_current_page()
            await page.wait_for_load_state()
            msg = f'🔄  Switched to tab {params.page_id}'
            # logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        async def open_tab(params:OpenTabAction, browser):
            """Open a new tab with a URL.

            Params (object):
            - url: str

            Returns: ActionResult("Opened new tab with <url>")"""

            await browser.create_new_tab(params.url)
            msg = f'🔗  Opened new tab with {params.url}'
            # logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        async def close_tab(params:CloseTabAction, browser):
            """Close a tab by page id (index).

                Params (object):
                - page_id: int

                Returns: ActionResult("Closed tab #<page_id> with url <url>")"""

            await browser.switch_to_tab(params.page_id)
            page = await browser.get_current_page()
            url = page.url
            await page.close()
            msg = f'❌  Closed tab #{params.page_id} with url {url}'
            # logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        # Content Actions
        async def extract_content(
                goal: str, should_strip_link_urls: bool, browser, page_extraction_llm
        ):
            """Extract page content relevant to a goal (markdownified DOM + iframes). Uses the configured LLM to format output as JSON; falls back to raw content if LLM fails.

            Params:
            - goal: str — what to extract / summarize.
            - should_strip_link_urls: bool — if True, remove <a> and <img> href/src from text.

            Returns: ActionResult(extracted_content=<string from LLM or raw markdown>)"""

            page = await browser.get_current_page()
            import markdownify

            strip = []
            if should_strip_link_urls:
                strip = ['a', 'img']

            content = markdownify.markdownify(await page.content(), strip=strip)

            # manually append iframe text into the content so it's readable by the LLM (includes cross-origin iframes)
            for iframe in page.frames:
                if iframe.url != page.url and not iframe.url.startswith('data:'):
                    content += f'\n\nIFRAME {iframe.url}:\n'
                    content += markdownify.markdownify(await iframe.content())

            prompt = 'Your task is to extract the content of the page. You will be given a page and a goal and you should extract all relevant information around this goal from the page. If the goal is vague, summarize the page. Respond in json format. Extraction goal: {goal}, Page: {page}'
            template = PromptTemplate(input_variables=['goal', 'page'], template=prompt)
            try:
                output = await page_extraction_llm.ainvoke(template.format(goal=goal, page=content))
                msg = f'📄  Extracted from page\n: {output.content}\n'
                # logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                # logger.debug(f'Error extracting content: {e}')
                msg = f'📄  Extracted from page\n: {content}\n'
                # logger.info(msg)
                return ActionResult(extracted_content=msg)

        async def scroll_down(params:ScrollAction, browser):
            """Scroll the page down.

            Params (object):
            - amount: int | null — pixels to scroll; if omitted, scroll half a viewport.

            Returns: ActionResult("Scrolled down the page by <amount>")"""

            page = await browser.get_current_page()
            if params.amount is not None:
                await page.evaluate(f'window.scrollBy(0, {params.amount});')
                amount = f'{params.amount} pixels'
            else:
                await page.evaluate('window.scrollBy(0, window.innerHeight / 2);')
                amount = 'half a page'

            msg = f'🔍  Scrolled down the page by {amount}'
            # logger.info(msg)
            return ActionResult(
                extracted_content=msg,
                include_in_memory=True,
            )

        # scroll up
        async def scroll_up(params:ScrollAction, browser):
            """Scroll the page up.

            Params (object):
            - amount: int | null — pixels to scroll; if omitted, scroll half a viewport.

            Returns: ActionResult("Scrolled up the page by <amount>")"""

            page = await browser.get_current_page()
            if params.amount is not None:
                await page.evaluate(f'window.scrollBy(0, -{params.amount});')
                amount = f'{params.amount} pixels'
            else:
                await page.evaluate('window.scrollBy(0, -window.innerHeight / 2);')
                amount = 'half a page'

            msg = f'🔍  Scrolled up the page by {amount}'
            # logger.info(msg)
            return ActionResult(
                extracted_content=msg,
                include_in_memory=True,
            )

        # send keys
        async def send_keys(params: KeysParam, browser):
            """Send key presses to the page (Playwright key names).

                Params (object):
                - keys: str | List[str] — key or list of keys, e.g. "Enter", "Control+f", ["ArrowDown","Enter"]

                Returns: ActionResult("Sent keys: <keys>")"""

            page = await browser.get_current_page()

            try:
                await page.keyboard.press(params.keys)
            except Exception as e:
                if 'Unknown key' in str(e):
                    # loop over the keys and try to send each one
                    for key in params.keys:
                        try:
                            await page.keyboard.press(key)
                        except Exception as e:
                            # logger.debug(f'Error sending key {key}: {str(e)}')
                            raise e
                else:
                    raise e
            msg = f'⌨️  Sent keys: {params.keys}'
            # logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        async def scroll_to_text(text: str, browser):  # type: ignore
            """Scroll until an element containing the given text is in view.

            Params:
            - text: str — visible text to locate.

            Returns: ActionResult("Scrolled to text: <text>" | "Text '<text>' not found...")"""

            page = await browser.get_current_page()
            try:
                # Try different locator strategies
                locators = [
                    page.get_by_text(text, exact=False),
                    page.locator(f'text={text}'),
                    page.locator(f"//*[contains(text(), '{text}')]"),
                ]

                for locator in locators:
                    try:
                        # First check if element exists and is visible
                        if await locator.count() > 0 and await locator.first.is_visible():
                            await locator.first.scroll_into_view_if_needed()
                            await asyncio.sleep(0.5)  # Wait for scroll to complete
                            msg = f'🔍  Scrolled to text: {text}'
                            # logger.info(msg)
                            return ActionResult(extracted_content=msg, include_in_memory=True)
                    except Exception as e:
                        # logger.debug(f'Locator attempt failed: {str(e)}')
                        continue

                msg = f"Text '{text}' not found or not visible on page"
                # logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

            except Exception as e:
                msg = f"Failed to scroll to text '{text}': {str(e)}"
                # logger.error(msg)
                return ActionResult(error=msg, include_in_memory=True)

        async def get_dropdown_options(index: int, browser) -> ActionResult:
            """List all options for a native <select> (frame-aware). Use exact text in selection.

            Params:
            - index: int — element index of the <select>.

            Returns: ActionResult(extracted_content="<idx>: text=\"...\"\\n...", include_in_memory=True)"""

            page = await browser.get_current_page()
            selector_map = await browser.get_selector_map()
            dom_element = selector_map[index]

            try:
                # Frame-aware approach since we know it works
                all_options = []
                frame_index = 0

                for frame in page.frames:
                    try:
                        options = await frame.evaluate(
                            """
                            (xpath) => {
                                const select = document.evaluate(xpath, document, null,
                                    XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                if (!select) return null;

                                return {
                                    options: Array.from(select.options).map(opt => ({
                                        text: opt.text, //do not trim, because we are doing exact match in select_dropdown_option
                                        value: opt.value,
                                        index: opt.index
                                    })),
                                    id: select.id,
                                    name: select.name
                                };
                            }
                        """,
                            dom_element.xpath,
                        )

                        if options:
                            # logger.debug(f'Found dropdown in frame {frame_index}')
                            # logger.debug(f'Dropdown ID: {options["id"]}, Name: {options["name"]}')

                            formatted_options = []
                            for opt in options['options']:
                                # encoding ensures AI uses the exact string in select_dropdown_option
                                encoded_text = json.dumps(opt['text'])
                                formatted_options.append(f'{opt["index"]}: text={encoded_text}')

                            all_options.extend(formatted_options)

                    except Exception as frame_e:
                        pass
                        # logger.debug(f'Frame {frame_index} evaluation failed: {str(frame_e)}')

                    frame_index += 1

                if all_options:
                    msg = '\n'.join(all_options)
                    msg += '\nUse the exact text string in select_dropdown_option'
                    # logger.info(msg)
                    return ActionResult(extracted_content=msg, include_in_memory=True)
                else:
                    msg = 'No options found in any frame for dropdown'
                    # logger.info(msg)
                    return ActionResult(extracted_content=msg, include_in_memory=True)

            except Exception as e:
                # logger.error(f'Failed to get dropdown options: {str(e)}')
                msg = f'Error getting options: {str(e)}'
                # logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

        async def select_dropdown_option(
                index: int,
                text: str,
                browser,
        ) -> ActionResult:
            """Select an option in a native <select> by its visible text (frame-aware).

            Params:
            - index: int — element index of the <select>.
            - text: str — exact visible text (use value returned by get_dropdown_options).

            Returns: ActionResult("selected option <text> with value <...>")"""

            page = await browser.get_current_page()
            selector_map = await browser.get_selector_map()
            dom_element = selector_map[index]

            # Validate that we're working with a select element
            if dom_element.tag_name != 'select':
                # logger.error(f'Element is not a select! Tag: {dom_element.tag_name}, Attributes: {dom_element.attributes}')
                msg = f'Cannot select option: Element with index {index} is a {dom_element.tag_name}, not a select'
                return ActionResult(extracted_content=msg, include_in_memory=True)

            # logger.debug(f"Attempting to select '{text}' using xpath: {dom_element.xpath}")
            # logger.debug(f'Element attributes: {dom_element.attributes}')
            # logger.debug(f'Element tag: {dom_element.tag_name}')

            xpath = '//' + dom_element.xpath

            try:
                frame_index = 0
                for frame in page.frames:
                    try:
                        # logger.debug(f'Trying frame {frame_index} URL: {frame.url}')

                        # First verify we can find the dropdown in this frame
                        find_dropdown_js = """
							(xpath) => {
								try {
									const select = document.evaluate(xpath, document, null,
										XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
									if (!select) return null;
									if (select.tagName.toLowerCase() !== 'select') {
										return {
											error: `Found element but it's a ${select.tagName}, not a SELECT`,
											found: false
										};
									}
									return {
										id: select.id,
										name: select.name,
										found: true,
										tagName: select.tagName,
										optionCount: select.options.length,
										currentValue: select.value,
										availableOptions: Array.from(select.options).map(o => o.text.trim())
									};
								} catch (e) {
									return {error: e.toString(), found: false};
								}
							}
						"""

                        dropdown_info = await frame.evaluate(find_dropdown_js, dom_element.xpath)

                        if dropdown_info:
                            if not dropdown_info.get('found'):
                                # logger.error(f'Frame {frame_index} error: {dropdown_info.get("error")}')
                                continue

                            # logger.debug(f'Found dropdown in frame {frame_index}: {dropdown_info}')

                            # "label" because we are selecting by text
                            # nth(0) to disable error thrown by strict mode
                            # timeout=1000 because we are already waiting for all network events, therefore ideally we don't need to wait a lot here (default 30s)
                            selected_option_values = (
                                await frame.locator('//' + dom_element.xpath).nth(0).select_option(label=text, timeout=1000)
                            )

                            msg = f'selected option {text} with value {selected_option_values}'
                            # logger.info(msg + f' in frame {frame_index}')

                            return ActionResult(extracted_content=msg, include_in_memory=True)

                    except Exception as frame_e:
                        # logger.error(f'Frame {frame_index} attempt failed: {str(frame_e)}')
                        # logger.error(f'Frame type: {type(frame)}')
                        # logger.error(f'Frame URL: {frame.url}')
                        pass

                    frame_index += 1

                msg = f"Could not select option '{text}' in any frame"
                # logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

            except Exception as e:
                msg = f'Selection failed: {str(e)}'
                # logger.error(msg)
                return ActionResult(error=msg, include_in_memory=True)

        async def drag_drop(params:DragDropAction, browser) -> ActionResult:
            """Perform a drag-and-drop between elements or coordinates.

            Params (object) — provide one of:
            1) Element-based:
            - element_source: str — CSS/XPath locator
            - element_target: str — CSS/XPath locator
            - element_source_offset: {x:int,y:int} (optional)
            - element_target_offset: {x:int,y:int} (optional)
            2) Coordinate-based:
            - coord_source_x: int, coord_source_y: int
            - coord_target_x: int, coord_target_y: int

            Optional:
            - steps: int (default: 10) — mouse move segments
            - delay_ms: int (default: 5) — delay between segments

            Returns: ActionResult with success or detailed error."""


            async def get_drag_elements(
                    page: Page,
                    source_selector: str,
                    target_selector: str,
            ) -> tuple[ElementHandle | None, ElementHandle | None]:
                """Get source and target elements with appropriate error handling."""
                source_element = None
                target_element = None

                try:
                    # page.locator() auto-detects CSS and XPath
                    source_locator = page.locator(source_selector)
                    target_locator = page.locator(target_selector)

                    # Check if elements exist
                    source_count = await source_locator.count()
                    target_count = await target_locator.count()

                    if source_count > 0:
                        source_element = await source_locator.first.element_handle()
                        # logger.debug(f'Found source element with selector: {source_selector}')
                    else:
                        # logger.warning(f'Source element not found: {source_selector}')
                        pass

                    if target_count > 0:
                        target_element = await target_locator.first.element_handle()
                        # logger.debug(f'Found target element with selector: {target_selector}')
                    else:
                        # logger.warning(f'Target element not found: {target_selector}')
                        pass

                except Exception as e:
                    # logger.error(f'Error finding elements: {str(e)}')
                    pass

                return source_element, target_element

            async def get_element_coordinates(
                    source_element: ElementHandle,
                    target_element: ElementHandle,
                    source_position,
                    target_position,
            ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
                """Get coordinates from elements with appropriate error handling."""
                source_coords = None
                target_coords = None

                try:
                    # Get source coordinates
                    if source_position:
                        source_coords = (source_position.x, source_position.y)
                    else:
                        source_box = await source_element.bounding_box()
                        if source_box:
                            source_coords = (
                                int(source_box['x'] + source_box['width'] / 2),
                                int(source_box['y'] + source_box['height'] / 2),
                            )

                    # Get target coordinates
                    if target_position:
                        target_coords = (target_position.x, target_position.y)
                    else:
                        target_box = await target_element.bounding_box()
                        if target_box:
                            target_coords = (
                                int(target_box['x'] + target_box['width'] / 2),
                                int(target_box['y'] + target_box['height'] / 2),
                            )
                except Exception as e:
                    # logger.error(f'Error getting element coordinates: {str(e)}')
                    pass

                return source_coords, target_coords

            async def execute_drag_operation(
                    page: Page,
                    source_x: int,
                    source_y: int,
                    target_x: int,
                    target_y: int,
                    steps: int,
                    delay_ms: int,
            ) -> tuple[bool, str]:
                """Execute the drag operation with comprehensive error handling."""
                try:
                    # Try to move to source position
                    try:
                        await page.mouse.move(source_x, source_y)
                        # logger.debug(f'Moved to source position ({source_x}, {source_y})')
                    except Exception as e:
                        # logger.error(f'Failed to move to source position: {str(e)}')
                        return False, f'Failed to move to source position: {str(e)}'

                    # Press mouse button down
                    await page.mouse.down()

                    # Move to target position with intermediate steps
                    for i in range(1, steps + 1):
                        ratio = i / steps
                        intermediate_x = int(source_x + (target_x - source_x) * ratio)
                        intermediate_y = int(source_y + (target_y - source_y) * ratio)

                        await page.mouse.move(intermediate_x, intermediate_y)

                        if delay_ms > 0:
                            await asyncio.sleep(delay_ms / 1000)

                    # Move to final target position
                    await page.mouse.move(target_x, target_y)

                    # Move again to ensure dragover events are properly triggered
                    await page.mouse.move(target_x, target_y)

                    # Release mouse button
                    await page.mouse.up()

                    return True, 'Drag operation completed successfully'

                except Exception as e:
                    return False, f'Error during drag operation: {str(e)}'

            page = await browser.get_current_page()

            try:
                # Initialize variables
                source_x: int | None = None
                source_y: int | None = None
                target_x: int | None = None
                target_y: int | None = None

                # Normalize parameters
                steps = max(1, params.steps or 10)
                delay_ms = max(0, params.delay_ms or 5)

                # Case 1: Element selectors provided
                if params.element_source and params.element_target:
                    # logger.debug('Using element-based approach with selectors')

                    source_element, target_element = await get_drag_elements(
                        page,
                        params.element_source,
                        params.element_target,
                    )

                    if not source_element or not target_element:
                        error_msg = f'Failed to find {"source" if not source_element else "target"} element'
                        return ActionResult(error=error_msg, include_in_memory=True)

                    source_coords, target_coords = await get_element_coordinates(
                        source_element, target_element, params.element_source_offset, params.element_target_offset
                    )

                    if not source_coords or not target_coords:
                        error_msg = f'Failed to determine {"source" if not source_coords else "target"} coordinates'
                        return ActionResult(error=error_msg, include_in_memory=True)

                    source_x, source_y = source_coords
                    target_x, target_y = target_coords

                # Case 2: Coordinates provided directly
                elif all(
                        coord is not None
                        for coord in [params.coord_source_x, params.coord_source_y, params.coord_target_x, params.coord_target_y]
                ):
                    # logger.debug('Using coordinate-based approach')
                    source_x = params.coord_source_x
                    source_y = params.coord_source_y
                    target_x = params.coord_target_x
                    target_y = params.coord_target_y
                else:
                    error_msg = 'Must provide either source/target selectors or source/target coordinates'
                    return ActionResult(error=error_msg, include_in_memory=True)

                # Validate coordinates
                if any(coord is None for coord in [source_x, source_y, target_x, target_y]):
                    error_msg = 'Failed to determine source or target coordinates'
                    return ActionResult(error=error_msg, include_in_memory=True)

                # Perform the drag operation
                success, message = await execute_drag_operation(
                    page,
                    cast(int, source_x),
                    cast(int, source_y),
                    cast(int, target_x),
                    cast(int, target_y),
                    steps,
                    delay_ms,
                )

                if not success:
                    # logger.error(f'Drag operation failed: {message}')
                    return ActionResult(error=message, include_in_memory=True)

                # Create descriptive message
                if params.element_source and params.element_target:
                    msg = f"🖱️ Dragged element '{params.element_source}' to '{params.element_target}'"
                else:
                    msg = f'🖱️ Dragged from ({source_x}, {source_y}) to ({target_x}, {target_y})'

                # logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

            except Exception as e:
                error_msg = f'Failed to perform drag and drop: {str(e)}'
                # logger.error(error_msg)
                return ActionResult(error=error_msg, include_in_memory=True)

        async def get_sheet_contents(browser):
            """Copy the entire sheet/grid selection to clipboard and return its TSV contents.

            Params: {}

            Returns: ActionResult(extracted_content="<tab-separated-values>")"""

            page = await browser.get_current_page()

            # select all cells
            await page.keyboard.press('Enter')
            await page.keyboard.press('Escape')
            await page.keyboard.press('ControlOrMeta+A')
            await page.keyboard.press('ControlOrMeta+C')

            extracted_tsv = await page.evaluate('() => navigator.clipboard.readText()')
            return ActionResult(extracted_content=extracted_tsv, include_in_memory=True)

        async def select_cell_or_range(browser, cell_or_range: str):
            """Jump to a cell or range in a spreadsheet-like UI via GoTo (e.g., "B12" or "B2:D5").

            Params:
            - cell_or_range: str

            Returns: ActionResult("Selected cell <cell_or_range>")"""

            page = await browser.get_current_page()

            await page.keyboard.press('Enter')  # make sure we dont delete current cell contents if we were last editing
            await page.keyboard.press('Escape')  # to clear current focus (otherwise select range popup is additive)
            await asyncio.sleep(0.1)
            await page.keyboard.press('Home')  # move cursor to the top left of the sheet first
            await page.keyboard.press('ArrowUp')
            await asyncio.sleep(0.1)
            await page.keyboard.press('Control+G')  # open the goto range popup
            await asyncio.sleep(0.2)
            await page.keyboard.type(cell_or_range, delay=0.05)
            await asyncio.sleep(0.2)
            await page.keyboard.press('Enter')
            await asyncio.sleep(0.2)
            await page.keyboard.press('Escape')  # to make sure the popup still closes in the case where the jump failed
            return ActionResult(extracted_content=f'Selected cell {cell_or_range}', include_in_memory=False)


        async def get_range_contents(browser, cell_or_range: str):
            """Copy the contents of a specific cell/range and return it as TSV.

            Params:
            - cell_or_range: str — e.g., "A1" or "B2:D5"

            Returns: ActionResult(extracted_content="<tab-separated-values>")"""

            page = await browser.get_current_page()

            await select_cell_or_range(browser, cell_or_range)

            await page.keyboard.press('ControlOrMeta+C')
            await asyncio.sleep(0.1)
            extracted_tsv = await page.evaluate('() => navigator.clipboard.readText()')
            return ActionResult(extracted_content=extracted_tsv, include_in_memory=True)

        async def clear_selected_range(browser):
            """Clear the currently selected cell/range.

            Params: {}

            Returns: ActionResult("Cleared selected range")"""

            page = await browser.get_current_page()

            await page.keyboard.press('Backspace')
            return ActionResult(extracted_content='Cleared selected range', include_in_memory=False)

        async def input_selected_cell_text(browser, text: str):
            """Type text into the currently selected cell and commit.

            Params:
            - text: str

            Returns: ActionResult("Inputted text <text>")"""

            page = await browser.get_current_page()

            await page.keyboard.type(text, delay=0.1)
            await page.keyboard.press('Enter')  # make sure to commit the input so it doesn't get overwritten by the next action
            await page.keyboard.press('ArrowUp')
            return ActionResult(extracted_content=f'Inputted text {text}', include_in_memory=False)

        async def update_range_contents(browser, range: str, new_contents_tsv: str):
            """Paste TSV data into a specific cell/range.

            Params:
            - range: str — target cell/range, e.g., "B2:D5"
            - new_contents_tsv: str — TSV payload to paste (\\t for columns, \\n for rows)

            Returns: ActionResult("Updated cell <range> with <new_contents_tsv>")"""

            page = await browser.get_current_page()

            await select_cell_or_range(browser, range)

            # simulate paste event from clipboard with TSV content
            await page.evaluate(f"""
				const clipboardData = new DataTransfer();
				clipboardData.setData('text/plain', `{new_contents_tsv}`);
				document.activeElement.dispatchEvent(new ClipboardEvent('paste', {{clipboardData}}));
			""")

            return ActionResult(extracted_content=f'Updated cell {range} with {new_contents_tsv}', include_in_memory=False)
        
        _auto_register(locals())


        tool_descriptions = {
            "done": "Mark the browser task as finished and return the collected results to the caller.",
            "take_screenshot": "Capture a PNG screenshot of the current page and store it in the local artifacts folder.",
            "find_in_page": "Search the current page for the query text and return counts plus context snippets.",
            "zoom_to_point": "Zoom toward the provided selector, text, or coordinates using the chosen strategy.",
            "sv_click_timeline_arrow": "In Google Street View, click the timeline arrows to move to earlier or later imagery.",
            "search_duckduckgo": "Open DuckDuckGo with the given query and wait for the results page.",
            "search_google": "Search Google for the query, handling consent prompts and captcha fallback.",
            "open_orcid_if_present": "Open an ORCID profile when the query includes a valid ORCID identifier.",
            "go_to_url": "Navigate the current tab directly to the provided URL.",
            "find_archive_url": "Query the Wayback Machine for the closest snapshot to the requested URL and date.",
            "go_back": "Navigate one step back in the browser history.",
            "wait": "Pause execution for the specified number of seconds before continuing.",
            "click_element_by_index": "Click the element referenced by the browser-use selector index, handling downloads and popups.",
            "input_text": "Type the supplied text into the element referenced by the selector index.",
            "save_pdf": "Print the current page to PDF and save it in the local artifacts folder.",
            "video_viewer": "Open the local YouTube viewer and seek to the requested timestamp for the supplied video.",
            "pdf_viewer": "Open or download a PDF via the local viewer and support scrolling, jumping, or search actions.",
            "switch_tab": "Switch focus to an existing browser tab by index and wait for it to load.",
            "open_tab": "Create a new browser tab and navigate it to the supplied URL.",
            "close_tab": "Close an existing browser tab by index and report which URL was closed.",
            "extract_content": "Extract structured page content relevant to a stated goal using the extraction LLM.",
            "scroll_down": "Scroll the page down by a specific pixel amount or half the viewport.",
            "scroll_up": "Scroll the page up by a specific pixel amount or half the viewport.",
            "send_keys": "Send raw keyboard shortcuts or keys to the active page.",
            "scroll_to_text": "Scroll until the first visible instance of the provided text is brought into view.",
            "get_dropdown_options": "Read all options from a native select element identified by selector index.",
            "select_dropdown_option": "Choose an option from a native select element by visible text.",
            "drag_drop": "Drag from a source element or coordinate to a target location with optional offsets.",
            "get_sheet_contents": "Copy the full spreadsheet selection to clipboard and return the TSV contents.",
            "select_cell_or_range": "Jump to a spreadsheet cell or range using the Go To dialog.",
            "get_range_contents": "Copy a specific spreadsheet cell or range as TSV data.",
            "clear_selected_range": "Clear the currently selected spreadsheet cells.",
            "input_selected_cell_text": "Type text into the currently selected spreadsheet cell and commit it.",
            "update_range_contents": "Paste TSV data into a spreadsheet cell or range.",
        }
        for _name, _description in tool_descriptions.items():
            if _name in self._action_specs and _description:
                spec = self._action_specs[_name]
                func = spec["func"]
                setattr(func, "description", _description)
                func.__doc__ = _description
                spec["description"] = _description
                if _name in self._registry_actions:
                    self._registry_actions[_name].description = _description


        class _CompatRegistry:
            def __init__(self, outer: "Controller"):
                self._outer = outer
                # Important: make .registry the same object so Agent can call
                # tools.registry.create_action_model(...)
                self.registry = self

            @property
            def actions(self):
                # Some Agent versions look for tools.registry.actions
                return self._outer._registry_actions

            async def execute_action(self, action_name, params, **kwargs):
                return await self._outer._execute_action(action_name, params, **kwargs)

            def action(self, description: str, **kwargs):
                name = kwargs.get("name")
                def _decorator(func):
                    setattr(func, "description", description)
                    if name:
                        setattr(func, "name", name)
                    self._outer.register_tool(func, name=name, description=description)
                    return func
                return _decorator

            @staticmethod
            def _norm_params(p):
                if isinstance(p, BaseModel):
                    return p.model_dump(exclude_none=True)
                if isinstance(p, dict):
                    return p
                return {"params": p}

            def create_action_model(
                self,
                include_actions: list[str] | None = None,
                page=None,
                page_url: str | None = None,   # new in browser-use ≥0.7.7
                **_ignored,                    # future-proof for new args
            ) -> type[ActionModel]:
                specs = self._outer._action_specs
                if include_actions is not None:
                    specs = {name: spec for name, spec in specs.items() if name in include_actions}

                fields = {}
                for name, spec in specs.items():
                    param_model = spec["param_model"]
                    fields[name] = (Optional[param_model], Field(default=None, description=spec["description"]))

                if fields:
                    return create_model("ActionModel", __base__=ActionModel, **fields)
                return create_model("ActionModel", __base__=ActionModel)

            def get_prompt_description(self, *_, **__):  # accept extra positional/keyword args
                lines = []
                for name, spec in self._outer._action_specs.items():
                    desc = spec["description"]
                    param_model = spec["param_model"]
                    takes_model = spec["takes_model"]

                    if takes_model and issubclass(param_model, BaseModel):
                        param_sig = f"params: {param_model.__name__}"
                    else:
                        model_fields = getattr(param_model, "model_fields", {})
                        parts = []
                        for field_name, field in model_fields.items():
                            ann = getattr(field.annotation, "__name__", str(field.annotation))
                            parts.append(f"{field_name}: {ann}")
                        param_sig = ", ".join(parts)

                    line = f"- {name}({param_sig})" if param_sig else f"- {name}()"
                    if desc:
                        line += f" - {desc}"
                    lines.append(line)
                return "\n".join(lines) or "No tools registered."

            def list_actions(self) -> list[str]:
                return list(self._outer._action_specs.keys())

            def get_action_description(self, name: str) -> str:
                spec = self._outer._action_specs.get(name)
                return spec["description"] if spec else ""

            async def act(self, action, *, browser_session=None, page_extraction_llm=None,
                        sensitive_data=None, available_file_paths=None, context=None, **_):
                if browser_session is None:
                    raise RuntimeError("No browser_session was provided by the Agent")
                # adapter = _BrowserAdapter(browser_session)  # <- the wrapper your tools use
                return await self._outer.act(
                    action=action,
                    browser_context=browser_session,                # <- your tools expect "browser"
                    page_extraction_llm=page_extraction_llm,
                    sensitive_data=sensitive_data,
                    available_file_paths=available_file_paths,
                    context=context,
                )

        self.registry = _CompatRegistry(self)
    
    async def _execute_action(
        self,
        action_name: str,
        params: Any,
        *,
        browser,
        page_extraction_llm,
        sensitive_data=None,
        available_file_paths=None,
        context=None,
    ):
        if action_name not in self._action_specs:
            raise KeyError(f"Action '{action_name}' not found")

        """if not hasattr(browser, "get_current_page"):
            browser = _BrowserAdapter(browser)"""

        spec = self._action_specs[action_name]
        fn = spec["func"]
        param_model = spec["param_model"]
        takes_model = spec["takes_model"]
        user_param_names = spec["user_param_names"]

        sig = inspect.signature(fn)
        kw = {}

        if "browser" in sig.parameters:
            kw["browser"] = browser
        if "page_extraction_llm" in sig.parameters:
            kw["page_extraction_llm"] = page_extraction_llm
        if "sensitive_data" in sig.parameters:
            kw["sensitive_data"] = sensitive_data
        if "available_file_paths" in sig.parameters:
            kw["available_file_paths"] = available_file_paths
        if "context" in sig.parameters:
            kw["context"] = context

        if takes_model:
            if isinstance(params, param_model):
                validated = params
            elif isinstance(params, BaseModel):
                validated = param_model.model_validate(params.model_dump())
            elif isinstance(params, dict):
                validated = param_model.model_validate(params)
            elif params is None:
                validated = param_model()
            else:
                normalized = self.registry._norm_params(params)
                payload = normalized if isinstance(normalized, dict) else {"params": params}
                validated = param_model.model_validate(payload)
        else:
            if isinstance(params, BaseModel):
                payload = params.model_dump(exclude_none=True)
            elif isinstance(params, dict):
                payload = params
            elif params is None:
                payload = {}
            elif len(user_param_names) == 1:
                payload = {user_param_names[0]: params}
            else:
                normalized = self.registry._norm_params(params)
                payload = normalized if isinstance(normalized, dict) else {}
            validated = param_model(**payload)

        payload = validated.model_dump(exclude_none=True) if isinstance(validated, BaseModel) else {}

        if takes_model:
            call = fn(validated, **kw)
        elif payload:
            call = fn(**payload, **kw)
        else:
            call = fn(**kw)

        if inspect.isawaitable(call):
            return await call
        return call

    # Register ---------------------------------------------------------------

    def action(self, description: str, **kwargs):
        """Decorator for registering custom actions (optional)."""
        name = kwargs.get("name")
        def _decorator(func):
            setattr(func, "description", description)
            if name:
                setattr(func, "name", name)
            # was: register_tool(...)  -> NameError outside __init__
            self.register_tool(func, name=name, description=description)  # <--
            return func
        return _decorator


    # Act --------------------------------------------------------------------

    @time_execution_sync('--act')
    async def act(
            self,
            action: ActionModel,
            browser_context,
            #
            page_extraction_llm,
            sensitive_data: dict[str, str] | None = None,
            available_file_paths: list[str] | None = None,
            #
            context: Context | None = None,
    ) -> ActionResult:
        """Execute an action"""

        try:
            for action_name, params in action.model_dump(exclude_unset=True).items():
                if params is not None:
                    # with Laminar.start_as_current_span(
                    # 	name=action_name,
                    # 	input={
                    # 		'action': action_name,
                    # 		'params': params,
                    # 	},
                    # 	span_type='TOOL',
                    # ):
                    result = await self._execute_action(
                        action_name,
                        params,
                        browser=browser_context,
                        page_extraction_llm=page_extraction_llm,
                        sensitive_data=sensitive_data,
                        available_file_paths=available_file_paths,
                        context=context,
                    )

                    # Laminar.set_span_output(result)
                    print(result)
                    if isinstance(result, ActionResult) and result.is_done:
                        try:
                            page = await browser_context.get_current_page()
                            url = page.url or ""
                            if "google.com/maps" in url:
                                check = await self._sv_verify_earliest(page)
                                # If we were not truly at the earliest, force the agent to continue
                                if not check["earliest"]:
                                    result.success = False
                                    result.is_done = False
                                    msg = (
                                        f"\n\n⚠️ Earlier Street View imagery detected."
                                        f" Moved timeline to earliest: '{check['label']}'."
                                        f" (aria_now={check['vnow']}, aria_min={check['vmin']}, clicks={check['clicks']})"
                                        f" Continue analysis from this earliest frame."
                                    )
                                    # logger.info(msg)
                                    result.extracted_content = (result.extracted_content or "") + msg
                                    return result
                                else:
                                    # logger.info("Check is earliest already")
                                    pass
                        except Exception:
                            # Fail open (don’t block task if guard hits a transient error)
                            pass

                    if isinstance(result, str):
                        return ActionResult(extracted_content=result)
                    elif isinstance(result, ActionResult):
                        return result
                    elif result is None:
                        return ActionResult()
                    else:
                        raise ValueError(f'Invalid action result type: {type(result)} of {result}')
            return ActionResult()
        except Exception as e:
            raise e
