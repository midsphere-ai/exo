import asyncio
import inspect
import os
from dotenv import load_dotenv
load_dotenv(verbose=True)
import shlex

from e2b_code_interpreter import Sandbox
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("e2b-python-interpreter")

# API keys
E2B_API_KEY = os.getenv("E2B_API_KEY")
LOGS_DIR = "e2b_logs"  # Directory where benchmark logs are stored

# DEFAULT TEMPLATE ID
# see README.md on how to build this
DEFAULT_TEMPLATE_ID = os.getenv("E2B_TEMPLATE_ID")

# DEFAULT CONFS
DEFAULT_TIMEOUT = 1200  # seconds
COMMAND_MAX_RETRIES = 5
NETWORK_TIMEOUT_SECONDS = 20
SANDBOX_HOME = "/home/user"

try:
    _SANDBOX_INIT_PARAMS = inspect.signature(Sandbox.__init__).parameters
except (OSError, ValueError, TypeError):
    _SANDBOX_INIT_PARAMS = {}

try:
    _SANDBOX_CONNECT_PARAMS = inspect.signature(Sandbox.connect).parameters
except (OSError, ValueError, TypeError):
    _SANDBOX_CONNECT_PARAMS = {}

_SANDBOX_CREATE_FUNC = getattr(Sandbox, "create", None)
if callable(_SANDBOX_CREATE_FUNC):
    try:
        _SANDBOX_CREATE_PARAMS = inspect.signature(_SANDBOX_CREATE_FUNC).parameters
    except (OSError, ValueError, TypeError):
        _SANDBOX_CREATE_PARAMS = {}
else:
    _SANDBOX_CREATE_FUNC = None
    _SANDBOX_CREATE_PARAMS = {}


def _has_var_keyword(params) -> bool:
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    ) if params else False


def _supports_api_key(params) -> bool:
    """Return True when the inspected signature accepts api_key or arbitrary kwargs."""
    if not params:
        return True
    return "api_key" in params or _has_var_keyword(params)


def _first_matching_param(params, candidates):
    for name in candidates:
        if params and name in params:
            return name
    return None


_SANDBOX_TEMPLATE_PARAM = _first_matching_param(
    _SANDBOX_INIT_PARAMS, ("template", "template_id")
)
_SANDBOX_INIT_SUPPORTS_TIMEOUT = "timeout" in _SANDBOX_INIT_PARAMS
_SANDBOX_SUPPORTS_API_KEY = _supports_api_key(_SANDBOX_INIT_PARAMS)
_SANDBOX_CONNECT_SUPPORTS_API_KEY = _supports_api_key(_SANDBOX_CONNECT_PARAMS)
_SANDBOX_CREATE_SUPPORTS_API_KEY = _supports_api_key(_SANDBOX_CREATE_PARAMS)
_SANDBOX_CREATE_TEMPLATE_PARAM = _first_matching_param(
    _SANDBOX_CREATE_PARAMS, ("template", "template_id")
)
_SANDBOX_CREATE_SUPPORTS_TIMEOUT = "timeout" in _SANDBOX_CREATE_PARAMS
_SANDBOX_INIT_REQUIRES_DETAILS = all(
    key in _SANDBOX_INIT_PARAMS
    for key in (
        "sandbox_id",
        "envd_version",
        "envd_access_token",
        "sandbox_domain",
        "connection_config",
    )
)
_SANDBOX_SHOULD_USE_CREATE = bool(_SANDBOX_CREATE_FUNC) and (
    _SANDBOX_INIT_REQUIRES_DETAILS or not _SANDBOX_TEMPLATE_PARAM
)


def _safe_set_timeout(sandbox, timeout: int = DEFAULT_TIMEOUT) -> None:
    """Set the sandbox timeout only if the method exists."""
    if sandbox is None:
        return

    set_timeout = getattr(sandbox, "set_timeout", None)
    if not callable(set_timeout):
        return

    try:
        set_timeout(timeout)
    except Exception:
        pass


def _instantiate_sandbox():
    """Create a sandbox instance compatible with both old and new SDKs."""
    if _SANDBOX_SHOULD_USE_CREATE and _SANDBOX_CREATE_FUNC:
        sandbox_kwargs = {}
        if E2B_API_KEY and _SANDBOX_CREATE_SUPPORTS_API_KEY:
            sandbox_kwargs["api_key"] = E2B_API_KEY
        if _SANDBOX_CREATE_SUPPORTS_TIMEOUT:
            sandbox_kwargs["timeout"] = DEFAULT_TIMEOUT
        if (
            DEFAULT_TEMPLATE_ID
            and _SANDBOX_CREATE_TEMPLATE_PARAM
            and _SANDBOX_CREATE_TEMPLATE_PARAM not in ("cls",)
        ):
            sandbox_kwargs[_SANDBOX_CREATE_TEMPLATE_PARAM] = DEFAULT_TEMPLATE_ID
        return Sandbox.create(**sandbox_kwargs)

    sandbox_kwargs = {}
    if E2B_API_KEY and _SANDBOX_SUPPORTS_API_KEY:
        sandbox_kwargs["api_key"] = E2B_API_KEY
    if _SANDBOX_INIT_SUPPORTS_TIMEOUT:
        sandbox_kwargs["timeout"] = DEFAULT_TIMEOUT
    if _SANDBOX_TEMPLATE_PARAM and DEFAULT_TEMPLATE_ID:
        sandbox_kwargs[_SANDBOX_TEMPLATE_PARAM] = DEFAULT_TEMPLATE_ID
    return Sandbox(**sandbox_kwargs)


def _connect_to_sandbox(sandbox_id: str):
    """Connect to a sandbox, passing api_key only when supported."""
    connect_kwargs = {}
    if _SANDBOX_CONNECT_SUPPORTS_API_KEY:
        connect_kwargs["api_key"] = E2B_API_KEY
    return Sandbox.connect(sandbox_id, **connect_kwargs)


def _is_valid_local_file(path: str) -> bool:
    """Return True if path points to an existing file on local filesystem."""
    return bool(path) and os.path.isfile(path)


def _normalize_sandbox_path(path: str) -> str:
    """Normalize a sandbox path and prevent escaping the user home."""
    if not path:
        return SANDBOX_HOME
    normalized = os.path.normpath(path)
    if not normalized.startswith(SANDBOX_HOME):
        normalized = os.path.join(SANDBOX_HOME, os.path.basename(normalized))
    return normalized


def _ensure_logs_dir() -> str:
    """Ensure LOGS_DIR exists and return it."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    return LOGS_DIR

# Common packages to install in sandbox
COMMON_PACKAGES = [
    "PyPDF2",  # for PyPDF2 import
    "geopy",  # for geopy import
    "PyMuPDF",  # for fitz import
    "docx2txt",  # for docx2txt import
    "pdfminer.six",  # for pdfminer import
    "rdkit",  # for rdkit import
    "pytesseract",  # for pytesseract import
    "python-chess",  # for python-chess import
    "stockfish",  # for stockfish import
    "yfinance",  # for yfinance import
    "CoolProp",  # for CoolProp import
    "seaborn",  # for seaborn import
    "python-pptx",  # for python-pptx import
    "python-docx",  # for python-docx import
    "pdfplumber",  # for pdfplumber import
    "geopandas",  # for geopandas import
    "yt-dlp",  # for yt-dlp import
    "pydub",  # for pydub import
    "pyaudio",  # for pyaudio import
    "biopython",  # for biopython import
    "pubchempy",  # for pubchempy import
    "googletrans",  # for googletrans import
    "pyshp",  # for pyshp import
    "SpeechRecognition",  # for SpeechRecognition import
    "opencv-python",  # for opencv-python import
    "selenium",  # for selenium import
    "waybackpy",  # for waybackpy import
    "networkx",  # for networkx import
    "wbdata",  # for wbdata import
]

# System packages to install in sandbox
SYSTEM_PACKAGES = [
    "poppler-utils",  # for pdfinfo, pdftotext, pdfimages, pdftoppm commands
    "pdfgrep",  # for pdfgrep command (PDF text search)
    "libimage-exiftool-perl",  # for exiftool command (image metadata)
    "unlambda",  # for unlambda interpreter
    "stockfish",  # for stockfish chess engine
]


async def _install_common_packages(sandbox, sandbox_id: str) -> bool:
    """Install common Python packages in the sandbox.

    Args:
        sandbox: The connected sandbox instance
        sandbox_id: The sandbox ID for error messages

    Returns:
        True if installation successful, False otherwise
    """
    # Install system packages first (pdfinfo, pdftotext, etc.)
    try:
        # Update package list
        update_result = sandbox.commands.run("sudo apt-get update")
        if update_result.exit_code != 0:
            print(
                f"Warning: Failed to update package list in sandbox {sandbox_id}: {update_result.stderr}"
            )

        # Install system packages one by one
        for package in SYSTEM_PACKAGES:
            install_cmd = f"sudo apt-get install -y {package}"
            install_result = sandbox.commands.run(install_cmd)

            if install_result.exit_code != 0:
                print(
                    f"Warning: Failed to install system package {package} in sandbox {sandbox_id}: {install_result.stderr}"
                )
                # Continue installing other packages even if one fails
            else:
                print(
                    f"Successfully installed system package {package} in sandbox {sandbox_id}"
                )

    except Exception as e:
        print(f"Warning: Error installing system packages in sandbox {sandbox_id}: {e}")

    # Update pip
    install_result = sandbox.commands.run("pip install --upgrade pip")
    if install_result.exit_code != 0:
        print(
            f"Warning: Failed to update pip in sandbox {sandbox_id}: {install_result.stderr}"
        )
    try:
        # Install packages one by one to avoid one failure affecting others
        for package in COMMON_PACKAGES:
            install_cmd = f"pip install {package}"
            install_result = sandbox.commands.run(install_cmd)

            if install_result.exit_code != 0:
                print(
                    f"Warning: Failed to install package {package} in sandbox {sandbox_id}: {install_result.stderr}"
                )
                # Continue installing other packages even if one fails
            else:
                print(
                    f"Successfully installed package {package} in sandbox {sandbox_id}"
                )

        return True
    except Exception as e:
        print(f"Warning: Error installing common packages in sandbox {sandbox_id}: {e}")
        return False


@mcp.tool()
async def create_sandbox() -> str:
    """Create a linux sandbox and get the `sandbox_id` for safely executing commands and running python code. Note that the `sandbox_id` can only be assigned and cannot be manually specified.

    The sandbox may timeout and automatically shutdown. If so, you will need to create a new sandbox.

    IMPORTANT: Do not execute `create_sandbox` and other sandbox tools in the same message. You must wait for `create_sandbox` to return the `sandbox_id`, then use that `sandbox_id` to specify the working sandbox in subsequent messages.

    Returns:
        The `sandbox_id` of the newly created sandbox. You should use this `sandbox_id` to run other tools in the sandbox.
    """
    _ensure_logs_dir()
    for attempt in range(1, COMMAND_MAX_RETRIES + 1):
        sandbox = None
        try:
            sandbox = _instantiate_sandbox()
            info = sandbox.get_info()

            # Install common packages before running code
            # await _install_common_packages(sandbox, info.sandbox_id)
            tmpfiles_dir = os.path.join(LOGS_DIR, "e2b_tmp_files")
            os.makedirs(tmpfiles_dir, exist_ok=True)

            return f"Sandbox created with sandbox_id: {info.sandbox_id}"
        except Exception as e:
            if attempt == COMMAND_MAX_RETRIES:
                return f"Failed to create sandbox after {COMMAND_MAX_RETRIES} attempts: {e}, please retry later."
            await asyncio.sleep(attempt * 2)  # Exponential backoff
        finally:
            # Set timeout before exit to prevent timeout after function exits
            _safe_set_timeout(sandbox)


@mcp.tool()
async def run_command(command: str, sandbox_id: str) -> str:
    """Execute a shell command in the linux sandbox.
    The sandbox is already installed with common system packages for the task.

    Args:
        command: The shell command to execute.
        sandbox_id: The id of the existing sandbox to execute the command in. (must be created first via `create_sandbox`).

    Returns:
        A result of the command execution, format like (stderr=..., stdout=..., exit_code=..., error=...)
    """
    sandbox = None
    try:
        sandbox = _connect_to_sandbox(sandbox_id)
    except Exception:
        return f"[ERROR]: Failed to connect to sandbox {sandbox_id}, retry later. Make sure the sandbox is created and the id is correct."

    for attempt in range(1, COMMAND_MAX_RETRIES + 1):
        try:
            _safe_set_timeout(
                sandbox
            )  # refresh the timeout for each command execution
            result = sandbox.commands.run(command)

            # Check if command contains package installation commands
            result_str = str(result)
            if "pip install" in command or "apt-get" in command:
                result_str += "\n\n[PACKAGE INSTALL STATUS]: The system packages and Python packages required for the task have been installed. No need to install them again unless a missing package error occurs during execution."

            return result_str
        except Exception as e:
            if attempt == COMMAND_MAX_RETRIES:
                error_msg = f"[ERROR]: Failed to run command after {COMMAND_MAX_RETRIES} attempts. Exception type: {type(e).__name__}, Details: {e}. \n\n[HINT]: Shell commands can be error-prone. Consider using the `run_python_code` tool instead to accomplish the same task with Python code, which often provides better error handling and more detailed error messages.\n\n[PERMISSION HINT]: You are running as user, not root. If you encounter permission issues, use `sudo` for commands that require administrator privileges (e.g., `sudo apt-get install`, `sudo systemctl`, etc.)."

                # Add package install status note for failed install commands too
                if "pip install" in command or "apt-get" in command:
                    error_msg += "\n\n[PACKAGE INSTALL STATUS]: The system packages and Python packages required for the task have been installed. No need to install them again unless a missing package error occurs during execution."

                return error_msg
            await asyncio.sleep(attempt * 2)  # Exponential backoff
        finally:
            # Set timeout before exit to prevent timeout after function exits
            _safe_set_timeout(sandbox)


@mcp.tool()
async def run_python_code(code_block: str, sandbox_id: str) -> str:
    """Run python code in the sandbox and return the execution result.
    The sandbox is already installed with common python packages for the task.

    Args:
        code_block: The python code to run.
        sandbox_id: The id of the existing sandbox to run the code in. (must be created first via `create_sandbox`).

    Returns:
        A result of the command execution, format like (stderr=..., stdout=..., exit_code=..., error=...)
    """
    sandbox = None
    try:
        sandbox = _connect_to_sandbox(sandbox_id)
    except Exception:
        return f"[ERROR]: Failed to connect to sandbox {sandbox_id}, retry later. Make sure the sandbox is created and the id is correct."

    for attempt in range(1, COMMAND_MAX_RETRIES + 1):
        try:
            _safe_set_timeout(sandbox)
            
            # Use commands.run() instead of run_code()
            # Execute Python code via shell command
            command = f'python3 -c {shlex.quote(code_block)}'
            result = sandbox.commands.run(command)
            
            return str(result)
        except Exception as e:
            if attempt == COMMAND_MAX_RETRIES:
                return f"[ERROR]: Failed to run code in sandbox {sandbox_id} after {COMMAND_MAX_RETRIES} attempts. Exception type: {type(e).__name__}, Details: {e}."
            await asyncio.sleep(attempt * 2)
        finally:
            _safe_set_timeout(sandbox)

@mcp.tool()
async def upload_file_from_local_to_sandbox(
    sandbox_id: str, local_file_path: str, sandbox_file_path: str = "/home/user"
) -> str:
    """Upload a local file to the `/home/user` dir of the sandbox.

    Args:
        sandbox_id: The id of the existing sandbox to update files in. To have a sandbox, use tool `create_sandbox`.
        local_file_path: The local path of the file to upload.
        sandbox_file_path: The path of directory to upload the file to in the sandbox. Default is `/home/user/`.

    Returns:
        The path of the uploaded file in the sandbox if the upload is successful.
    """
    sandbox = None
    try:
        sandbox = _connect_to_sandbox(sandbox_id)
    except Exception:
        return f"[ERROR]: Failed to connect to sandbox {sandbox_id}, retry later. Make sure the sandbox is created and the id is correct."

    if not _is_valid_local_file(local_file_path):
        return f"[ERROR]: Local file does not exist or is not a file: {local_file_path}"

    target_dir = _normalize_sandbox_path(sandbox_file_path)
    try:
        _safe_set_timeout(
            sandbox
        )  # refresh the timeout for each command execution

        # Get the uploaded file path
        uploaded_file_path = os.path.join(
            target_dir, os.path.basename(local_file_path)
        )

        # Upload the file
        with open(local_file_path, "rb") as f:
            sandbox.files.write(uploaded_file_path, f)

        return f"File uploaded to {uploaded_file_path}\n\n[INFO]: For directly reading local files without uploading to sandbox, consider using the `read_file` tool which can read various file types (Doc, PPT, PDF, Excel, CSV, ZIP, etc.) directly from local paths or URLs. Note that `read_file` doesn't support files already in the sandbox."
    except Exception as e:
        return f"[ERROR]: Failed to upload file {local_file_path} to sandbox {sandbox_id}: {e}\n\n[INFO]: This tool is for uploading local files to the sandbox. For security reasons, downloading files from sandbox to local system is not supported. Alternatively, consider using the `read_file` tool which can directly read various file types (Doc, PPT, PDF, Excel, CSV, ZIP, etc.) from local paths or URLs without uploading to sandbox."
    finally:
        # Set timeout before exit to prevent timeout after function exits
        _safe_set_timeout(sandbox)


@mcp.tool()
async def download_file_from_internet_to_sandbox(
    sandbox_id: str, url: str, sandbox_file_path: str = "/home/user"
) -> str:
    """Download a file from the internet to the `/home/user` dir of the sandbox.
    You should use this tool to download files from the internet.

    Args:
        sandbox_id: The id of the existing sandbox to download the file to. To have a sandbox, use tool `create_sandbox`.
        url: The URL of the file to download.
        sandbox_file_path: The path of directory to download the file to in the sandbox. Default is `/home/user/`.

    Returns:
        The path of the downloaded file in the sandbox if the download is successful.
    """
    sandbox = None
    try:
        sandbox = _connect_to_sandbox(sandbox_id)
    except Exception:
        return f"[ERROR]: Failed to connect to sandbox {sandbox_id}, retry later. Make sure the sandbox is created and the id is correct."

    if not url or not url.lower().startswith(("http://", "https://")):
        return "[ERROR]: Invalid URL. Only http/https URLs are supported."

    target_dir = _normalize_sandbox_path(sandbox_file_path)

    try:
        _safe_set_timeout(
            sandbox
        )  # refresh the timeout for each command execution

        downloaded_file_path = os.path.join(target_dir, os.path.basename(url))

        # Download the file with retry logic
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            result = sandbox.commands.run(f"wget --timeout={NETWORK_TIMEOUT_SECONDS} {url} -O {downloaded_file_path}")
            if result.exit_code == 0:
                return f"File downloaded to {downloaded_file_path}\n\n[INFO]: For directly reading files from internet URLs without downloading to sandbox, consider using the `read_file` tool which can read various file types (Doc, PPT, PDF, Excel, CSV, ZIP, etc.) directly from URLs. Note that `read_file` doesn't support files already in the sandbox."
            elif attempt < max_retries:
                await asyncio.sleep(4**attempt)
                continue  # Retry
            else:
                return f"[ERROR]: Failed to download file from {url} to {downloaded_file_path} after {max_retries} attempts: {result}.\n\n[INFO]: This tool is for downloading files from the internet to the sandbox. To upload local files to the sandbox, use `upload_file_from_local_to_sandbox` instead. Alternatively, consider using the `read_file` tool which can directly read various file types (Doc, PPT, PDF, Excel, CSV, ZIP, etc.) from internet URLs without downloading to sandbox."
    except Exception as e:
        return f"[ERROR]: Failed to download file from {url}: {e}\n\n[INFO]: This tool is for downloading files from the internet to the sandbox. To upload local files to the sandbox, use `upload_file_from_local_to_sandbox` instead. Alternatively, consider using the `read_file` tool which can directly read various file types (Doc, PPT, PDF, Excel, CSV, ZIP, etc.) from internet URLs without downloading to sandbox."
    finally:
        # Set timeout before exit to prevent timeout after function exits
        _safe_set_timeout(sandbox)


@mcp.tool()
async def download_file_from_sandbox_to_local(
    sandbox_id: str, sandbox_file_path: str, local_filename: str = None
) -> str:
    """Download a file from the sandbox to local system. Files in sandbox cannot be processed by tools from other servers - only local files and internet URLs can be processed by them.

    Args:
        sandbox_id: The id of the sandbox to download the file from. To have a sandbox, use tool `create_sandbox`.
        sandbox_file_path: The path of the file to download on the sandbox.
        local_filename: Optional filename to save as. If not provided, uses the original filename from sandbox_file_path.

    Returns:
        The local path of the downloaded file if successful, otherwise error message.
    """
    sandbox = None
    try:
        sandbox = _connect_to_sandbox(sandbox_id)
    except Exception:
        return f"[ERROR]: Failed to connect to sandbox {sandbox_id}, retry later. Make sure the sandbox is created and the id is correct."

    try:
        _safe_set_timeout(
            sandbox
        )  # refresh the timeout for each command execution

        # Create tmpfiles directory if it doesn't exist
        logs_root = _ensure_logs_dir()
        tmpfiles_dir = os.path.join(logs_root, "tmpfiles")
        os.makedirs(tmpfiles_dir, exist_ok=True)

        # Determine local filename
        if local_filename is None or local_filename.strip() == "":
            local_filename = os.path.basename(sandbox_file_path)

        local_file_path = os.path.join(
            tmpfiles_dir, f"sandbox_{sandbox_id}_{local_filename}"
        )

        # Download the file
        with open(local_file_path, "wb") as f:
            content = sandbox.files.read(sandbox_file_path, format="bytes")
            f.write(content)

        return f"File downloaded successfully to: {local_file_path}\n\n[INFO]: The file can now be accessed by other tools (reading, question-answering, etc.) which only support local files and internet URLs, not sandbox files."
    except Exception as e:
        return f"[ERROR]: Failed to download file {sandbox_file_path} from sandbox {sandbox_id}: {e}\n\n[INFO]: This tool is for downloading files from the sandbox to local system. To upload local files to the sandbox, use `upload_file_from_local_to_sandbox` instead."
    finally:
        # Set timeout before exit to prevent timeout after function exits
        _safe_set_timeout(sandbox)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Python MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="sse",
        help="Transport method: 'stdio' or 'sse' (default: sse)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to use when running with SSE transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8931,
        help="Port to use when running with SSE transport (default: 8931)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", host=args.host, port=args.port)
