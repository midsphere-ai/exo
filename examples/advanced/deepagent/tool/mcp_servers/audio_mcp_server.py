import os
import tempfile
import requests
from urllib.parse import urlparse
from fastmcp import FastMCP
from openai import OpenAI
import base64
import mimetypes
import wave
import contextlib
from mutagen import File as MutagenFile
import asyncio
import time
import hashlib
import hmac
import json

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ACR_ACCESS_KEY = os.environ.get("ACR_ACCESS_KEY", "")
ACR_ACCESS_SECRET = os.environ.get("ACR_ACCESS_SECRET", "")
ACR_BASE_URL = os.environ.get("ACR_BASE_URL", "https://identify-ap-southeast-1.acrcloud.com/v1/identify")
HTTP_TIMEOUT = 20
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25MB safety limit
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Initialize FastMCP server
mcp = FastMCP("audio-mcp-server")


def _get_audio_extension(url: str, content_type: str = None) -> str:
    """
    Determine the appropriate audio file extension from URL or content type.

    Args:
        url: The URL of the audio file
        content_type: The content type from HTTP headers

    Returns:
        File extension (with dot) to use for temporary file
    """
    # First try to get extension from URL
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()

    # Common audio extensions
    audio_extensions = [".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma"]
    for ext in audio_extensions:
        if path.endswith(ext):
            return ext

    # If no extension found in URL, try content type
    if content_type:
        content_type = content_type.lower()
        if "mp3" in content_type or "mpeg" in content_type:
            return ".mp3"
        elif "wav" in content_type:
            return ".wav"
        elif "m4a" in content_type:
            return ".m4a"
        elif "aac" in content_type:
            return ".aac"
        elif "ogg" in content_type:
            return ".ogg"
        elif "flac" in content_type:
            return ".flac"

    # Default fallback to mp3
    return ".mp3"


def _get_audio_duration(audio_path: str) -> float:
    """
    Get audio duration in seconds.

    Tries to use wave (for .wav), then falls back to mutagen (for mp3, etc).
    """
    # Try using wave for .wav files
    try:
        with contextlib.closing(wave.open(audio_path, "rb")) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            if duration > 0:
                return duration
    except Exception:
        pass  # Not a wav file or failed

    # Try using mutagen for other audio formats (mp3, etc)
    try:
        audio = MutagenFile(audio_path)
        if (
            audio is not None
            and hasattr(audio, "info")
            and hasattr(audio.info, "length")
        ):
            duration = float(audio.info.length)
            if duration > 0:
                return duration
    except Exception:
        pass

    raise ValueError("Unable to determine audio duration")


def _encode_audio_file(audio_path: str) -> tuple[str, str]:
    """Encode audio file to base64 and determine format."""
    with open(audio_path, "rb") as audio_file:
        audio_data = audio_file.read()
        encoded_string = base64.b64encode(audio_data).decode("utf-8")

    # Determine file format from file extension
    mime_type, _ = mimetypes.guess_type(audio_path)
    if mime_type and mime_type.startswith("audio/"):
        mime_format = mime_type.split("/")[-1]
        # Map MIME type formats to OpenAI supported formats
        format_mapping = {
            "mpeg": "mp3",  # audio/mpeg -> mp3
            "wav": "wav",  # audio/wav -> wav
            "wave": "wav",  # audio/wave -> wav
        }
        file_format = format_mapping.get(mime_format, "mp3")
    else:
        # Default to mp3 if we can't determine
        file_format = "mp3"

    return encoded_string, file_format


def _fetch_audio_to_temp(url: str) -> str:
    """Download audio from URL into a temp file with size and timeout safeguards."""
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    with requests.get(url, headers=headers, timeout=HTTP_TIMEOUT, stream=True) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        file_extension = _get_audio_extension(url, content_type)
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            bytes_written = 0
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    bytes_written += len(chunk)
                    if bytes_written > MAX_AUDIO_BYTES:
                        raise ValueError("Audio file exceeds size limit (25MB).")
                    temp_file.write(chunk)
            return temp_file.name


@mcp.tool()
async def audio_transcription(audio_path_or_url: str) -> str:
    """
    Utilises OpenAI API to transcribe audio file to text and return the transcription.
    Args:
        audio_path_or_url: The path of the audio file locally or its URL. Path from sandbox is not supported. YouTube URL is not supported.

    Returns:
        The transcription of the audio file.
    """
    max_retries = 3
    retry = 0
    transcription = None

    if not OPENAI_API_KEY:
        return "[ERROR]: OPENAI_API_KEY is not configured for audio transcription."

    while retry < max_retries:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            if os.path.exists(audio_path_or_url):  # Check if the file exists locally
                with open(audio_path_or_url, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="gpt-4o-transcribe", file=audio_file
                    )
            elif "home/user" in audio_path_or_url:
                return "The audio_transcription tool cannot access to sandbox file, please use the local path provided by original instruction"
            else:
                temp_audio_path = _fetch_audio_to_temp(audio_path_or_url)
                try:
                    with open(temp_audio_path, "rb") as audio_file:
                        transcription = client.audio.transcriptions.create(
                            model="gpt-4o-mini-transcribe", file=audio_file
                        )
                finally:
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
            break

        except requests.RequestException as e:
            retry += 1
            if retry >= max_retries:
                return f"[ERROR]: Audio transcription failed: Failed to download audio file - {e}.\nNote: Files from sandbox are not available. You should use local path given in the instruction. \nURLs must include the proper scheme (e.g., 'https://') and be publicly accessible. The file should be in a common audio format such as MP3, WAV, or M4A.\nNote: YouTube video URL is not supported."
            await asyncio.sleep(5 * (2**retry))
        except Exception as e:
            retry += 1
            if retry >= max_retries:
                return f"[ERROR]: Audio transcription failed: {e}\nNote: Files from sandbox are not available. You should use local path given in the instruction. The file should be in a common audio format such as MP3, WAV, or M4A.\nNote: YouTube video URL is not supported."
            await asyncio.sleep(5 * (2**retry))

    return transcription.text


@mcp.tool()
async def audio_question_answering(audio_path_or_url: str, question: str) -> str:
    """
    Answer the question based on the given audio information. Please note that this tool cannot be used for audio name recognition, and instead focuses more on the details of the audio content.

    Args:
        audio_path_or_url: The path of the audio file locally or its URL. Path from sandbox is not supported. YouTube URL is not supported.
        question: The question to answer.

    Returns:
        The answer to the question, and the duration of the audio file.
    """
    if not OPENAI_API_KEY:
        return "[ERROR]: OPENAI_API_KEY is not configured for audio question answering."

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        text_prompt = f"""Answer the following question based on the given audio information:\n\n{question}"""

        if os.path.exists(audio_path_or_url):  # Check if the file exists locally
            encoded_string, file_format = _encode_audio_file(audio_path_or_url)
            duration = _get_audio_duration(audio_path_or_url)
        elif "home/user" in audio_path_or_url:
            return "The audio_question_answering tool cannot access to sandbox file, please use the local path provided by original instruction"
        else:
            temp_audio_path = _fetch_audio_to_temp(audio_path_or_url)
            try:
                encoded_string, file_format = _encode_audio_file(temp_audio_path)
                duration = _get_audio_duration(temp_audio_path)
            finally:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

        if encoded_string is None or file_format is None:
            return "[ERROR]: Audio question answering failed: Failed to encode audio file.\nNote: Files from sandbox are not available. You should use local path given in the instruction. \nURLs must include the proper scheme (e.g., 'https://') and be publicly accessible. The file should be in a common audio format such as MP3.\nNote: YouTube video URL is not supported."

        response = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant specializing in audio analysis.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encoded_string,
                                "format": file_format,
                            },
                        },
                    ],
                },
            ],
        )
    except Exception as e:
        return f"[ERROR]: Audio question answering failed when calling OpenAI API: {e}\nNote: Files from sandbox are not available. You should use local path given in the instruction. The file should be in a common audio format such as MP3, WAV, or M4A.\nNote: YouTube video URL is not supported."

    response = response.choices[0].message.content
    response += f"\n\nAudio duration: {duration} seconds"

    return response



@mcp.tool()
async def audio_metadata(audio_path_or_url: str) -> str:
    """
    Identify the metadata (name, author, year) of the given audio file using the ACRCloud API. 
    You are highly recommended to cut large files into snippets of less than 15 seconds for better identification results, otherwise the file will be rejected.

    Args:
        audio_path_or_url: The path of the audio file locally or its URL. Path from sandbox is not supported. YouTube URL is not supported.

    Returns:
        Metadata (name, author, year) of the audio file, or duration-only if ACR keys are absent.
    """

    try:
        download_file = False

        if os.path.exists(audio_path_or_url):  # Check if the file exists locally
            audio_path_or_url = audio_path_or_url
        elif "home/user" in audio_path_or_url:
            return "The audio_question_answering tool cannot access to sandbox file, please use a local path instead. \
                If the audio file has been preprocessed in the sandbox, please download the file to your local machine and use the local path instead."
        else:
            temp_audio_path = _fetch_audio_to_temp(audio_path_or_url)
            audio_path_or_url = temp_audio_path
            download_file = True

        duration = _get_audio_duration(audio_path_or_url)
        # If no ACR credentials, return duration-only.
        if not ACR_ACCESS_KEY or not ACR_ACCESS_SECRET:
            return f"Duration (seconds): {duration:.2f}\nNote: Title/artist identification is disabled because ACR credentials are not provided."

        # If the audio file is too long, return a warning message
        if duration > 15:
            return "The audio_metadata tool is better used to process audio file with less than 15 seconds, please cut your audio file to a small one and try again."


        requrl = ACR_BASE_URL
        http_method = "POST"
        http_uri = "/v1/identify"
        data_type = "audio"
        signature_version = "1"
        timestamp = time.time()

        string_to_sign = (
            http_method + "\n" + http_uri + "\n" + ACR_ACCESS_KEY + "\n" + data_type + "\n" + signature_version + "\n" + str(timestamp)
        )
        sign = base64.b64encode(
            hmac.new(ACR_ACCESS_SECRET.encode("ascii"), string_to_sign.encode("ascii"), digestmod=hashlib.sha1).digest()
        ).decode("ascii")

        file_name = os.path.basename(audio_path_or_url)
        sample_bytes = os.path.getsize(audio_path_or_url)
        mime_type, _ = mimetypes.guess_type(audio_path_or_url)
        if mime_type and mime_type.startswith("audio/"):
            mime_format = mime_type.split("/")[-1]
            format_mapping = {"mpeg": "mp3", "wav": "wav", "wave": "wav"}
            file_format = format_mapping.get(mime_format, "mp3")
        else:
            file_format = "mp3"

        files = [("sample", (file_name, open(audio_path_or_url, "rb"), file_format))]
        data = {
            "access_key": ACR_ACCESS_KEY,
            "sample_bytes": sample_bytes,
            "timestamp": str(timestamp),
            "signature": sign,
            "data_type": data_type,
            "signature_version": signature_version,
        }

        r = requests.post(requrl, files=files, data=data, timeout=HTTP_TIMEOUT)
        r.encoding = "utf-8"
        f = json.loads(r.text)

        if "humming" in f.get("metadata", {}):
            datas = f["metadata"]["humming"]
            filters = []
            for i in datas:
                score = i.get("score")
                title = i.get("title")
                name = i.get("artists", [{}])[0].get("name")
                release_date = i.get("release_date")
                duration_ms = i.get("duration_ms")
                filters.append((duration_ms, title, name, release_date, score))
            filters = sorted(filters, key=lambda x: x[0] or 0, reverse=True)
            best = filters[0]
            return f"Name: {best[1]}, Artist: {best[2]}, Release Date: {best[3]}. Note: score={best[4]}"
        elif "music" in f.get("metadata", {}):
            datas = f["metadata"]["music"][0]
            return f"Name: {datas['title']}, Artist: {datas['artists'][0]['name']}, Release Date: {datas['release_date']}."
        else:
            return f"Duration (seconds): {duration:.2f}\nACR: No metadata found for the given audio file."

    except Exception as e:
        return f"[ERROR]: Audio metadata identification failed: {e}\n"
    
    finally:
        if download_file and os.path.exists(audio_path_or_url):
            os.remove(audio_path_or_url)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio MCP Server")
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
        default=8933,
        help="Port to use when running with SSE transport (default: 8933)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", host=args.host, port=args.port)
