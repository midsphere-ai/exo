import logging
import os
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("deepagent")


def setup_file_logging(log_file_path: str = "super_agent_test_run.log", simple_format: bool = True) -> None:
    """Configure the deepagent logger to write to a rotating file with real-time flushing.

    Adds a ``RotatingFileHandler`` to the ``deepagent`` logger so that all
    calls to ``logger.info()``, etc. are written to both the console (if a
    console handler is present) and the specified log file.

    Args:
        log_file_path: Path to the log file (default: "super_agent_test_run.log").
        simple_format: If True, use a clean ``timestamp | level | message``
            format. If False, include the module name as well.
    """
    # Ensure the logger itself is at DEBUG so handlers control filtering
    logger.setLevel(logging.DEBUG)

    # Add a console handler if none exist yet
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(console)

    # Rotating file handler — matches original behaviour (20 MB, 20 backups)
    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=20 * 1024 * 1024,
        backupCount=20,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)

    if simple_format:
        fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    else:
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    file_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    logger.info("File logging enabled: %s", os.path.abspath(log_file_path))


def process_input(task_description, task_file_name):
    """
    Enriches task description with file context information.

    Args:
        task_description: The original task description
        task_file_name: Optional file path associated with the task

    Returns:
        Enhanced task description with file handling instructions
    """
    # TODO: Support URL differentiation (YouTube, Wikipedia, general URLs)

    if not task_file_name:
        return task_description

    # Validate file existence
    if not os.path.isfile(task_file_name):
        raise FileNotFoundError(f"Error: File not found {task_file_name}")

    # Map file extensions to descriptive types
    extension_mappings = {
        'image': ['jpg', 'jpeg', 'png', 'gif', 'webp'],
        'text': ['txt'],
        'json': ['json', 'jsonld'],
        'excel': ['xlsx', 'xls'],
        'pdf': ['pdf'],
        'document': ['docx', 'doc'],
        'html': ['html', 'htm'],
        'ppt': ['pptx', 'ppt'],
        'wav': ['wav'],
        'mp3': ['mp3', 'm4a'],
        'zip': ['zip']
    }

    # Extract and normalize file extension
    ext = task_file_name.rsplit('.', 1)[-1].lower()

    # Determine file type category
    file_category = ext  # Default to extension itself
    for category, extensions in extension_mappings.items():
        if ext in extensions:
            file_category = category.capitalize()
            break

    # Append file context and usage instructions
    file_instruction = (
        f"\nNote: A {file_category} file '{task_file_name}' is associated with this task. "
        f"You should use available tools to read its content if necessary through {task_file_name}. "
        f"Additionally, if you need to analyze this file by Linux commands or python codes, "
        f"you should upload it to the sandbox first. Files in the sandbox cannot be accessed by other tools.\n\n"
    )

    return task_description + file_instruction

