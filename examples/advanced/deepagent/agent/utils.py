import os
import logging
from typing import Optional


def setup_file_logging(log_file_path: str = "super_agent_test_run.log", simple_format: bool = True) -> None:
    """
    Configure the openjiuwen logger to write to a file using the built-in logging system.
    Logs will be written in real-time as they occur (auto-flushed).
    
    This adds file output to the existing "common" logger, so all existing logger calls
    will automatically write to both console and file.
    
    Args:
        log_file_path: Path to the log file (default: "super_agent_test_run.log")
        simple_format: If True, use a cleaner, simpler format for file logs (default: True)
    """
    from openjiuwen.core.common.logging import logger, LogManager
    
    try:
        from openjiuwen.extensions.common.configs.log_config import log_config
        
        # Get the existing "common" logger config and update it to include file output
        common_config = log_config.get_common_config()
        common_config['log_file'] = log_file_path
        common_config['output'] = ["console", "file"]
        
        # Use a simpler format for file logs if requested
        if simple_format:
            # Clean format: timestamp | level | message (no filename, line number, function name)
            # This makes logs much more readable
            common_config['format'] = '%(asctime)s | %(levelname)-8s | %(message)s'
        
        # Recreate the common logger with file output enabled
        from openjiuwen.extensions.common.log.default_impl import DefaultLogger
        updated_logger = DefaultLogger("common", common_config)
        LogManager.register_logger("common", updated_logger)
        
        # If simple_format is requested, replace the file handler with a simpler formatter
        if simple_format:
            actual_logger = updated_logger._logger
            # Find and replace the file handler
            for handler in actual_logger.handlers[:]:
                if isinstance(handler, logging.FileHandler) or hasattr(handler, 'baseFilename'):
                    # Remove the old file handler
                    actual_logger.removeHandler(handler)
                    handler.close()
                    
                    # Create a new file handler with simple format
                    from openjiuwen.extensions.common.log.default_impl import SafeRotatingFileHandler
                    backup_count = common_config.get('backup_count', 20)
                    max_bytes = common_config.get('max_bytes', 20 * 1024 * 1024)
                    
                    simple_file_handler = SafeRotatingFileHandler(
                        filename=log_file_path,
                        maxBytes=max_bytes,
                        backupCount=backup_count,
                        encoding='utf-8'
                    )
                    simple_file_handler.setLevel(logging.DEBUG)
                    
                    # Use a simple formatter (not CallerAwareFormatter)
                    simple_formatter = logging.Formatter(
                        '%(asctime)s | %(levelname)-8s | %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                    )
                    simple_file_handler.setFormatter(simple_formatter)
                    
                    # Add filter if needed
                    from openjiuwen.extensions.common.log.default_impl import ThreadContextFilter
                    simple_file_handler.addFilter(ThreadContextFilter("common"))
                    
                    actual_logger.addHandler(simple_file_handler)
                    break
        
        logger.info(f"File logging enabled: {os.path.abspath(log_file_path)}")
    except Exception as e:
        # Fallback: if openjiuwen config system isn't available, use manual handler
        logger.warning(f"Could not use openjiuwen logger config: {e}. Using fallback method.")
        
        # Force logger initialization
        _ = logger.info
        
        # Get the underlying Python logger from the common logger
        actual_logger = None
        if hasattr(logger, '_logger'):
            actual_logger = logger._logger
            if hasattr(actual_logger, '_logger'):
                actual_logger = actual_logger._logger
        
        # Create file handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.DEBUG)
        
        # Use a clean, simple format for file logging
        if simple_format:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        file_handler.setFormatter(formatter)
        
        # Add handler to the appropriate logger
        if actual_logger:
            actual_logger.addHandler(file_handler)
        else:
            # Fallback: add to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
            root_logger.setLevel(logging.DEBUG)
        
        logger.info(f"File logging enabled (fallback): {os.path.abspath(log_file_path)}")


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

