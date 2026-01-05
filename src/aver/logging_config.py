"""
AVER Benchmark - Structured Logging Configuration

Provides consistent logging across all AVER components with:
- Console output with color coding
- Optional file logging
- JSON structured logs for production
- Component-level log filtering
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class AVERFormatter(logging.Formatter):
    """Custom formatter with AVER prefix and optional colors"""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname

        # Add component name if available
        component = getattr(record, 'component', record.name.split('.')[-1])

        if self.use_colors:
            color = self.COLORS.get(level, '')
            reset = self.COLORS['RESET']
            prefix = f"{color}[AVER:{component}]{reset}"
        else:
            prefix = f"[AVER:{component}]"

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')

        return f"{timestamp} {prefix} {record.getMessage()}"


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging in production"""

    def format(self, record: logging.LogRecord) -> str:
        import json

        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'component': getattr(record, 'component', record.name.split('.')[-1]),
            'message': record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ('msg', 'args', 'created', 'levelname', 'levelno',
                          'pathname', 'filename', 'module', 'lineno', 'funcName',
                          'exc_info', 'exc_text', 'stack_info', 'component', 'name',
                          'thread', 'threadName', 'processName', 'process',
                          'message', 'msecs', 'relativeCreated'):
                log_data[key] = value

        return json.dumps(log_data)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    verbose: bool = False
) -> logging.Logger:
    """
    Configure AVER logging

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        json_format: Use JSON structured logging
        verbose: Enable DEBUG level for AVER components

    Returns:
        Root AVER logger
    """
    # Get or create root AVER logger
    logger = logging.getLogger('aver')

    # Clear existing handlers
    logger.handlers.clear()

    # Set level
    if verbose:
        level = "DEBUG"
    logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(AVERFormatter(use_colors=True))
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(AVERFormatter(use_colors=False))
        logger.addHandler(file_handler)

    # Don't propagate to root logger
    logger.propagate = False

    return logger


def get_logger(component: str) -> logging.Logger:
    """
    Get a logger for a specific AVER component

    Args:
        component: Component name (e.g., 'evaluator', 'green_agent')

    Returns:
        Logger for the component
    """
    return logging.getLogger(f'aver.{component}')


# Auto-configure on import based on environment
_log_level = os.environ.get('AVER_LOG_LEVEL', 'INFO')
_log_file = os.environ.get('AVER_LOG_FILE')
_json_logs = os.environ.get('AVER_JSON_LOGS', 'false').lower() == 'true'
_verbose = os.environ.get('AVER_VERBOSE', 'false').lower() == 'true'

# Setup logging on module import
root_logger = setup_logging(
    level=_log_level,
    log_file=_log_file,
    json_format=_json_logs,
    verbose=_verbose
)
