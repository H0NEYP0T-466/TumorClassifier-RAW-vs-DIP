import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = "TumorClassifier") -> logging.Logger:
    """
    Set up and configure logger with both file and console handlers.
    
    Args:
        name: Name of the logger
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).parent. parent. parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Log file with timestamp
    log_filename = logs_dir / f"app_{datetime.now().strftime('%Y-%m-%d')}.log"
    
    # Formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    simple_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # File handler - captures everything (DEBUG and above)
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler - INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger. addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Create a default logger instance
logger = setup_logger()