import logging, colorlog, os



logger = logging.getLogger("custom_logger")
logger.setLevel(logging.DEBUG)  # Capture all levels

# Logging functions
def log_debug(message):
    """Log a debug message."""
    logger.debug(message)

def log_info(message):
    """Log an informational message."""
    logger.info(message)

def log_warning(message):
    """Log a warning message."""
    logger.warning(message)

def log_error(message):
    """Log an error message."""
    logger.error(message)

def log_critical(message):
    """Log a critical error message."""
    logger.critical(message)