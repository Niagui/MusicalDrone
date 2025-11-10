import logging
import os
import colorlog

logger = logging.getLogger("clap_logger")
logger.setLevel(logging.DEBUG)  # Capture all levels

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, "pipeline.log"), mode="a")
file_handler.setLevel(logging.INFO)

color_formatter = colorlog.ColoredFormatter(
    fmt="%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG":    "cyan",
        "INFO":     "green",
        "WARNING":  "yellow",
        "ERROR":    "red",
        "CRITICAL": "bold_red",
    }
)

formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# --- Add handlers to logger ---
if not logger.handlers:  # prevent adding twice if re-imported
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

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

if __name__ == "__main__":
    log_debug("Debugging details.")
    log_info("Pipeline started successfully.")
    log_warning("Potential issue detected.")
    log_error("An error occurred during processing.")
    log_critical("Critical failure! Immediate attention required.")