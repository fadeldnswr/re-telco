'''
Logging utilities for the project.
'''
import sys
from loguru import logger
from pathlib import Path

# Configure loguru to write logs to a file with rotation and retention policies
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)

# Setup loguru logger
def setup_logger():
  # Remove default logger to avoid duplicate logs
  logger.remove()
  
  # Console logger
  logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
  )
  
  # File logger
  logger.add(
    LOG_DIR / "simulation.log",
    level="DEBUG",
    rotation="10 MB",  # Rotate log file after it reaches 10 MB
    retention="7 days",  # Retain log files for 7 days
    compression="zip",  # Compress old log files
    format="{time} | {level} | {name}:{function}:{line} - {message}"
  )
  
  # Error only log
  logger.add(
    LOG_DIR / "error.log",
    level="ERROR",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
  )
  
  return logger