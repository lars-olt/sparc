"""Logging utilities for SPARC pipeline."""

import logging
import sys
from ..utils.threading import configure_global_settings


def setup_logger(name: str = "sparc") -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def configure_logging(verbose: bool = False) -> None:
    """
    Configure global SPARC logging level and format.
    
    Args:
        verbose: If True, set level to DEBUG, else INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure both namespaces to handle different import styles
    # 'sparc' for direct imports, 'src.sparc' for project structure imports
    namespaces = ["sparc", "src.sparc"]
    
    for name in namespaces:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Prevent adding duplicate handlers if re-configured
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        else:
            for handler in logger.handlers:
                handler.setLevel(level)
                
    # Also ensure the matplotlib logger doesn't spam debug messages
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


# Configure environment (threading rules, warning suppression) on module import
configure_global_settings()

# Initialize default logging (INFO level)
configure_logging(verbose=False)