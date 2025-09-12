#!/usr/bin/env python3
"""
Unicode-safe logging configuration for all agents.
Fixes emoji character encoding issues on Windows systems.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional

def setup_unicode_logging(
    name: str = __name__,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup unicode-safe logging configuration.
    
    Args:
        name: Logger name
        log_level: Logging level
        log_file: Log file path (optional)
        console_output: Enable console output
    
    Returns:
        Configured logger instance
    """
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add file handler with UTF-8 encoding
    if log_file:
        file_handler = logging.FileHandler(
            log_file, 
            encoding='utf-8', 
            mode='a'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler with UTF-8 support
    if console_output:
        # Try to configure stdout for UTF-8
        try:
            if sys.stdout.encoding != 'utf-8':
                sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            # Python < 3.7 doesn't have reconfigure
            pass
        except Exception:
            # Fallback: just proceed with default encoding
            pass
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def sanitize_unicode_for_logging(text: str) -> str:
    """
    Remove problematic unicode characters for logging.
    
    Args:
        text: Input text with potential unicode issues
        
    Returns:
        Sanitized text safe for Windows console
    """
    # Replace common problematic emoji/unicode chars
    replacements = {
        '🔎': '[SEARCH]',
        '✅': '[OK]',
        '✗': '[FAIL]', 
        '→': '->',
        '←': '<-',
        '📊': '[DATA]',
        '⚠️': '[WARN]',
        '🚨': '[ALERT]',
        '💡': '[INFO]',
        '🔧': '[TOOL]',
        '📝': '[NOTE]',
        '🎯': '[TARGET]',
        '⭐': '[STAR]',
        '🔥': '[HOT]',
        '💻': '[COMP]',
        '📱': '[MOBILE]',
        '🌐': '[WEB]',
        '🔒': '[SECURE]',
        '🔓': '[UNLOCK]'
    }
    
    result = text
    for emoji, replacement in replacements.items():
        result = result.replace(emoji, replacement)
    
    # Remove any remaining problematic unicode
    try:
        result.encode('ascii', errors='ignore').decode('ascii')
    except:
        # If still problematic, use more aggressive cleaning
        result = result.encode('ascii', errors='ignore').decode('ascii')
    
    return result

class UnicodeLogger:
    """
    Wrapper logger that auto-sanitizes unicode characters.
    """
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = setup_unicode_logging(
            name=name,
            log_file=log_file or f"logs/{name.split('.')[-1]}.log"
        )
    
    def _sanitize_args(self, *args):
        """Sanitize all arguments for safe logging."""
        return tuple(
            sanitize_unicode_for_logging(str(arg)) if isinstance(arg, str) else arg 
            for arg in args
        )
    
    def info(self, msg, *args, **kwargs):
        sanitized_args = self._sanitize_args(*args)
        sanitized_msg = sanitize_unicode_for_logging(str(msg))
        self.logger.info(sanitized_msg, *sanitized_args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        sanitized_args = self._sanitize_args(*args)
        sanitized_msg = sanitize_unicode_for_logging(str(msg))
        self.logger.warning(sanitized_msg, *sanitized_args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        sanitized_args = self._sanitize_args(*args)
        sanitized_msg = sanitize_unicode_for_logging(str(msg))
        self.logger.error(sanitized_msg, *sanitized_args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        sanitized_args = self._sanitize_args(*args)
        sanitized_msg = sanitize_unicode_for_logging(str(msg))
        self.logger.debug(sanitized_msg, *sanitized_args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        sanitized_args = self._sanitize_args(*args)
        sanitized_msg = sanitize_unicode_for_logging(str(msg))
        self.logger.critical(sanitized_msg, *sanitized_args, **kwargs)

# Pre-configured loggers for common components
def get_orchestrator_logger():
    return UnicodeLogger("orchestrator", "logs/orchestrator.log")

def get_fallback_logger():
    return UnicodeLogger("fallback", "logs/fallback.log")

def get_api_logger():
    return UnicodeLogger("research_agent_api", "logs/api.log")

def get_fda_logger():
    return UnicodeLogger("fda_agent", "logs/fda_agent.log")

def get_clinical_trials_logger():
    return UnicodeLogger("clinical_trials_agent", "logs/clinical_trials.log")

def get_pubmed_logger():
    return UnicodeLogger("pubmed_agent", "logs/pubmed.log")

def get_local_logger():
    return UnicodeLogger("local_agent", "logs/local_agent.log")

# Global configuration function
def configure_all_loggers():
    """Configure all loggers with unicode safety."""
    import logging
    
    # Set root logger configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/system.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    print("Unicode-safe logging configured for all components")

if __name__ == "__main__":
    # Test the unicode logging
    test_logger = UnicodeLogger("test")
    
    # Test problematic unicode
    test_logger.info("🔎 Testing unicode characters ✅")
    test_logger.warning("→ This should work now ✗")
    test_logger.error("📊 Data processing complete 🎯")
    
    print("Unicode logging test completed - check logs/test.log")