"""
Progress Logger - Beautiful Progress Bars with File Logging
Redirects technical logs to files while showing user-friendly progress
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Optional, Any, Dict, List
from contextlib import contextmanager
import threading

class ProgressLogger:
    """Manages progress display and file logging"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.setup_logging()
        self.current_progress = 0
        self.total_progress = 100
        self.status_message = ""
        self.start_time = None
        self._lock = threading.Lock()
        
    def setup_logging(self):
        """Setup file logging configuration"""
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"ct_analysis_{timestamp}.log")
        
        # Configure file logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.NullHandler()  # Suppress console output
            ]
        )
        
        self.logger = logging.getLogger("CTReader")
        
    def log_to_file(self, message: str, level: str = "INFO"):
        """Log message to file only"""
        if level == "ERROR":
            self.logger.error(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "DEBUG":
            self.logger.debug(message)
        else:
            self.logger.info(message)
    
    def start_progress(self, total: int, description: str = "Обработка"):
        """Start progress tracking"""
        self.total_progress = total
        self.current_progress = 0
        self.status_message = description
        self.start_time = time.time()
        self.log_to_file(f"Starting progress: {description} (total: {total})")
        
    def update_progress(self, current: int, status: str = ""):
        """Update progress bar"""
        with self._lock:
            self.current_progress = current
            if status:
                self.status_message = status
                
            # Calculate progress percentage
            percentage = (current / self.total_progress) * 100 if self.total_progress > 0 else 0
            
            # Calculate elapsed time and ETA
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            eta = (elapsed_time / current * (self.total_progress - current)) if current > 0 else 0
            
            # Create progress bar
            bar_length = 30
            filled_length = int(bar_length * percentage / 100)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            
            # Format time
            elapsed_str = self._format_time(elapsed_time)
            eta_str = self._format_time(eta)
            
            # Print progress (overwrite previous line)
            progress_line = f"\r🔄 {self.status_message}: [{bar}] {percentage:.1f}% ({current}/{self.total_progress}) | ⏱️ {elapsed_str} | ETA: {eta_str}"
            print(progress_line, end="", flush=True)
            
            # Log to file
            self.log_to_file(f"Progress: {current}/{self.total_progress} ({percentage:.1f}%) - {status}")
    
    def complete_progress(self, message: str = "Завершено"):
        """Complete progress tracking"""
        with self._lock:
            self.current_progress = self.total_progress
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            elapsed_str = self._format_time(elapsed_time)
            
            # Final progress bar
            bar = "█" * 30
            print(f"\r✅ {message}: [{bar}] 100.0% ({self.total_progress}/{self.total_progress}) | ⏱️ {elapsed_str}          ")
            
            self.log_to_file(f"Progress completed: {message} (total time: {elapsed_str})")
    
    def show_step(self, step_name: str, step_number: int = None, total_steps: int = None):
        """Show current step"""
        if step_number and total_steps:
            step_info = f"[{step_number}/{total_steps}] "
        else:
            step_info = ""
            
        print(f"\n🔧 {step_info}{step_name}")
        self.log_to_file(f"Step: {step_info}{step_name}")
    
    def show_success(self, message: str):
        """Show success message"""
        print(f"✅ {message}")
        self.log_to_file(f"SUCCESS: {message}")
    
    def show_warning(self, message: str):
        """Show warning message"""
        print(f"⚠️ {message}")
        self.log_to_file(f"WARNING: {message}")
    
    def show_error(self, message: str):
        """Show error message"""
        print(f"❌ {message}")
        self.log_to_file(f"ERROR: {message}")
    
    def show_info(self, message: str):
        """Show info message"""
        print(f"ℹ️ {message}")
        self.log_to_file(f"INFO: {message}")
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    @contextmanager
    def suppress_prints(self):
        """Context manager to suppress print statements"""
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        class LogCapture:
            def __init__(self, logger, level="INFO"):
                self.logger = logger
                self.level = level
                self.content = []
                
            def write(self, text):
                if text.strip():
                    self.content.append(text.strip())
                    self.logger.log_to_file(text.strip(), self.level)
                    
            def flush(self):
                pass
        
        try:
            sys.stdout = LogCapture(self, "INFO")
            sys.stderr = LogCapture(self, "ERROR")
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    
    def get_log_file_path(self) -> str:
        """Get current log file path"""
        return self.log_file

# Global progress logger instance
progress_logger = ProgressLogger()

# Convenience functions
def start_progress(total: int, description: str = "Обработка"):
    """Start progress tracking"""
    progress_logger.start_progress(total, description)

def update_progress(current: int, status: str = ""):
    """Update progress"""
    progress_logger.update_progress(current, status)

def complete_progress(message: str = "Завершено"):
    """Complete progress"""
    progress_logger.complete_progress(message)

def show_step(step_name: str, step_number: int = None, total_steps: int = None):
    """Show current step"""
    progress_logger.show_step(step_name, step_number, total_steps)

def show_success(message: str):
    """Show success message"""
    progress_logger.show_success(message)

def show_warning(message: str):
    """Show warning message"""
    progress_logger.show_warning(message)

def show_error(message: str):
    """Show error message"""
    progress_logger.show_error(message)

def show_info(message: str):
    """Show info message"""
    progress_logger.show_info(message)

def log_to_file(message: str, level: str = "INFO"):
    """Log message to file"""
    progress_logger.log_to_file(message, level)

def suppress_prints():
    """Context manager to suppress prints"""
    return progress_logger.suppress_prints()

def get_log_file() -> str:
    """Get current log file path"""
    return progress_logger.get_log_file_path() 