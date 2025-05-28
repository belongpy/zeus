"""
Zeus Utilities - Helper Functions and Common Operations
Collection of utility functions used throughout Zeus system

Features:
- Wallet address validation
- Data formatting and conversion
- Time and date utilities
- File operations
- Performance monitoring
- Error handling helpers
"""

import os
import re
import time
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import base58

logger = logging.getLogger("zeus.utils")

class ZeusUtils:
    """Collection of utility functions for Zeus system."""
    
    @staticmethod
    def validate_solana_address(address: str) -> Dict[str, Any]:
        """
        Validate Solana wallet/token address.
        
        Args:
            address: Address to validate
            
        Returns:
            Dict with validation result and details
        """
        try:
            if not address or not isinstance(address, str):
                return {
                    'valid': False,
                    'error': 'Address must be a non-empty string'
                }
            
            # Remove whitespace
            address = address.strip()
            
            # Check length (Solana addresses are typically 32-44 characters)
            if not (32 <= len(address) <= 44):
                return {
                    'valid': False,
                    'error': f'Address length {len(address)} invalid (expected 32-44 characters)'
                }
            
            # Check base58 encoding
            try:
                decoded = base58.b58decode(address)
                if len(decoded) != 32:
                    return {
                        'valid': False,
                        'error': 'Decoded address must be 32 bytes'
                    }
            except Exception:
                return {
                    'valid': False,
                    'error': 'Invalid base58 encoding'
                }
            
            # Additional checks for common invalid patterns
            if address == '1' * len(address):
                return {
                    'valid': False,
                    'error': 'Address cannot be all same character'
                }
            
            return {
                'valid': True,
                'address': address,
                'length': len(address),
                'type': 'solana_address'
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}'
            }
    
    @staticmethod
    def format_wallet_address(address: str, prefix_len: int = 8, suffix_len: int = 4) -> str:
        """
        Format wallet address for display.
        
        Args:
            address: Full wallet address
            prefix_len: Length of prefix to show
            suffix_len: Length of suffix to show
            
        Returns:
            Formatted address string
        """
        if not address or len(address) < prefix_len + suffix_len:
            return address
        
        return f"{address[:prefix_len]}...{address[-suffix_len:]}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """
        Format percentage value for display.
        
        Args:
            value: Percentage value
            decimals: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        try:
            if value >= 0:
                return f"+{value:.{decimals}f}%"
            else:
                return f"{value:.{decimals}f}%"
        except:
            return "0.0%"
    
    @staticmethod
    def format_sol_amount(amount: float, decimals: int = 4) -> str:
        """
        Format SOL amount for display.
        
        Args:
            amount: SOL amount
            decimals: Number of decimal places
            
        Returns:
            Formatted SOL string
        """
        try:
            if amount >= 1000:
                return f"{amount/1000:.1f}K SOL"
            elif amount >= 1:
                return f"{amount:.{decimals}f} SOL"
            else:
                return f"{amount:.{min(6, decimals)}f} SOL"
        except:
            return "0 SOL"
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Format duration in human-readable format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        try:
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                minutes = seconds / 60
                return f"{minutes:.1f}m"
            elif seconds < 86400:
                hours = seconds / 3600
                return f"{hours:.1f}h"
            else:
                days = seconds / 86400
                return f"{days:.1f}d"
        except:
            return "0s"
    
    @staticmethod
    def calculate_roi_percentage(initial_value: float, final_value: float) -> float:
        """
        Calculate ROI percentage.
        
        Args:
            initial_value: Initial investment value
            final_value: Final value
            
        Returns:
            ROI percentage
        """
        try:
            if initial_value <= 0:
                return 0.0
            return ((final_value / initial_value) - 1) * 100
        except:
            return 0.0
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safe division with default value.
        
        Args:
            numerator: Numerator
            denominator: Denominator
            default: Default value if division by zero
            
        Returns:
            Division result or default
        """
        try:
            if denominator == 0:
                return default
            return numerator / denominator
        except:
            return default
    
    @staticmethod
    def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime:
        """
        Convert timestamp to datetime object.
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            datetime object
        """
        try:
            return datetime.fromtimestamp(timestamp)
        except:
            return datetime.now()
    
    @staticmethod
    def datetime_to_timestamp(dt: datetime) -> int:
        """
        Convert datetime to timestamp.
        
        Args:
            dt: datetime object
            
        Returns:
            Unix timestamp
        """
        try:
            return int(dt.timestamp())
        except:
            return int(time.time())
    
    @staticmethod
    def get_time_ago_string(timestamp: Union[int, float]) -> str:
        """
        Get human-readable time ago string.
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            Time ago string
        """
        try:
            dt = ZeusUtils.timestamp_to_datetime(timestamp)
            now = datetime.now()
            diff = now - dt
            
            if diff.days > 0:
                return f"{diff.days}d ago"
            elif diff.seconds > 3600:
                hours = diff.seconds // 3600
                return f"{hours}h ago"
            elif diff.seconds > 60:
                minutes = diff.seconds // 60
                return f"{minutes}m ago"
            else:
                return f"{diff.seconds}s ago"
        except:
            return "unknown"
    
    @staticmethod
    def load_wallet_list(file_path: str) -> List[str]:
        """
        Load wallet addresses from file.
        
        Args:
            file_path: Path to wallet list file
            
        Returns:
            List of valid wallet addresses
        """
        wallets = []
        
        if not os.path.exists(file_path):
            logger.warning(f"Wallet file not found: {file_path}")
            return wallets
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Validate address
                    validation = ZeusUtils.validate_solana_address(line)
                    if validation['valid']:
                        wallets.append(validation['address'])
                    else:
                        logger.warning(f"Line {line_num}: Invalid address '{line}' - {validation['error']}")
            
            logger.info(f"Loaded {len(wallets)} valid wallet addresses from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading wallet list: {str(e)}")
        
        return wallets
    
    @staticmethod
    def save_wallet_list(wallets: List[str], file_path: str) -> bool:
        """
        Save wallet addresses to file.
        
        Args:
            wallets: List of wallet addresses
            file_path: Output file path
            
        Returns:
            bool: True if successful
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# Zeus Wallet List\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Total wallets: {len(wallets)}\n\n")
                
                for wallet in wallets:
                    f.write(f"{wallet}\n")
            
            logger.info(f"Saved {len(wallets)} wallets to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving wallet list: {str(e)}")
            return False
    
    @staticmethod
    def ensure_output_directory(file_path: str) -> str:
        """
        Ensure output directory exists and return full path.
        
        Args:
            file_path: File path
            
        Returns:
            Full file path with ensured directory
        """
        try:
            # If no directory specified, use outputs folder
            if not os.path.dirname(file_path):
                output_dir = os.path.join(os.getcwd(), "outputs")
                file_path = os.path.join(output_dir, file_path)
            
            # Ensure directory exists
            directory = os.path.dirname(file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error ensuring output directory: {str(e)}")
            return file_path
    
    @staticmethod
    def generate_filename_with_timestamp(base_name: str, extension: str = "csv") -> str:
        """
        Generate filename with timestamp.
        
        Args:
            base_name: Base filename
            extension: File extension
            
        Returns:
            Filename with timestamp
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.{extension}"
    
    @staticmethod
    def hash_wallet_address(address: str) -> str:
        """
        Generate hash of wallet address for privacy.
        
        Args:
            address: Wallet address
            
        Returns:
            SHA256 hash (first 16 characters)
        """
        try:
            hash_obj = hashlib.sha256(address.encode())
            return hash_obj.hexdigest()[:16]
        except:
            return "unknown_hash"
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename for filesystem compatibility.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limit length
        if len(sanitized) > 200:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:200-len(ext)] + ext
        
        return sanitized
    
    @staticmethod
    def calculate_statistics(values: List[float]) -> Dict[str, float]:
        """
        Calculate basic statistics for a list of values.
        
        Args:
            values: List of numeric values
            
        Returns:
            Dict with statistical measures
        """
        try:
            if not values:
                return {
                    'count': 0,
                    'mean': 0,
                    'median': 0,
                    'std': 0,
                    'min': 0,
                    'max': 0,
                    'sum': 0
                }
            
            import statistics
            
            sorted_values = sorted(values)
            
            return {
                'count': len(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'sum': sum(values),
                'q1': sorted_values[len(sorted_values)//4] if len(sorted_values) > 3 else sorted_values[0],
                'q3': sorted_values[3*len(sorted_values)//4] if len(sorted_values) > 3 else sorted_values[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return {'error': str(e)}

class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type:
            logger.error(f"{self.operation_name} failed after {duration:.2f}s")
        else:
            logger.debug(f"{self.operation_name} completed in {duration:.2f}s")
    
    @property
    def duration(self) -> float:
        """Get operation duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_second: float = 10.0):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second if calls_per_second > 0 else 0
        self.last_call_time = 0
    
    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limit."""
        if self.min_interval <= 0:
            return
        
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            time.sleep(wait_time)
        
        self.last_call_time = time.time()

class DataCache:
    """Simple in-memory cache with TTL."""
    
    def __init__(self, default_ttl: int = 300):
        self.cache = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key not in self.cache:
            return None
        
        data, expiry = self.cache[key]
        
        if time.time() > expiry:
            del self.cache[key]
            return None
        
        return data
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with TTL."""
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        self.cache[key] = (value, expiry)
    
    def clear(self) -> None:
        """Clear all cached values."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)

def setup_zeus_logging(level: str = "INFO", log_file: str = "zeus.log") -> None:
    """
    Setup Zeus logging configuration.
    
    Args:
        level: Logging level
        log_file: Log file path
    """
    import sys
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger("zeus")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Prevent duplicate logs
    root_logger.propagate = False

def create_sample_wallet_file(file_path: str = "wallets.txt") -> bool:
    """
    Create sample wallet file with example addresses.
    
    Args:
        file_path: Path to create sample file
        
    Returns:
        bool: True if successful
    """
    try:
        sample_wallets = [
            "# Zeus Sample Wallet List",
            "# Add your wallet addresses below (one per line)",
            "# Lines starting with # are comments and will be ignored",
            "",
            "# Example wallets (replace with real addresses):",
            "7xG8k9mPqR3nW2sJ5tY8vL4hE6dF1aZ9bN3cM7uV2iK1",
            "9aB2c3D4e5F6g7H8i9J1k2L3m4N5o6P7q8R9s1T2u3V4",
            "5eF7g8H9i1J2k3L4m5N6o7P8q9R1s2T3u4V5w6X7y8Z9",
            "",
            "# Add your wallets here:",
            ""
        ]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sample_wallets))
        
        logger.info(f"Created sample wallet file: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating sample wallet file: {str(e)}")
        return False