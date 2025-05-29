"""
Zeus Utilities - FIXED with Safe Validation and Type Checking
CRITICAL FIXES:
- Fixed all type comparison errors in validation functions
- Added proper type guards for all operations
- Safe handling of complex data types (dicts, lists)
- Defensive programming with try-catch blocks
- Preserved all existing utility functionality
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
    """Collection of utility functions for Zeus system with SAFE validation and Token PnL features."""
    
    @staticmethod
    def validate_solana_address(address: str) -> Dict[str, Any]:
        """
        Validate Solana wallet/token address with SAFE type checking.
        
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
        Format percentage value for display with updated precision.
        
        Args:
            value: Percentage value
            decimals: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        try:
            if isinstance(value, (int, float)):
                if value >= 0:
                    return f"+{value:.{decimals}f}%"
                else:
                    return f"{value:.{decimals}f}%"
            else:
                return "0.0%"
        except:
            return "0.0%"
    
    @staticmethod
    def format_sol_amount(amount: float, decimals: int = 1) -> str:
        """
        Format SOL amount for display with updated precision.
        
        Args:
            amount: SOL amount
            decimals: Number of decimal places (updated default to 1)
            
        Returns:
            Formatted SOL string
        """
        try:
            if isinstance(amount, (int, float)):
                if amount >= 1000:
                    return f"{amount/1000:.1f}K SOL"
                elif amount >= 1:
                    return f"{amount:.{decimals}f} SOL"
                else:
                    return f"{amount:.{min(6, decimals)}f} SOL"
            else:
                return "0 SOL"
        except:
            return "0 SOL"
    
    @staticmethod
    def format_duration(seconds: float, use_updated_thresholds: bool = True) -> str:
        """
        Format duration in human-readable format with updated thresholds.
        
        Args:
            seconds: Duration in seconds
            use_updated_thresholds: Use updated thresholds (5min/24hr)
            
        Returns:
            Formatted duration string
        """
        try:
            if not isinstance(seconds, (int, float)) or seconds < 0:
                return "0s"
                
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                minutes = seconds / 60
                # Mark very short holds with updated threshold
                if use_updated_thresholds and minutes < 5:
                    return f"{minutes:.1f}m ⚡"  # Very short indicator
                return f"{minutes:.1f}m"
            elif seconds < 86400:
                hours = seconds / 3600
                return f"{hours:.1f}h"
            else:
                days = seconds / 86400
                # Mark long holds with updated threshold
                if use_updated_thresholds and days >= 1:  # 24+ hours
                    return f"{days:.1f}d 🔒"  # Long hold indicator
                return f"{days:.1f}d"
        except:
            return "0s"
    
    @staticmethod
    def calculate_roi_percentage(initial_value: float, final_value: float) -> float:
        """
        Calculate ROI percentage with enhanced precision and SAFE type checking.
        
        Args:
            initial_value: Initial investment value
            final_value: Final value
            
        Returns:
            ROI percentage
        """
        try:
            if not isinstance(initial_value, (int, float)) or not isinstance(final_value, (int, float)):
                return 0.0
            if initial_value <= 0:
                return 0.0
            return round(((final_value / initial_value) - 1) * 100, 2)
        except:
            return 0.0
    
    @staticmethod
    def identify_trader_pattern(metrics: Dict[str, Any], use_updated_thresholds: bool = True) -> str:
        """
        Identify trader pattern based on metrics with updated thresholds and SAFE type checking.
        
        Args:
            metrics: Trading metrics
            use_updated_thresholds: Use updated thresholds (5min/24hr)
            
        Returns:
            Identified pattern
        """
        try:
            if not isinstance(metrics, dict):
                return 'analysis_error'
                
            avg_hold_time_hours = metrics.get('avg_hold_time_hours', 24)
            avg_roi = metrics.get('avg_roi', 0)
            moonshot_rate = metrics.get('moonshot_rate', 0)
            win_rate = metrics.get('win_rate', 50)
            total_trades = metrics.get('total_trades', 0)
            
            # SAFE type checking for all values
            if not isinstance(avg_hold_time_hours, (int, float)):
                avg_hold_time_hours = 24
            if not isinstance(avg_roi, (int, float)):
                avg_roi = 0
            if not isinstance(moonshot_rate, (int, float)):
                moonshot_rate = 0
            if not isinstance(win_rate, (int, float)):
                win_rate = 50
            if not isinstance(total_trades, (int, float)):
                total_trades = 0
            
            # Updated thresholds
            if use_updated_thresholds:
                very_short_threshold = 0.083  # 5 minutes
                long_hold_threshold = 24      # 24 hours
            else:
                very_short_threshold = 0.2    # 12 minutes (old)
                long_hold_threshold = 48      # 48 hours (old)
            
            # Pattern identification with updated thresholds
            if avg_hold_time_hours < very_short_threshold:
                return 'flipper'
            elif avg_hold_time_hours < 1:
                return 'sniper' if avg_roi > 30 else 'impulsive_trader'
            elif moonshot_rate > 10 and avg_roi > 100:
                return 'gem_hunter'
            elif avg_hold_time_hours > long_hold_threshold:
                return 'position_trader' if avg_roi > 50 else 'bag_holder'
            elif win_rate > 60 and avg_roi > 20:
                return 'consistent_trader'
            else:
                return 'mixed_strategy'
                
        except Exception as e:
            logger.error(f"Error identifying trader pattern: {str(e)}")
            return 'analysis_error'
    
    @staticmethod
    def calculate_smart_tp_sl(pattern: str, actual_performance: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculate smart TP/SL levels based on trader pattern and actual performance with SAFE validation.
        
        Args:
            pattern: Identified trader pattern
            actual_performance: Actual performance metrics from Token PnL analysis
            
        Returns:
            Dict with smart TP/SL recommendations
        """
        try:
            # Default pattern-based recommendations
            pattern_defaults = {
                'flipper': {
                    'tp1': 30, 'tp2': 60, 'tp3': 120, 'stop_loss': -15,
                    'reasoning': 'Flipper pattern - Quick profits with tight SL'
                },
                'gem_hunter': {
                    'tp1': 200, 'tp2': 500, 'tp3': 1000, 'stop_loss': -50,
                    'reasoning': 'Gem hunter pattern - High TPs with patient SL'
                },
                'consistent_trader': {
                    'tp1': 75, 'tp2': 150, 'tp3': 300, 'stop_loss': -25,
                    'reasoning': 'Consistent pattern - Balanced TP/SL'
                },
                'position_trader': {
                    'tp1': 100, 'tp2': 300, 'tp3': 600, 'stop_loss': -40,
                    'reasoning': 'Position trader - Patient approach'
                }
            }
            
            base_recommendation = pattern_defaults.get(pattern, pattern_defaults['consistent_trader'])
            
            # SAFE enhancement with actual performance data
            if actual_performance and isinstance(actual_performance, dict) and actual_performance.get('based_on_actual_exits'):
                try:
                    actual_tp1 = actual_performance.get('avg_tp1', base_recommendation['tp1'])
                    actual_tp2 = actual_performance.get('avg_tp2', base_recommendation['tp2'])
                    actual_sl = actual_performance.get('avg_stop_loss', base_recommendation['stop_loss'])
                    
                    # SAFE type checking and range validation
                    if isinstance(actual_tp1, (int, float)) and 10 <= actual_tp1 <= 2000:
                        safety_tp1 = int(actual_tp1 * 1.1)  # 10% buffer
                    else:
                        safety_tp1 = base_recommendation['tp1']
                    
                    if isinstance(actual_tp2, (int, float)) and 20 <= actual_tp2 <= 5000:
                        safety_tp2 = int(actual_tp2 * 1.1)
                    else:
                        safety_tp2 = base_recommendation['tp2']
                    
                    if isinstance(actual_tp2, (int, float)):
                        safety_tp3 = int(max(safety_tp2 * 2.0, base_recommendation['tp3']))
                    else:
                        safety_tp3 = base_recommendation['tp3']
                    
                    if isinstance(actual_sl, (int, float)) and -90 <= actual_sl <= -5:
                        safety_sl = int(actual_sl * 0.9)  # 10% tighter SL
                    else:
                        safety_sl = base_recommendation['stop_loss']
                    
                    return {
                        'tp1': safety_tp1,
                        'tp2': safety_tp2,
                        'tp3': safety_tp3,
                        'stop_loss': safety_sl,
                        'reasoning': f'{pattern} pattern with actual exit data + 10% safety buffer',
                        'based_on_actual_exits': True,
                        'original_performance': actual_performance
                    }
                except Exception as perf_error:
                    logger.debug(f"Error processing actual performance data: {str(perf_error)}")
            
            return {
                **base_recommendation,
                'based_on_actual_exits': False,
                'pattern_based': True
            }
            
        except Exception as e:
            logger.error(f"Error calculating smart TP/SL: {str(e)}")
            return {
                'tp1': 75, 'tp2': 200, 'tp3': 500, 'stop_loss': -35,
                'reasoning': f'Default due to error: {str(e)}',
                'based_on_actual_exits': False
            }
    
    @staticmethod
    def validate_cielo_field_data(data: Dict[str, Any], field_name: str, 
                                expected_type: type, expected_range: Tuple[float, float] = None) -> Dict[str, Any]:
        """
        SAFELY validate Cielo API field data with enhanced validation and proper type checking.
        CRITICAL FIX: No more type comparison errors!
        
        Args:
            data: Cielo API response data
            field_name: Field name to validate
            expected_type: Expected data type
            expected_range: Expected value range (min, max) - only applied to numeric types
            
        Returns:
            Validation result
        """
        try:
            if not isinstance(data, dict):
                return {
                    'valid': False,
                    'error': f'Data is not a dictionary, got {type(data).__name__}',
                    'field_name': field_name
                }
            
            if field_name not in data:
                return {
                    'valid': False,
                    'error': f'Field {field_name} not found',
                    'field_name': field_name
                }
            
            value = data[field_name]
            
            # SAFE type validation
            if not isinstance(value, expected_type):
                return {
                    'valid': False,
                    'error': f'Field {field_name} has type {type(value).__name__}, expected {expected_type.__name__}',
                    'field_name': field_name,
                    'actual_value': str(value)[:100]  # Truncate long values
                }
            
            # SAFE range validation - ONLY for numeric types
            if expected_range and isinstance(value, (int, float)) and isinstance(expected_type, type) and issubclass(expected_type, (int, float)):
                try:
                    min_val, max_val = expected_range
                    if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                        if not (min_val <= value <= max_val):
                            return {
                                'valid': False,
                                'error': f'Field {field_name} value {value} outside expected range [{min_val}, {max_val}]',
                                'field_name': field_name,
                                'actual_value': value,
                                'expected_range': expected_range
                            }
                except Exception as range_error:
                    logger.debug(f"Range validation error for {field_name}: {str(range_error)}")
                    # Continue without range validation if there's an error
            
            return {
                'valid': True,
                'field_name': field_name,
                'value': value,
                'type': type(value).__name__
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation error for {field_name}: {str(e)}',
                'field_name': field_name
            }
    
    @staticmethod
    def extract_direct_field_value(data: Dict[str, Any], field_names: List[str], 
                                 default_value: Any = None, 
                                 validation_func: callable = None) -> Dict[str, Any]:
        """
        SAFELY extract direct field value from Cielo API response with multiple field name options.
        
        Args:
            data: API response data
            field_names: List of possible field names (in priority order)
            default_value: Default value if no field found
            validation_func: Optional validation function
            
        Returns:
            Extraction result
        """
        try:
            if not isinstance(data, dict):
                return {
                    'success': False,
                    'field_names_tried': field_names,
                    'value': default_value,
                    'error': f'Data is not a dictionary, got {type(data).__name__}'
                }
            
            if not isinstance(field_names, list):
                return {
                    'success': False,
                    'field_names_tried': [],
                    'value': default_value,
                    'error': 'field_names must be a list'
                }
            
            for field_name in field_names:
                if isinstance(field_name, str) and field_name in data and data[field_name] is not None:
                    value = data[field_name]
                    
                    # SAFE validation if provided
                    if validation_func and callable(validation_func):
                        try:
                            if not validation_func(value):
                                continue
                        except Exception as val_error:
                            logger.debug(f"Validation function error for {field_name}: {str(val_error)}")
                            continue
                    
                    return {
                        'success': True,
                        'field_name': field_name,
                        'value': value,
                        'type': type(value).__name__
                    }
            
            # No valid field found
            return {
                'success': False,
                'field_names_tried': field_names,
                'value': default_value,
                'error': f'None of the fields {field_names} found or valid'
            }
            
        except Exception as e:
            return {
                'success': False,
                'field_names_tried': field_names,
                'value': default_value,
                'error': f'Field extraction error: {str(e)}'
            }
    
    @staticmethod
    def calculate_api_cost_estimate(wallet_count: int, features_enabled: Dict[str, bool]) -> Dict[str, Any]:
        """
        Calculate estimated API costs based on enabled features with SAFE validation.
        
        Args:
            wallet_count: Number of wallets to analyze
            features_enabled: Dict of enabled features
            
        Returns:
            Cost estimate breakdown
        """
        try:
            # SAFE input validation
            if not isinstance(wallet_count, int) or wallet_count < 0:
                wallet_count = 0
            
            if not isinstance(features_enabled, dict):
                features_enabled = {}
            
            # Default API costs
            api_costs = {
                'cielo_trading_stats': 30,
                'cielo_token_pnl': 5,
                'birdeye_token_price': 1,
                'helius_transactions': 0  # Free tier available
            }
            
            cost_breakdown = {}
            total_cost = 0
            
            # Trading Stats (REQUIRED)
            trading_stats_cost = wallet_count * api_costs['cielo_trading_stats']
            cost_breakdown['trading_stats'] = {
                'cost_per_wallet': api_costs['cielo_trading_stats'],
                'total_cost': trading_stats_cost,
                'required': True
            }
            total_cost += trading_stats_cost
            
            # Token PnL Analysis (NEW FEATURE)
            if features_enabled.get('enable_token_pnl_analysis', True):
                token_pnl_cost = wallet_count * api_costs['cielo_token_pnl']
                cost_breakdown['token_pnl'] = {
                    'cost_per_wallet': api_costs['cielo_token_pnl'],
                    'total_cost': token_pnl_cost,
                    'required': False,
                    'feature': 'Token PnL Analysis'
                }
                total_cost += token_pnl_cost
            
            # Birdeye (OPTIONAL)
            if features_enabled.get('enable_birdeye_analysis', False):
                birdeye_cost = wallet_count * api_costs['birdeye_token_price']
                cost_breakdown['birdeye'] = {
                    'cost_per_wallet': api_costs['birdeye_token_price'],
                    'total_cost': birdeye_cost,
                    'required': False,
                    'feature': 'Enhanced Token Analysis'
                }
                total_cost += birdeye_cost
            
            return {
                'total_cost': total_cost,
                'cost_per_wallet': round(total_cost / wallet_count, 2) if wallet_count > 0 else 0,
                'cost_breakdown': cost_breakdown,
                'wallet_count': wallet_count,
                'estimated_duration_minutes': wallet_count * 0.5,  # 30 seconds per wallet
                'features_enabled': features_enabled
            }
            
        except Exception as e:
            logger.error(f"Error calculating API cost estimate: {str(e)}")
            return {
                'total_cost': 0,
                'error': str(e),
                'wallet_count': wallet_count
            }
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safe division with default value and SAFE type checking.
        
        Args:
            numerator: Numerator
            denominator: Denominator
            default: Default value if division by zero
            
        Returns:
            Division result or default
        """
        try:
            if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
                return default
            if denominator == 0:
                return default
            return numerator / denominator
        except:
            return default
    
    @staticmethod
    def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime:
        """
        Convert timestamp to datetime object with SAFE type checking.
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            datetime object
        """
        try:
            if isinstance(timestamp, (int, float)) and timestamp > 0:
                return datetime.fromtimestamp(timestamp)
            else:
                return datetime.now()
        except:
            return datetime.now()
    
    @staticmethod
    def datetime_to_timestamp(dt: datetime) -> int:
        """
        Convert datetime to timestamp with SAFE type checking.
        
        Args:
            dt: datetime object
            
        Returns:
            Unix timestamp
        """
        try:
            if isinstance(dt, datetime):
                return int(dt.timestamp())
            else:
                return int(time.time())
        except:
            return int(time.time())
    
    @staticmethod
    def get_time_ago_string(timestamp: Union[int, float], use_updated_format: bool = True) -> str:
        """
        Get human-readable time ago string with updated precision and SAFE type checking.
        
        Args:
            timestamp: Unix timestamp
            use_updated_format: Use updated format with 1 decimal precision
            
        Returns:
            Time ago string
        """
        try:
            if not isinstance(timestamp, (int, float)) or timestamp <= 0:
                return "unknown"
                
            dt = ZeusUtils.timestamp_to_datetime(timestamp)
            now = datetime.now()
            diff = now - dt
            
            if diff.days > 0:
                if use_updated_format:
                    return f"{diff.days + diff.seconds/86400:.1f}d ago"
                return f"{diff.days}d ago"
            elif diff.seconds > 3600:
                hours = diff.seconds / 3600
                if use_updated_format:
                    return f"{hours:.1f}h ago"
                return f"{int(hours)}h ago"
            elif diff.seconds > 60:
                minutes = diff.seconds / 60
                if use_updated_format:
                    return f"{minutes:.1f}m ago"
                return f"{int(minutes)}m ago"
            else:
                return f"{diff.seconds}s ago"
        except:
            return "unknown"
    
    @staticmethod
    def load_wallet_list(file_path: str) -> List[str]:
        """
        Load wallet addresses from file with SAFE validation.
        
        Args:
            file_path: Path to wallet list file
            
        Returns:
            List of valid wallet addresses
        """
        wallets = []
        
        try:
            if not isinstance(file_path, str) or not file_path.strip():
                logger.warning("Invalid file path provided")
                return wallets
                
            if not os.path.exists(file_path):
                logger.warning(f"Wallet file not found: {file_path}")
                return wallets
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
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
                    except Exception as line_error:
                        logger.warning(f"Line {line_num}: Error processing line - {str(line_error)}")
            
            logger.info(f"Loaded {len(wallets)} valid wallet addresses from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading wallet list: {str(e)}")
        
        return wallets
    
    @staticmethod
    def save_wallet_list(wallets: List[str], file_path: str) -> bool:
        """
        Save wallet addresses to file with SAFE validation.
        
        Args:
            wallets: List of wallet addresses
            file_path: Output file path
            
        Returns:
            bool: True if successful
        """
        try:
            if not isinstance(wallets, list):
                logger.error("Wallets must be a list")
                return False
                
            if not isinstance(file_path, str) or not file_path.strip():
                logger.error("Invalid file path")
                return False
            
            # Ensure directory exists
            directory = os.path.dirname(file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# Zeus Wallet List - Token PnL Analysis Enabled\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Total wallets: {len(wallets)}\n")
                f.write(f"# Features: Direct field extraction, Smart TP/SL, SAFE validation\n\n")
                
                for wallet in wallets:
                    if isinstance(wallet, str) and wallet.strip():
                        f.write(f"{wallet.strip()}\n")
            
            logger.info(f"Saved {len(wallets)} wallets to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving wallet list: {str(e)}")
            return False
    
    @staticmethod
    def ensure_output_directory(file_path: str) -> str:
        """
        Ensure output directory exists and return full path with SAFE validation.
        
        Args:
            file_path: File path
            
        Returns:
            Full file path with ensured directory
        """
        try:
            if not isinstance(file_path, str):
                file_path = str(file_path) if file_path else "output.txt"
            
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
    def generate_filename_with_timestamp(base_name: str, extension: str = "csv", 
                                       include_features: bool = True) -> str:
        """
        Generate filename with timestamp and optional feature indicators.
        
        Args:
            base_name: Base filename
            extension: File extension
            include_features: Include feature indicators in filename
            
        Returns:
            Filename with timestamp and features
        """
        try:
            if not isinstance(base_name, str):
                base_name = "zeus_analysis"
            if not isinstance(extension, str):
                extension = "csv"
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if include_features:
                return f"{base_name}_token_pnl_smart_tp_sl_safe_{timestamp}.{extension}"
            else:
                return f"{base_name}_{timestamp}.{extension}"
        except Exception as e:
            logger.error(f"Error generating filename: {str(e)}")
            return f"zeus_analysis_{int(time.time())}.csv"
    
    @staticmethod
    def hash_wallet_address(address: str) -> str:
        """
        Generate hash of wallet address for privacy with SAFE validation.
        
        Args:
            address: Wallet address
            
        Returns:
            SHA256 hash (first 16 characters)
        """
        try:
            if isinstance(address, str) and address.strip():
                hash_obj = hashlib.sha256(address.encode())
                return hash_obj.hexdigest()[:16]
            else:
                return "unknown_hash"
        except:
            return "unknown_hash"
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename for filesystem compatibility with SAFE validation.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        try:
            if not isinstance(filename, str):
                filename = str(filename) if filename else "sanitized_file"
            
            # Remove invalid characters
            sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
            
            # Limit length
            if len(sanitized) > 200:
                name, ext = os.path.splitext(sanitized)
                sanitized = name[:200-len(ext)] + ext
            
            return sanitized
        except Exception as e:
            logger.error(f"Error sanitizing filename: {str(e)}")
            return "sanitized_file"
    
    @staticmethod
    def calculate_statistics(values: List[float]) -> Dict[str, float]:
        """
        Calculate basic statistics for a list of values with enhanced precision and SAFE validation.
        
        Args:
            values: List of numeric values
            
        Returns:
            Dict with statistical measures
        """
        try:
            if not isinstance(values, list) or not values:
                return {
                    'count': 0,
                    'mean': 0,
                    'median': 0,
                    'std': 0,
                    'min': 0,
                    'max': 0,
                    'sum': 0
                }
            
            # SAFE filtering of numeric values
            numeric_values = []
            for val in values:
                if isinstance(val, (int, float)) and not (isinstance(val, float) and (val != val)):  # Check for NaN
                    numeric_values.append(float(val))
            
            if not numeric_values:
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
            
            sorted_values = sorted(numeric_values)
            
            return {
                'count': len(numeric_values),
                'mean': round(statistics.mean(numeric_values), 2),
                'median': round(statistics.median(numeric_values), 2),
                'std': round(statistics.stdev(numeric_values), 2) if len(numeric_values) > 1 else 0,
                'min': round(min(numeric_values), 2),
                'max': round(max(numeric_values), 2),
                'sum': round(sum(numeric_values), 2),
                'q1': round(sorted_values[len(sorted_values)//4], 2) if len(sorted_values) > 3 else sorted_values[0],
                'q3': round(sorted_values[3*len(sorted_values)//4], 2) if len(sorted_values) > 3 else sorted_values[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return {'error': str(e)}
    
    @staticmethod
    def create_feature_status_summary(features_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create summary of enabled features with their status and SAFE validation.
        
        Args:
            features_config: Features configuration
            
        Returns:
            Feature status summary
        """
        try:
            if not isinstance(features_config, dict):
                features_config = {}
                
            feature_descriptions = {
                'enable_token_pnl_analysis': 'Token PnL Analysis (5 credits per wallet)',
                'enable_smart_tp_sl': 'Smart TP/SL Recommendations',
                'enable_direct_field_extraction': 'Direct Field Extraction (no scaling)',
                'enable_pattern_based_strategies': 'Pattern-based Strategies',
                'enable_field_validation': 'SAFE Field Validation',
                'enable_enhanced_transactions': 'Enhanced Transaction Analysis',
                'enable_price_analysis': 'Price Analysis',
                'enable_market_cap_analysis': 'Market Cap Analysis',
                'enable_bot_detection': 'Bot Detection'
            }
            
            enabled_features = []
            disabled_features = []
            
            for feature_key, description in feature_descriptions.items():
                if features_config.get(feature_key, False):
                    enabled_features.append(description)
                else:
                    disabled_features.append(description)
            
            return {
                'enabled_features': enabled_features,
                'disabled_features': disabled_features,
                'enabled_count': len(enabled_features),
                'total_count': len(feature_descriptions),
                'feature_percentage': round((len(enabled_features) / len(feature_descriptions)) * 100, 1)
            }
            
        except Exception as e:
            logger.error(f"Error creating feature status summary: {str(e)}")
            return {
                'enabled_features': [],
                'disabled_features': [],
                'error': str(e)
            }

class PerformanceTimer:
    """Context manager for timing operations with enhanced logging and SAFE validation."""
    
    def __init__(self, operation_name: str = "Operation", log_threshold_seconds: float = 1.0):
        self.operation_name = str(operation_name) if operation_name else "Operation"
        self.log_threshold_seconds = float(log_threshold_seconds) if isinstance(log_threshold_seconds, (int, float)) else 1.0
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
            # Only log if duration exceeds threshold
            if duration >= self.log_threshold_seconds:
                logger.info(f"{self.operation_name} completed in {duration:.2f}s")
            else:
                logger.debug(f"{self.operation_name} completed in {duration:.2f}s")
    
    @property
    def duration(self) -> float:
        """Get operation duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

class RateLimiter:
    """Enhanced rate limiter for API calls with cost tracking and SAFE validation."""
    
    def __init__(self, calls_per_second: float = 10.0, cost_per_call: float = 0.0):
        self.calls_per_second = max(0.1, float(calls_per_second)) if isinstance(calls_per_second, (int, float)) else 10.0
        self.cost_per_call = max(0.0, float(cost_per_call)) if isinstance(cost_per_call, (int, float)) else 0.0
        self.min_interval = 1.0 / self.calls_per_second if self.calls_per_second > 0 else 0
        self.last_call_time = 0
        self.total_calls = 0
        self.total_cost = 0.0
    
    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limit and track costs."""
        if self.min_interval <= 0:
            return
        
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            time.sleep(wait_time)
        
        self.last_call_time = time.time()
        self.total_calls += 1
        self.total_cost += self.cost_per_call
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            'total_calls': self.total_calls,
            'total_cost': self.total_cost,
            'calls_per_second': self.calls_per_second,
            'cost_per_call': self.cost_per_call
        }

class DataCache:
    """Enhanced in-memory cache with TTL and size limits and SAFE validation."""
    
    def __init__(self, default_ttl: int = 300, max_size: int = 1000):
        self.default_ttl = max(1, int(default_ttl)) if isinstance(default_ttl, (int, float)) else 300
        self.max_size = max(1, int(max_size)) if isinstance(max_size, (int, float)) else 1000
        self.cache = {}
        self.access_times = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired with SAFE validation."""
        try:
            if not isinstance(key, str) or key not in self.cache:
                return None
            
            data, expiry = self.cache[key]
            
            if time.time() > expiry:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                return None
            
            # Update access time for LRU
            self.access_times[key] = time.time()
            return data
        except Exception as e:
            logger.debug(f"Cache get error for key {key}: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with TTL and size management with SAFE validation."""
        try:
            if not isinstance(key, str):
                return
            
            # Enforce size limit
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            ttl = ttl or self.default_ttl
            if not isinstance(ttl, (int, float)) or ttl <= 0:
                ttl = self.default_ttl
            
            expiry = time.time() + ttl
            self.cache[key] = (value, expiry)
            self.access_times[key] = time.time()
        except Exception as e:
            logger.debug(f"Cache set error for key {key}: {str(e)}")
    
    def _evict_oldest(self) -> None:
        """Evict oldest accessed item."""
        try:
            if not self.access_times:
                return
            
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            if oldest_key in self.cache:
                del self.cache[oldest_key]
            if oldest_key in self.access_times:
                del self.access_times[oldest_key]
        except Exception as e:
            logger.debug(f"Cache eviction error: {str(e)}")
    
    def clear(self) -> None:
        """Clear all cached values."""
        self.cache.clear()
        self.access_times.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'default_ttl': self.default_ttl,
            'oldest_access': min(self.access_times.values()) if self.access_times else None,
            'newest_access': max(self.access_times.values()) if self.access_times else None
        }

def setup_zeus_logging(level: str = "INFO", log_file: str = "zeus.log", 
                      enable_enhanced_logging: bool = True) -> None:
    """
    Setup Zeus logging configuration with enhanced features and SAFE validation.
    
    Args:
        level: Logging level
        log_file: Log file path
        enable_enhanced_logging: Enable enhanced logging features
    """
    import sys
    
    try:
        # SAFE parameter validation
        if not isinstance(level, str):
            level = "INFO"
        if not isinstance(log_file, str):
            log_file = "zeus.log"
        
        # Create formatter
        if enable_enhanced_logging:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Setup file handler with rotation
        if enable_enhanced_logging:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=3, encoding='utf-8'
            )
        else:
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
        
        if enable_enhanced_logging:
            logger.info("Enhanced logging enabled with rotation, detailed formatting, and SAFE validation")
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")

def create_sample_wallet_file(file_path: str = "wallets.txt") -> bool:
    """
    Create sample wallet file with example addresses and new feature info with SAFE validation.
    
    Args:
        file_path: Path to create sample file
        
    Returns:
        bool: True if successful
    """
    try:
        if not isinstance(file_path, str):
            file_path = "wallets.txt"
            
        sample_wallets = [
            "# Zeus Sample Wallet List - Token PnL Analysis Enabled with SAFE Validation",
            "# Add your wallet addresses below (one per line)",
            "# Lines starting with # are comments and will be ignored",
            "",
            "# NEW FEATURES:",
            "# - Token PnL Analysis: Real trade pattern analysis (5 credits per wallet)",
            "# - Smart TP/SL: Pattern-based recommendations (flippers vs gem hunters)",
            "# - Direct Field Extraction: No scaling/conversion from Cielo",
            "# - SAFE Validation: Fixed type comparison errors",
            "# - Updated Thresholds: 5min (very short) | 24hr (long holds)",
            "",
            "# Example wallets (replace with real addresses):",
            "7xG8k9mPqR3nW2sJ5tY8vL4hE6dF1aZ9bN3cM7uV2iK1",
            "9aB2c3D4e5F6g7H8i9J1k2L3m4N5o6P7q8R9s1T2u3V4",
            "5eF7g8H9i1J2k3L4m5N6o7P8q9R1s2T3u4V5w6X7y8Z9",
            "",
            "# Add your wallets here:",
            "# Estimated cost: 35 credits per wallet (30 Trading Stats + 5 Token PnL)",
            "# SAFE validation ensures no type comparison errors",
            ""
        ]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sample_wallets))
        
        logger.info(f"Created sample wallet file with NEW FEATURES and SAFE validation info: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating sample wallet file: {str(e)}")
        return False