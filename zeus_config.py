"""
Zeus Configuration Management - UPDATED for Token PnL & Direct Field Features
Handles configuration loading, validation, and defaults

MAJOR UPDATES:
- Token PnL analysis configuration options
- Direct field extraction settings
- Updated pattern recognition thresholds (5min/24hr)
- Enhanced API validation with credit cost awareness
- Smart TP/SL strategy configuration options
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("zeus.config")

class ZeusConfig:
    """Zeus configuration manager with Token PnL and direct field extraction features."""
    
    DEFAULT_CONFIG = {
        "api_keys": {
            "birdeye_api_key": "",
            "cielo_api_key": "",
            "helius_api_key": "",  # REQUIRED
            "solana_rpc_url": "https://api.mainnet-beta.solana.com"
        },
        "analysis": {
            "days_to_analyze": 30,
            "min_unique_tokens": 6,
            "initial_token_sample": 5,
            "max_token_sample": 10,
            "composite_score_threshold": 65.0,
            "exit_quality_threshold": 70.0,
            "enable_smart_sampling": True,
            "timeout_seconds": 300,
            "require_real_timestamps": True,
            # UPDATED THRESHOLDS
            "very_short_threshold_minutes": 5,   # NEW: 5 minutes (was 12)
            "long_hold_threshold_hours": 24,     # NEW: 24 hours (was 48)
            # TOKEN PNL ANALYSIS SETTINGS
            "enable_token_pnl_analysis": True,   # NEW: Enable Token PnL endpoint
            "token_pnl_max_limit": 10,          # NEW: Max tokens to analyze per wallet
            "token_pnl_fallback_enabled": True, # NEW: Fallback if Token PnL fails
            # DIRECT FIELD EXTRACTION SETTINGS
            "enable_direct_field_extraction": True,  # NEW: Enable direct Cielo field extraction
            "disable_scaling_conversions": True,     # NEW: Disable all scaling/conversion logic
            "field_validation_enabled": True        # NEW: Enable field validation
        },
        "scoring": {
            "component_weights": {
                "risk_adjusted_performance": 0.30,
                "distribution_quality": 0.25,
                "trading_discipline": 0.20,
                "market_impact_awareness": 0.15,
                "consistency_reliability": 0.10
            },
            "volume_qualifier": {
                "baseline_tokens": 6,
                "emerging_tokens": 4,
                "new_tokens": 2,
                "baseline_multiplier": 1.0,
                "emerging_multiplier": 0.8,
                "new_multiplier": 0.6
            },
            "risk_adjusted": {
                "roi_thresholds": [25, 50, 100],
                "roi_scores": [0.4, 0.6, 0.8, 1.0],
                "std_thresholds": [50, 100, 200],
                "std_scores": [1.0, 0.9, 0.8, 0.7]
            },
            "distribution": {
                "moonshot_threshold": 400,
                "big_win_min": 100,
                "big_win_max": 400,
                "heavy_loss_threshold": -50,
                "moonshot_rate_thresholds": [5, 10, 15],
                "big_win_rate_thresholds": [10, 20, 30]
            },
            "discipline": {
                "quick_cut_hours": 4,
                "slow_cut_days": 7,
                "same_block_seconds": 300,  # UPDATED: 5 minutes (was 60 seconds)
                "flipper_rate_threshold": 20
            }
        },
        # SMART TP/SL CONFIGURATION
        "tp_sl_strategy": {
            "enable_pattern_based_tp_sl": True,
            "patterns": {
                "flipper": {
                    "tp1_range": [20, 50],
                    "tp2_range": [40, 100],
                    "tp3_range": [80, 200],
                    "stop_loss_range": [-25, -10]
                },
                "gem_hunter": {
                    "tp1_range": [100, 300],
                    "tp2_range": [300, 700],
                    "tp3_range": [500, 1200],
                    "stop_loss_range": [-60, -30]
                },
                "consistent_trader": {
                    "tp1_range": [50, 100],
                    "tp2_range": [100, 250],
                    "tp3_range": [200, 500],
                    "stop_loss_range": [-40, -20]
                },
                "position_trader": {
                    "tp1_range": [75, 150],
                    "tp2_range": [200, 400],
                    "tp3_range": [300, 800],
                    "stop_loss_range": [-50, -25]
                }
            },
            "safety_buffers": {
                "tp_buffer_multiplier": 1.1,    # 10% safety buffer for TPs
                "sl_buffer_multiplier": 0.9     # 10% tighter SL for safety
            }
        },
        "output": {
            "default_csv": "zeus_analysis.csv",
            "default_summary": "zeus_summary.txt",
            "default_bot_config": "zeus_bot_config.json",
            "include_failed_analyses": True,
            "sort_by_score": True,
            "export_formats": ["csv", "summary"],
            "timestamp_format": "%Y%m%d_%H%M%S",
            # UPDATED CSV SETTINGS
            "csv_precision": {
                "days_since_last_trade": 1,     # 1 decimal place
                "avg_sol_buy_per_token": 1,     # 1 decimal place
                "composite_score": 1,           # 1 decimal place
                "roi_values": 2,                # 2 decimal places
                "usd_values": 1                 # 1 decimal place
            },
            "exclude_columns": ["total_buys_30_days", "total_sells_30_days"],  # REMOVED columns
            "include_columns": ["unique_tokens_30d"]  # NEW column
        },
        "logging": {
            "level": "INFO",
            "file": "zeus.log",
            "max_file_size_mb": 10,
            "backup_count": 3,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            # ENHANCED LOGGING
            "log_api_responses": False,         # Log full API responses (debug mode)
            "log_field_discovery": True,       # Log Cielo field discovery
            "log_pattern_analysis": True       # Log trade pattern analysis
        },
        "performance": {
            "max_concurrent_analyses": 3,
            "api_timeout_seconds": 30,
            "retry_attempts": 3,
            "retry_delay_seconds": 1,
            "rate_limit_delay_ms": 100,
            # API COST MANAGEMENT
            "cost_tracking_enabled": True,
            "daily_cost_limit": 10000,         # Daily credit limit
            "warn_cost_threshold": 0.8,        # Warn at 80% of limit
            "api_costs": {
                "cielo_trading_stats": 30,
                "cielo_token_pnl": 5,
                "cielo_total_stats": 20,
                "birdeye_token_price": 1
            }
        },
        "features": {
            "enable_enhanced_transactions": True,
            "enable_price_analysis": True,
            "enable_market_cap_analysis": True,
            "enable_bot_detection": True,
            "enable_pattern_recognition": True,
            # NEW FEATURES
            "enable_token_pnl_analysis": True,      # NEW: Token PnL analysis
            "enable_smart_tp_sl": True,             # NEW: Smart TP/SL recommendations
            "enable_direct_field_extraction": True, # NEW: Direct field extraction
            "enable_field_validation": True,        # NEW: Field validation
            "enable_pattern_based_strategies": True # NEW: Pattern-based strategies
        }
    }
    
    # REQUIRED API KEYS - System cannot function without these
    REQUIRED_API_KEYS = [
        "cielo_api_key",    # Required for wallet trading stats + Token PnL
        "helius_api_key"    # Required for accurate timestamps
    ]
    
    # RECOMMENDED API KEYS - System works but with limitations
    RECOMMENDED_API_KEYS = [
        "birdeye_api_key"   # Recommended for enhanced token analysis
    ]
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize Zeus configuration with Token PnL and direct field extraction features.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.config_file = config_file or self._get_default_config_path()
        self.config = self._load_config()
        self._validate_config()
        self._setup_new_features()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path - prioritize local config."""
        # Try multiple locations (LOCAL FIRST)
        possible_paths = [
            os.path.join(os.getcwd(), "config", "zeus_config.json"),  # Project config FIRST
            os.path.join(os.getcwd(), "zeus_config.json"),           # Project root
            os.path.expanduser("~/.zeus_config.json")                # User home LAST
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Return local config as default (create if needed)
        return possible_paths[0]
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file with environment variable overrides."""
        config = self.DEFAULT_CONFIG.copy()
        
        # Load from file if exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    config = self._deep_merge(config, file_config)
                logger.info(f"Loaded configuration from: {self.config_file}")
            except Exception as e:
                logger.warning(f"Error loading config file {self.config_file}: {str(e)}")
                logger.info("Using default configuration")
        else:
            logger.info("Configuration file not found, using defaults with NEW features")
        
        # Override with environment variables
        config = self._apply_env_overrides(config)
        
        return config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        env_mappings = {
            'ZEUS_BIRDEYE_API_KEY': ['api_keys', 'birdeye_api_key'],
            'ZEUS_CIELO_API_KEY': ['api_keys', 'cielo_api_key'],
            'ZEUS_HELIUS_API_KEY': ['api_keys', 'helius_api_key'],
            'ZEUS_SOLANA_RPC_URL': ['api_keys', 'solana_rpc_url'],
            'ZEUS_ANALYSIS_DAYS': ['analysis', 'days_to_analyze'],
            'ZEUS_MIN_TOKENS': ['analysis', 'min_unique_tokens'],
            'ZEUS_SCORE_THRESHOLD': ['analysis', 'composite_score_threshold'],
            'ZEUS_LOG_LEVEL': ['logging', 'level'],
            # NEW ENVIRONMENT VARIABLES
            'ZEUS_ENABLE_TOKEN_PNL': ['features', 'enable_token_pnl_analysis'],
            'ZEUS_ENABLE_SMART_TP_SL': ['features', 'enable_smart_tp_sl'],
            'ZEUS_VERY_SHORT_MINUTES': ['analysis', 'very_short_threshold_minutes'],
            'ZEUS_LONG_HOLD_HOURS': ['analysis', 'long_hold_threshold_hours']
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                # Navigate to the nested config location
                current = config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Set the final value with type conversion
                final_key = config_path[-1]
                try:
                    # Try to convert to appropriate type
                    if final_key in ['days_to_analyze', 'min_unique_tokens', 'initial_token_sample', 
                                   'max_token_sample', 'very_short_threshold_minutes', 'long_hold_threshold_hours']:
                        current[final_key] = int(env_value)
                    elif final_key in ['composite_score_threshold', 'exit_quality_threshold']:
                        current[final_key] = float(env_value)
                    elif final_key in ['enable_token_pnl_analysis', 'enable_smart_tp_sl']:
                        current[final_key] = env_value.lower() in ['true', '1', 'yes', 'on']
                    else:
                        current[final_key] = env_value
                    
                    logger.info(f"Applied environment override: {env_var} = {env_value}")
                except ValueError:
                    logger.warning(f"Invalid value for {env_var}: {env_value}")
        
        return config
    
    def _validate_config(self) -> None:
        """Validate configuration values with REQUIRED API key checks."""
        try:
            # Validate REQUIRED API keys
            api_keys = self.config.get('api_keys', {})
            missing_required = []
            
            for required_key in self.REQUIRED_API_KEYS:
                if not api_keys.get(required_key, '').strip():
                    missing_required.append(required_key)
            
            if missing_required:
                error_msg = f"CRITICAL: Missing REQUIRED API keys: {', '.join(missing_required)}"
                logger.error(f"❌ {error_msg}")
                logger.error("Zeus cannot function without these API keys!")
                logger.error("- cielo_api_key: Required for Trading Stats (30 credits) + Token PnL (5 credits)")
                logger.error("- helius_api_key: Required for accurate transaction timestamps")
                raise ValueError(error_msg)
            
            # Check RECOMMENDED API keys
            missing_recommended = []
            for recommended_key in self.RECOMMENDED_API_KEYS:
                if not api_keys.get(recommended_key, '').strip():
                    missing_recommended.append(recommended_key)
            
            if missing_recommended:
                logger.warning(f"⚠️ Missing RECOMMENDED API keys: {', '.join(missing_recommended)}")
                logger.warning("System will work but with limited functionality")
            
            # Validate analysis settings
            analysis = self.config.get('analysis', {})
            
            if not (1 <= analysis.get('days_to_analyze', 30) <= 365):
                logger.warning("days_to_analyze should be between 1 and 365")
            
            if not (1 <= analysis.get('min_unique_tokens', 6) <= 100):
                logger.warning("min_unique_tokens should be between 1 and 100")
            
            # Validate NEW thresholds
            very_short_minutes = analysis.get('very_short_threshold_minutes', 5)
            if not (0.1 <= very_short_minutes <= 60):
                logger.warning("very_short_threshold_minutes should be between 0.1 and 60")
            
            long_hold_hours = analysis.get('long_hold_threshold_hours', 24)
            if not (1 <= long_hold_hours <= 168):  # Max 1 week
                logger.warning("long_hold_threshold_hours should be between 1 and 168")
            
            # Validate Token PnL settings
            token_pnl_limit = analysis.get('token_pnl_max_limit', 10)
            if not (1 <= token_pnl_limit <= 20):
                logger.warning("token_pnl_max_limit should be between 1 and 20")
            
            # Validate scoring thresholds
            score_threshold = analysis.get('composite_score_threshold', 65.0)
            if not (0 <= score_threshold <= 100):
                logger.warning("composite_score_threshold should be between 0 and 100")
            
            # Validate component weights
            scoring = self.config.get('scoring', {})
            weights = scoring.get('component_weights', {})
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Component weights sum to {total_weight}, should be 1.0")
            
            logger.info("✅ Configuration validation completed - All REQUIRED APIs configured")
            logger.info("🎯 NEW FEATURES: Token PnL analysis, Smart TP/SL, Direct field extraction")
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Configuration validation failed: {str(e)}")
            raise
    
    def _setup_new_features(self) -> None:
        """Setup and validate new feature configurations."""
        try:
            features = self.config.get('features', {})
            
            # Token PnL Analysis setup
            if features.get('enable_token_pnl_analysis', True):
                logger.info("✅ Token PnL Analysis: ENABLED (5 credits per wallet)")
                
                # Validate Token PnL settings
                analysis = self.config.get('analysis', {})
                max_limit = analysis.get('token_pnl_max_limit', 10)
                logger.info(f"   Max tokens per analysis: {max_limit}")
                
                fallback_enabled = analysis.get('token_pnl_fallback_enabled', True)
                logger.info(f"   Fallback enabled: {fallback_enabled}")
            else:
                logger.warning("⚠️ Token PnL Analysis: DISABLED")
            
            # Smart TP/SL setup
            if features.get('enable_smart_tp_sl', True):
                logger.info("✅ Smart TP/SL Recommendations: ENABLED")
                
                tp_sl_config = self.config.get('tp_sl_strategy', {})
                pattern_based = tp_sl_config.get('enable_pattern_based_tp_sl', True)
                logger.info(f"   Pattern-based strategies: {pattern_based}")
                
                patterns = tp_sl_config.get('patterns', {})
                logger.info(f"   Configured patterns: {list(patterns.keys())}")
            else:
                logger.warning("⚠️ Smart TP/SL Recommendations: DISABLED")
            
            # Direct Field Extraction setup
            if features.get('enable_direct_field_extraction', True):
                logger.info("✅ Direct Field Extraction: ENABLED")
                
                analysis = self.config.get('analysis', {})
                disable_scaling = analysis.get('disable_scaling_conversions', True)
                logger.info(f"   Scaling/conversions disabled: {disable_scaling}")
                
                field_validation = analysis.get('field_validation_enabled', True)
                logger.info(f"   Field validation enabled: {field_validation}")
            else:
                logger.warning("⚠️ Direct Field Extraction: DISABLED")
            
            # Pattern Recognition setup
            if features.get('enable_pattern_based_strategies', True):
                logger.info("✅ Pattern-based Strategies: ENABLED")
                
                analysis = self.config.get('analysis', {})
                very_short = analysis.get('very_short_threshold_minutes', 5)
                long_hold = analysis.get('long_hold_threshold_hours', 24)
                logger.info(f"   Updated thresholds: <{very_short}min | >{long_hold}hr")
            else:
                logger.warning("⚠️ Pattern-based Strategies: DISABLED")
            
        except Exception as e:
            logger.warning(f"Warning during new feature setup: {str(e)}")
    
    def validate_system_readiness(self) -> Dict[str, Any]:
        """
        Validate that the system is ready to run with all required APIs and new features.
        
        Returns:
            Dict with readiness status and details
        """
        api_keys = self.get_api_config()
        
        # Check required APIs
        required_status = {}
        for key in self.REQUIRED_API_KEYS:
            required_status[key] = bool(api_keys.get(key, '').strip())
        
        # Check recommended APIs  
        recommended_status = {}
        for key in self.RECOMMENDED_API_KEYS:
            recommended_status[key] = bool(api_keys.get(key, '').strip())
        
        # Check new features
        features = self.config.get('features', {})
        new_features_status = {
            'token_pnl_analysis': features.get('enable_token_pnl_analysis', True),
            'smart_tp_sl': features.get('enable_smart_tp_sl', True),
            'direct_field_extraction': features.get('enable_direct_field_extraction', True),
            'pattern_based_strategies': features.get('enable_pattern_based_strategies', True)
        }
        
        # Determine overall readiness
        all_required_configured = all(required_status.values())
        
        return {
            'system_ready': all_required_configured,
            'required_apis': required_status,
            'recommended_apis': recommended_status,
            'new_features': new_features_status,
            'missing_required': [k for k, v in required_status.items() if not v],
            'missing_recommended': [k for k, v in recommended_status.items() if not v],
            'readiness_summary': {
                'trading_stats_ready': required_status.get('cielo_api_key', False),
                'token_pnl_ready': required_status.get('cielo_api_key', False) and new_features_status.get('token_pnl_analysis', False),
                'timestamp_accuracy': required_status.get('helius_api_key', False),
                'smart_tp_sl_ready': all_required_configured and new_features_status.get('smart_tp_sl', False),
                'direct_extraction_ready': new_features_status.get('direct_field_extraction', False),
                'enhanced_features': recommended_status.get('birdeye_api_key', False)
            }
        }
    
    def get_cost_estimate(self, wallet_count: int) -> Dict[str, Any]:
        """
        Calculate estimated API costs for analysis.
        
        Args:
            wallet_count: Number of wallets to analyze
            
        Returns:
            Dict with cost breakdown
        """
        try:
            costs = self.config.get('performance', {}).get('api_costs', {})
            features = self.config.get('features', {})
            
            cost_breakdown = {}
            total_cost = 0
            
            # Trading Stats cost (REQUIRED)
            trading_stats_cost = wallet_count * costs.get('cielo_trading_stats', 30)
            cost_breakdown['trading_stats'] = {
                'cost_per_wallet': costs.get('cielo_trading_stats', 30),
                'total_cost': trading_stats_cost,
                'description': 'Cielo Trading Stats (REQUIRED)'
            }
            total_cost += trading_stats_cost
            
            # Token PnL cost (NEW FEATURE)
            if features.get('enable_token_pnl_analysis', True):
                token_pnl_cost = wallet_count * costs.get('cielo_token_pnl', 5)
                cost_breakdown['token_pnl'] = {
                    'cost_per_wallet': costs.get('cielo_token_pnl', 5),
                    'total_cost': token_pnl_cost,
                    'description': 'Token PnL Analysis (NEW)'
                }
                total_cost += token_pnl_cost
            
            # Optional Birdeye costs
            if self.get_api_config().get('birdeye_api_key'):
                birdeye_cost = wallet_count * costs.get('birdeye_token_price', 1)
                cost_breakdown['birdeye'] = {
                    'cost_per_wallet': costs.get('birdeye_token_price', 1),
                    'total_cost': birdeye_cost,
                    'description': 'Birdeye Enhanced Analysis (OPTIONAL)'
                }
                total_cost += birdeye_cost
            
            # Cost limits
            performance = self.config.get('performance', {})
            daily_limit = performance.get('daily_cost_limit', 10000)
            warn_threshold = performance.get('warn_cost_threshold', 0.8)
            
            return {
                'cost_breakdown': cost_breakdown,
                'total_cost': total_cost,
                'cost_per_wallet': total_cost / wallet_count if wallet_count > 0 else 0,
                'daily_limit': daily_limit,
                'warn_threshold': warn_threshold,
                'exceeds_daily_limit': total_cost > daily_limit,
                'exceeds_warn_threshold': total_cost > (daily_limit * warn_threshold),
                'wallet_count': wallet_count
            }
            
        except Exception as e:
            logger.error(f"Error calculating cost estimate: {str(e)}")
            return {
                'error': str(e),
                'total_cost': 0,
                'wallet_count': wallet_count
            }
    
    def save_config(self, config_file: Optional[str] = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_file: Optional config file path
            
        Returns:
            bool: True if successful
        """
        try:
            save_path = config_file or self.config_file
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save configuration
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated key path (e.g., 'analysis.days_to_analyze')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key_path.split('.')
            current = self.config
            
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            
            return current
            
        except Exception:
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated key path
            value: Value to set
        """
        try:
            keys = key_path.split('.')
            current = self.config
            
            # Navigate to parent
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set final value
            current[keys[-1]] = value
            
        except Exception as e:
            logger.error(f"Error setting config value {key_path}: {str(e)}")
    
    def get_api_config(self) -> Dict[str, str]:
        """Get API configuration from nested api_keys section."""
        return self.config.get('api_keys', {})
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration."""
        return self.config.get('analysis', {})
    
    def get_scoring_config(self) -> Dict[str, Any]:
        """Get scoring configuration."""
        return self.config.get('scoring', {})
    
    def get_tp_sl_config(self) -> Dict[str, Any]:
        """Get TP/SL strategy configuration (NEW)."""
        return self.config.get('tp_sl_strategy', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.config.get('output', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self.config.get('performance', {})
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if a feature is enabled.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            bool: True if feature is enabled
        """
        features = self.config.get('features', {})
        return features.get(feature_name, False)
    
    def get_pattern_thresholds(self) -> Dict[str, float]:
        """Get updated pattern recognition thresholds (NEW)."""
        analysis = self.get_analysis_config()
        return {
            'very_short_threshold_minutes': analysis.get('very_short_threshold_minutes', 5),
            'very_short_threshold_hours': analysis.get('very_short_threshold_minutes', 5) / 60.0,
            'long_hold_threshold_hours': analysis.get('long_hold_threshold_hours', 24)
        }
    
    def update_api_key(self, api_name: str, api_key: str) -> None:
        """
        Update API key in nested api_keys configuration.
        
        Args:
            api_name: Name of the API (birdeye, cielo, helius)
            api_key: API key value
        """
        # Ensure api_keys section exists
        if 'api_keys' not in self.config:
            self.config['api_keys'] = {}
        
        # Update the API key in the nested structure
        self.config['api_keys'][f'{api_name}_api_key'] = api_key
        logger.info(f"Updated {api_name} API key in nested config")
        
        # Re-validate after updating
        try:
            self._validate_config()
        except Exception as e:
            logger.warning(f"Configuration validation warning after update: {str(e)}")
    
    def get_binary_decision_thresholds(self) -> Dict[str, float]:
        """Get binary decision thresholds."""
        analysis = self.get_analysis_config()
        return {
            'composite_score_threshold': analysis.get('composite_score_threshold', 65.0),
            'exit_quality_threshold': analysis.get('exit_quality_threshold', 70.0)
        }
    
    def get_volume_qualifier_config(self) -> Dict[str, Any]:
        """Get volume qualifier configuration."""
        scoring = self.get_scoring_config()
        return scoring.get('volume_qualifier', {})
    
    def create_default_config_file(self, config_path: str) -> bool:
        """
        Create a default configuration file.
        
        Args:
            config_path: Path to create config file
            
        Returns:
            bool: True if successful
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Write default configuration
            with open(config_path, 'w') as f:
                json.dump(self.DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created default configuration file: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating default config file: {str(e)}")
            return False
    
    def export_config_template(self, output_path: str) -> bool:
        """
        Export configuration template with comments for new features.
        
        Args:
            output_path: Path to save template
            
        Returns:
            bool: True if successful
        """
        try:
            template = {
                "_comment": "Zeus Wallet Analysis System Configuration",
                "_version": "2.2",
                "_description": {
                    "api_keys": "API keys for external services (Cielo & Helius REQUIRED, Birdeye recommended)",
                    "analysis": "Analysis parameters and thresholds with NEW Token PnL settings",
                    "scoring": "Scoring system weights and thresholds", 
                    "tp_sl_strategy": "NEW: Smart TP/SL strategy configuration",
                    "output": "Output format and file settings with NEW CSV options",
                    "logging": "Logging configuration",
                    "performance": "Performance and rate limiting settings with cost tracking",
                    "features": "Feature toggles including NEW features"
                },
                "_required_apis": {
                    "cielo_api_key": "REQUIRED - Trading Stats (30 credits) + Token PnL (5 credits)",
                    "helius_api_key": "REQUIRED - Accurate transaction timestamps"
                },
                "_recommended_apis": {
                    "birdeye_api_key": "RECOMMENDED - Enhanced token analysis"
                },
                "_new_features": {
                    "token_pnl_analysis": "NEW - Real trade pattern analysis (5 credits per wallet)",
                    "smart_tp_sl": "NEW - Pattern-based TP/SL recommendations",
                    "direct_field_extraction": "NEW - Direct Cielo field extraction (no scaling)",
                    "updated_thresholds": "NEW - 5min (very short) | 24hr (long holds)"
                }
            }
            
            # Merge with default config
            template.update(self.DEFAULT_CONFIG)
            
            with open(output_path, 'w') as f:
                json.dump(template, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration template exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting config template: {str(e)}")
            return False
    
    def __str__(self) -> str:
        """String representation of configuration."""
        api_keys = self.get_api_config()
        configured_apis = [name.replace('_api_key', '') for name, key in api_keys.items() if key]
        
        readiness = self.validate_system_readiness()
        status = "READY" if readiness['system_ready'] else "NOT READY"
        
        # Show new features status
        new_features = readiness.get('new_features', {})
        enabled_features = [name for name, enabled in new_features.items() if enabled]
        
        return f"ZeusConfig({status}, APIs: {', '.join(configured_apis)}, Features: {', '.join(enabled_features)}, File: {self.config_file})"

def load_zeus_config(config_file: Optional[str] = None) -> ZeusConfig:
    """
    Load Zeus configuration with Token PnL and direct field extraction features.
    
    Args:
        config_file: Optional configuration file path
        
    Returns:
        ZeusConfig instance
        
    Raises:
        ValueError: If required API keys are missing
    """
    return ZeusConfig(config_file)