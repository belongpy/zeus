"""
Zeus Configuration Management - Enhanced for Corrected Exit Analysis
Handles configuration loading, validation, and defaults

MAJOR ENHANCEMENTS:
- Configuration for corrected exit analysis features
- Realistic TP/SL pattern-based defaults configuration
- Enhanced validation to prevent inflated TP/SL recommendations
- Pattern threshold configuration (5min/24hr)
- Exit behavior inference settings
- TP/SL validation and correction settings
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("zeus.config")

class ZeusConfig:
    """Zeus configuration manager with corrected exit analysis and realistic TP/SL features."""
    
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
            # CORRECTED THRESHOLDS
            "very_short_threshold_minutes": 5,   # 5 minutes (corrected)
            "long_hold_threshold_hours": 24,     # 24 hours (corrected)
            # TOKEN PNL ANALYSIS SETTINGS
            "enable_token_pnl_analysis": True,   # Enable Token PnL endpoint
            "token_pnl_max_limit": 10,          # Max tokens to analyze per wallet
            "token_pnl_fallback_enabled": True, # Fallback if Token PnL fails
            # CORRECTED EXIT ANALYSIS SETTINGS
            "enable_corrected_exit_analysis": True,     # NEW: Enable corrected exit analysis
            "exit_behavior_inference_enabled": True,    # NEW: Infer actual vs final ROI
            "validate_tp_sl_for_patterns": True,        # NEW: Validate TP/SL makes sense for pattern
            "auto_correct_inflated_tp_sl": True,        # NEW: Auto-correct inflated recommendations
            "use_realistic_pattern_defaults": True,     # NEW: Use realistic TP/SL defaults
            # DIRECT FIELD EXTRACTION SETTINGS
            "enable_direct_field_extraction": True,  # Enable direct Cielo field extraction
            "disable_scaling_conversions": True,     # Disable all scaling/conversion logic
            "field_validation_enabled": True        # Enable field validation
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
                "same_block_seconds": 300,  # 5 minutes (corrected)
                "flipper_rate_threshold": 20
            },
            # CORRECTED EXIT ANALYSIS SCORING
            "corrected_exit_analysis": {
                "enable_exit_behavior_bonus": True,        # Bonus for good exit behavior
                "flipper_moonshot_penalty": 0.5,           # Penalty for flippers claiming moonshots
                "actual_exit_roi_weight": 0.7,             # Weight for actual vs final ROI
                "exit_strategy_consistency_bonus": 0.1     # Bonus for consistent exit strategy
            }
        },
        # REALISTIC TP/SL STRATEGY CONFIGURATION
        "tp_sl_strategy": {
            "enable_pattern_based_tp_sl": True,
            "enable_tp_sl_validation": True,            # NEW: Validate TP/SL makes sense
            "auto_correct_inflated_levels": True,       # NEW: Auto-correct inflated levels
            "use_corrected_exit_analysis": True,        # NEW: Use corrected exit data
            "patterns": {
                "flipper": {
                    "tp1_range": [15, 50],              # CORRECTED: Realistic flipper levels
                    "tp2_range": [30, 80],              # CORRECTED: Realistic flipper levels
                    "tp3_range": [50, 120],             # CORRECTED: Realistic flipper levels
                    "stop_loss_range": [-25, -8],       # Tight SL for flippers
                    "max_tp1": 80,                      # Hard cap for TP1
                    "max_tp2": 120,                     # Hard cap for TP2
                    "validation_enabled": True          # Enable validation
                },
                "skilled_flipper": {
                    "tp1_range": [25, 60],
                    "tp2_range": [40, 100],
                    "tp3_range": [60, 150],
                    "stop_loss_range": [-20, -10],
                    "max_tp1": 100,
                    "max_tp2": 150,
                    "validation_enabled": True
                },
                "sniper": {
                    "tp1_range": [30, 80],
                    "tp2_range": [60, 150],
                    "tp3_range": [100, 250],
                    "stop_loss_range": [-30, -12],
                    "max_tp1": 120,
                    "max_tp2": 200,
                    "validation_enabled": True
                },
                "gem_hunter": {
                    "tp1_range": [60, 200],             # Higher but still realistic
                    "tp2_range": [150, 400],            # Higher but still realistic
                    "tp3_range": [300, 700],            # Higher but still realistic
                    "stop_loss_range": [-50, -20],
                    "max_tp1": 300,                     # Cap even for gem hunters
                    "max_tp2": 600,                     # Cap even for gem hunters
                    "validation_enabled": True
                },
                "verified_gem_hunter": {
                    "tp1_range": [100, 250],
                    "tp2_range": [200, 500],
                    "tp3_range": [400, 800],
                    "stop_loss_range": [-45, -25],
                    "max_tp1": 350,
                    "max_tp2": 700,
                    "validation_enabled": True
                },
                "position_trader": {
                    "tp1_range": [50, 150],
                    "tp2_range": [100, 300],
                    "tp3_range": [200, 500],
                    "stop_loss_range": [-40, -20],
                    "max_tp1": 200,
                    "max_tp2": 400,
                    "validation_enabled": True
                },
                "consistent_trader": {
                    "tp1_range": [40, 100],
                    "tp2_range": [80, 200],
                    "tp3_range": [150, 350],
                    "stop_loss_range": [-35, -15],
                    "max_tp1": 150,
                    "max_tp2": 300,
                    "validation_enabled": True
                }
            },
            "safety_buffers": {
                "tp_buffer_multiplier": 1.0,        # No inflating buffer
                "sl_buffer_multiplier": 1.0         # No deflating buffer
            },
            # EXIT BEHAVIOR INFERENCE SETTINGS
            "exit_behavior_inference": {
                "enable_inference": True,                       # Enable exit behavior inference
                "flipper_exit_roi_cap": 80,                    # Cap flipper exit ROI estimates
                "partial_exit_roi_multiplier": 0.6,            # Assume 60% of pump for partial exits
                "diamond_hands_roi_multiplier": 0.9,           # Assume 90% for long holds
                "quick_loss_cut_max": -20,                     # Max loss for quick cuts
                "confidence_thresholds": {
                    "high_confidence_patterns": ["flipper", "skilled_flipper"],
                    "medium_confidence_patterns": ["sniper", "gem_hunter"],
                    "low_confidence_patterns": ["mixed_strategy", "bag_holder"]
                }
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
            # CSV SETTINGS WITH CORRECTED TP/SL
            "csv_precision": {
                "days_since_last_trade": 1,     # 1 decimal place
                "avg_sol_buy_per_token": 1,     # 1 decimal place
                "composite_score": 1,           # 1 decimal place
                "roi_values": 2,                # 2 decimal places
                "usd_values": 1                 # 1 decimal place
            },
            "exclude_columns": ["total_buys_30_days", "total_sells_30_days"],
            "include_columns": ["unique_tokens_30d"],
            # CORRECTED TP/SL EXPORT SETTINGS
            "tp_sl_export": {
                "validate_before_export": True,         # Validate TP/SL before export
                "show_validation_warnings": True,       # Show validation warnings in export
                "include_pattern_context": True,        # Include pattern in TP/SL export
                "cap_inflated_levels": True,            # Cap inflated levels automatically
                "export_corrected_values": True         # Export corrected values instead of raw
            }
        },
        "logging": {
            "level": "INFO",
            "file": "zeus.log",
            "max_file_size_mb": 10,
            "backup_count": 3,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            # ENHANCED LOGGING FOR CORRECTED ANALYSIS
            "log_api_responses": False,                 # Log full API responses (debug mode)
            "log_field_discovery": True,               # Log Cielo field discovery
            "log_pattern_analysis": True,              # Log trade pattern analysis
            "log_exit_behavior_inference": True,       # NEW: Log exit behavior inference
            "log_tp_sl_validation": True,              # NEW: Log TP/SL validation
            "log_corrected_analysis": True,            # NEW: Log corrected analysis details
            "log_inflation_detection": True            # NEW: Log inflated TP/SL detection
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
            # CORRECTED EXIT ANALYSIS FEATURES
            "enable_token_pnl_analysis": True,              # Token PnL analysis
            "enable_corrected_exit_analysis": True,         # NEW: Corrected exit analysis
            "enable_realistic_tp_sl": True,                 # NEW: Realistic TP/SL recommendations
            "enable_exit_behavior_inference": True,         # NEW: Exit behavior inference
            "enable_tp_sl_validation": True,                # NEW: TP/SL validation
            "enable_pattern_based_strategies": True,        # Pattern-based strategies
            "enable_inflation_detection": True,             # NEW: Detect inflated TP/SL
            "enable_auto_correction": True,                 # NEW: Auto-correct inflated levels
            # LEGACY FEATURES
            "enable_smart_tp_sl": True,                     # Legacy: Smart TP/SL (now corrected)
            "enable_direct_field_extraction": True,        # Direct field extraction
            "enable_field_validation": True                # Field validation
        },
        # VALIDATION AND CORRECTION SETTINGS
        "validation": {
            "enable_comprehensive_validation": True,        # Enable comprehensive validation
            "tp_sl_validation": {
                "enable_pattern_validation": True,         # Validate TP/SL for patterns
                "enable_inflation_detection": True,        # Detect inflated levels
                "enable_auto_correction": True,            # Auto-correct inflated levels
                "severity_thresholds": {
                    "critical": {"flipper_tp1_over": 150, "any_tp1_over": 500},
                    "high": {"flipper_tp1_over": 100, "any_tp1_over": 300},
                    "moderate": {"flipper_tp1_over": 80, "any_tp1_over": 200}
                }
            },
            "exit_analysis_validation": {
                "require_minimum_swaps": 1,                # Minimum swaps for analysis
                "require_realistic_hold_times": True,      # Validate hold times
                "flag_impossible_scenarios": True          # Flag impossible exit scenarios
            }
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
        Initialize Zeus configuration with corrected exit analysis features.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.config_file = config_file or self._get_default_config_path()
        self.config = self._load_config()
        self._validate_config()
        self._setup_corrected_features()
    
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
                logger.info("Using default configuration with CORRECTED EXIT ANALYSIS")
        else:
            logger.info("Configuration file not found, using defaults with CORRECTED EXIT ANALYSIS features")
        
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
            # CORRECTED EXIT ANALYSIS ENVIRONMENT VARIABLES
            'ZEUS_ENABLE_CORRECTED_EXIT_ANALYSIS': ['features', 'enable_corrected_exit_analysis'],
            'ZEUS_ENABLE_REALISTIC_TP_SL': ['features', 'enable_realistic_tp_sl'],
            'ZEUS_ENABLE_TP_SL_VALIDATION': ['features', 'enable_tp_sl_validation'],
            'ZEUS_ENABLE_AUTO_CORRECTION': ['features', 'enable_auto_correction'],
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
                    elif final_key in ['enable_corrected_exit_analysis', 'enable_realistic_tp_sl', 
                                     'enable_tp_sl_validation', 'enable_auto_correction']:
                        current[final_key] = env_value.lower() in ['true', '1', 'yes', 'on']
                    else:
                        current[final_key] = env_value
                    
                    logger.info(f"Applied environment override: {env_var} = {env_value}")
                except ValueError:
                    logger.warning(f"Invalid value for {env_var}: {env_value}")
        
        return config
    
    def _validate_config(self) -> None:
        """Validate configuration values with REQUIRED API key checks and corrected features."""
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
            
            # Validate CORRECTED thresholds
            very_short_minutes = analysis.get('very_short_threshold_minutes', 5)
            if not (0.1 <= very_short_minutes <= 60):
                logger.warning("very_short_threshold_minutes should be between 0.1 and 60")
            
            long_hold_hours = analysis.get('long_hold_threshold_hours', 24)
            if not (1 <= long_hold_hours <= 168):  # Max 1 week
                logger.warning("long_hold_threshold_hours should be between 1 and 168")
            
            # Validate TP/SL pattern configuration
            self._validate_tp_sl_patterns()
            
            logger.info("✅ Configuration validation completed - All REQUIRED APIs configured")
            logger.info("🎯 CORRECTED EXIT ANALYSIS: Realistic TP/SL, Pattern validation, Auto-correction")
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Configuration validation failed: {str(e)}")
            raise
    
    def _validate_tp_sl_patterns(self) -> None:
        """Validate TP/SL pattern configuration for realism."""
        try:
            tp_sl_config = self.config.get('tp_sl_strategy', {})
            patterns = tp_sl_config.get('patterns', {})
            
            for pattern_name, pattern_config in patterns.items():
                if not isinstance(pattern_config, dict):
                    continue
                
                # Check for realistic ranges
                tp1_range = pattern_config.get('tp1_range', [0, 100])
                tp2_range = pattern_config.get('tp2_range', [0, 200])
                max_tp1 = pattern_config.get('max_tp1', 100)
                max_tp2 = pattern_config.get('max_tp2', 200)
                
                # Validate flipper patterns have realistic caps
                if pattern_name in ['flipper', 'skilled_flipper']:
                    if max_tp1 > 150:
                        logger.warning(f"Pattern {pattern_name} has high max_tp1: {max_tp1}% (consider <100%)")
                    if max_tp2 > 200:
                        logger.warning(f"Pattern {pattern_name} has high max_tp2: {max_tp2}% (consider <150%)")
                
                # Validate ranges make sense
                if len(tp1_range) >= 2 and tp1_range[1] > max_tp1:
                    logger.warning(f"Pattern {pattern_name}: tp1_range max ({tp1_range[1]}) > max_tp1 ({max_tp1})")
                
                if len(tp2_range) >= 2 and tp2_range[1] > max_tp2:
                    logger.warning(f"Pattern {pattern_name}: tp2_range max ({tp2_range[1]}) > max_tp2 ({max_tp2})")
            
            logger.info("✅ TP/SL pattern configuration validated")
            
        except Exception as e:
            logger.warning(f"TP/SL pattern validation warning: {str(e)}")
    
    def _setup_corrected_features(self) -> None:
        """Setup and validate corrected exit analysis features."""
        try:
            features = self.config.get('features', {})
            
            # Corrected Exit Analysis setup
            if features.get('enable_corrected_exit_analysis', True):
                logger.info("✅ Corrected Exit Analysis: ENABLED (separates actual vs final ROI)")
                
                # Validate corrected analysis settings
                analysis = self.config.get('analysis', {})
                exit_behavior_enabled = analysis.get('exit_behavior_inference_enabled', True)
                logger.info(f"   Exit behavior inference: {exit_behavior_enabled}")
                
                tp_sl_validation = analysis.get('validate_tp_sl_for_patterns', True)
                logger.info(f"   TP/SL pattern validation: {tp_sl_validation}")
                
                auto_correct = analysis.get('auto_correct_inflated_tp_sl', True)
                logger.info(f"   Auto-correct inflated TP/SL: {auto_correct}")
            else:
                logger.warning("⚠️ Corrected Exit Analysis: DISABLED")
            
            # Realistic TP/SL setup
            if features.get('enable_realistic_tp_sl', True):
                logger.info("✅ Realistic TP/SL Recommendations: ENABLED")
                
                tp_sl_config = self.config.get('tp_sl_strategy', {})
                pattern_based = tp_sl_config.get('enable_pattern_based_tp_sl', True)
                validation_enabled = tp_sl_config.get('enable_tp_sl_validation', True)
                auto_correct_enabled = tp_sl_config.get('auto_correct_inflated_levels', True)
                
                logger.info(f"   Pattern-based TP/SL: {pattern_based}")
                logger.info(f"   TP/SL validation: {validation_enabled}")
                logger.info(f"   Auto-correction: {auto_correct_enabled}")
                
                patterns = tp_sl_config.get('patterns', {})
                logger.info(f"   Configured patterns: {list(patterns.keys())}")
            else:
                logger.warning("⚠️ Realistic TP/SL Recommendations: DISABLED")
            
            # TP/SL Validation setup
            if features.get('enable_tp_sl_validation', True):
                logger.info("✅ TP/SL Validation: ENABLED (prevents inflated recommendations)")
                
                validation_config = self.config.get('validation', {}).get('tp_sl_validation', {})
                pattern_validation = validation_config.get('enable_pattern_validation', True)
                inflation_detection = validation_config.get('enable_inflation_detection', True)
                auto_correction = validation_config.get('enable_auto_correction', True)
                
                logger.info(f"   Pattern validation: {pattern_validation}")
                logger.info(f"   Inflation detection: {inflation_detection}")
                logger.info(f"   Auto-correction: {auto_correction}")
            else:
                logger.warning("⚠️ TP/SL Validation: DISABLED")
            
            # Pattern Recognition setup with corrected thresholds
            if features.get('enable_pattern_based_strategies', True):
                logger.info("✅ Pattern-based Strategies: ENABLED with CORRECTED thresholds")
                
                analysis = self.config.get('analysis', {})
                very_short = analysis.get('very_short_threshold_minutes', 5)
                long_hold = analysis.get('long_hold_threshold_hours', 24)
                logger.info(f"   CORRECTED thresholds: <{very_short}min | >{long_hold}hr")
                logger.info(f"   Pattern validation: Enabled for realistic TP/SL")
            else:
                logger.warning("⚠️ Pattern-based Strategies: DISABLED")
            
        except Exception as e:
            logger.warning(f"Warning during corrected features setup: {str(e)}")
    
    def get_corrected_exit_analysis_config(self) -> Dict[str, Any]:
        """Get corrected exit analysis configuration."""
        return {
            'enabled': self.is_feature_enabled('enable_corrected_exit_analysis'),
            'exit_behavior_inference': self.config.get('analysis', {}).get('exit_behavior_inference_enabled', True),
            'tp_sl_validation': self.config.get('analysis', {}).get('validate_tp_sl_for_patterns', True),
            'auto_correct_inflated': self.config.get('analysis', {}).get('auto_correct_inflated_tp_sl', True),
            'use_realistic_defaults': self.config.get('analysis', {}).get('use_realistic_pattern_defaults', True),
            'thresholds': self.get_pattern_thresholds(),
            'exit_behavior_inference_config': self.config.get('tp_sl_strategy', {}).get('exit_behavior_inference', {})
        }
    
    def get_tp_sl_validation_config(self) -> Dict[str, Any]:
        """Get TP/SL validation configuration."""
        return self.config.get('validation', {}).get('tp_sl_validation', {})
    
    def get_realistic_tp_sl_config(self) -> Dict[str, Any]:
        """Get realistic TP/SL configuration."""
        tp_sl_config = self.config.get('tp_sl_strategy', {})
        return {
            'enabled': tp_sl_config.get('enable_pattern_based_tp_sl', True),
            'validation_enabled': tp_sl_config.get('enable_tp_sl_validation', True),
            'auto_correct_enabled': tp_sl_config.get('auto_correct_inflated_levels', True),
            'use_corrected_analysis': tp_sl_config.get('use_corrected_exit_analysis', True),
            'patterns': tp_sl_config.get('patterns', {}),
            'exit_behavior_inference': tp_sl_config.get('exit_behavior_inference', {})
        }
    
    def validate_system_readiness_corrected(self) -> Dict[str, Any]:
        """
        Validate that the system is ready to run with corrected exit analysis features.
        
        Returns:
            Dict with readiness status and corrected features details
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
        
        # Check corrected features
        features = self.config.get('features', {})
        corrected_features_status = {
            'corrected_exit_analysis': features.get('enable_corrected_exit_analysis', True),
            'realistic_tp_sl': features.get('enable_realistic_tp_sl', True),
            'tp_sl_validation': features.get('enable_tp_sl_validation', True),
            'exit_behavior_inference': features.get('enable_exit_behavior_inference', True),
            'inflation_detection': features.get('enable_inflation_detection', True),
            'auto_correction': features.get('enable_auto_correction', True)
        }
        
        # Determine overall readiness
        all_required_configured = all(required_status.values())
        
        return {
            'system_ready': all_required_configured,
            'required_apis': required_status,
            'recommended_apis': recommended_status,
            'corrected_features': corrected_features_status,
            'missing_required': [k for k, v in required_status.items() if not v],
            'missing_recommended': [k for k, v in recommended_status.items() if not v],
            'readiness_summary': {
                'trading_stats_ready': required_status.get('cielo_api_key', False),
                'token_pnl_ready': required_status.get('cielo_api_key', False) and corrected_features_status.get('corrected_exit_analysis', False),
                'timestamp_accuracy': required_status.get('helius_api_key', False),
                'corrected_exit_analysis_ready': all_required_configured and corrected_features_status.get('corrected_exit_analysis', False),
                'realistic_tp_sl_ready': corrected_features_status.get('realistic_tp_sl', False),
                'tp_sl_validation_ready': corrected_features_status.get('tp_sl_validation', False),
                'enhanced_features': recommended_status.get('birdeye_api_key', False)
            }
        }
    
    def get_cost_estimate_corrected(self, wallet_count: int) -> Dict[str, Any]:
        """
        Calculate estimated API costs for analysis with corrected features.
        
        Args:
            wallet_count: Number of wallets to analyze
            
        Returns:
            Cost estimate breakdown with corrected features
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
                'required': True,
                'description': 'Cielo Trading Stats (REQUIRED for corrected analysis)'
            }
            total_cost += trading_stats_cost
            
            # Token PnL cost for corrected exit analysis
            if features.get('enable_corrected_exit_analysis', True):
                token_pnl_cost = wallet_count * costs.get('cielo_token_pnl', 5)
                cost_breakdown['corrected_token_pnl'] = {
                    'cost_per_wallet': costs.get('cielo_token_pnl', 5),
                    'total_cost': token_pnl_cost,
                    'required': False,
                    'feature': 'Corrected Exit Analysis & Realistic TP/SL',
                    'description': 'Token PnL for exit behavior inference'
                }
                total_cost += token_pnl_cost
            
            # Optional Birdeye costs
            if self.get_api_config().get('birdeye_api_key'):
                birdeye_cost = wallet_count * costs.get('birdeye_token_price', 1)
                cost_breakdown['birdeye'] = {
                    'cost_per_wallet': costs.get('birdeye_token_price', 1),
                    'total_cost': birdeye_cost,
                    'required': False,
                    'feature': 'Enhanced Token Analysis',
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
                'cost_per_wallet': round(total_cost / wallet_count, 2) if wallet_count > 0 else 0,
                'daily_limit': daily_limit,
                'warn_threshold': warn_threshold,
                'exceeds_daily_limit': total_cost > daily_limit,
                'exceeds_warn_threshold': total_cost > (daily_limit * warn_threshold),
                'wallet_count': wallet_count,
                'corrected_features_enabled': True,
                'features_breakdown': {
                    'corrected_exit_analysis': features.get('enable_corrected_exit_analysis', True),
                    'realistic_tp_sl': features.get('enable_realistic_tp_sl', True),
                    'tp_sl_validation': features.get('enable_tp_sl_validation', True)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating corrected cost estimate: {str(e)}")
            return {
                'total_cost': 0,
                'error': str(e),
                'wallet_count': wallet_count
            }
    
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
        """Get TP/SL strategy configuration."""
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
        """Get corrected pattern recognition thresholds."""
        analysis = self.get_analysis_config()
        return {
            'very_short_threshold_minutes': analysis.get('very_short_threshold_minutes', 5),
            'very_short_threshold_hours': analysis.get('very_short_threshold_minutes', 5) / 60.0,
            'long_hold_threshold_hours': analysis.get('long_hold_threshold_hours', 24)
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        api_keys = self.get_api_config()
        configured_apis = [name.replace('_api_key', '') for name, key in api_keys.items() if key]
        
        readiness = self.validate_system_readiness_corrected()
        status = "READY" if readiness['system_ready'] else "NOT READY"
        
        # Show corrected features status
        corrected_features = readiness.get('corrected_features', {})
        enabled_features = [name for name, enabled in corrected_features.items() if enabled]
        
        return f"ZeusConfig({status}, APIs: {', '.join(configured_apis)}, Corrected Features: {', '.join(enabled_features)}, File: {self.config_file})"

def load_zeus_config(config_file: Optional[str] = None) -> ZeusConfig:
    """
    Load Zeus configuration with corrected exit analysis features.
    
    Args:
        config_file: Optional configuration file path
        
    Returns:
        ZeusConfig instance
        
    Raises:
        ValueError: If required API keys are missing
    """
    return ZeusConfig(config_file)