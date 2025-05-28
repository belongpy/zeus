"""
Zeus Configuration Management
Handles configuration loading, validation, and defaults

Features:
- JSON-based configuration
- Environment variable support
- Configuration validation
- Default value management
- Configuration migration/updates
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("zeus.config")

class ZeusConfig:
    """Zeus configuration manager with validation and defaults."""
    
    DEFAULT_CONFIG = {
        "api_keys": {
            "birdeye_api_key": "",
            "cielo_api_key": "",
            "helius_api_key": "",
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
            "timeout_seconds": 300
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
                "same_block_seconds": 60,
                "flipper_rate_threshold": 20
            }
        },
        "output": {
            "default_csv": "zeus_analysis.csv",
            "default_summary": "zeus_summary.txt",
            "default_bot_config": "zeus_bot_config.json",
            "include_failed_analyses": True,
            "sort_by_score": True,
            "export_formats": ["csv", "summary"],
            "timestamp_format": "%Y%m%d_%H%M%S"
        },
        "logging": {
            "level": "INFO",
            "file": "zeus.log",
            "max_file_size_mb": 10,
            "backup_count": 3,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "performance": {
            "max_concurrent_analyses": 3,
            "api_timeout_seconds": 30,
            "retry_attempts": 3,
            "retry_delay_seconds": 1,
            "rate_limit_delay_ms": 100
        },
        "features": {
            "enable_enhanced_transactions": True,
            "enable_price_analysis": True,
            "enable_market_cap_analysis": True,
            "enable_bot_detection": True,
            "enable_pattern_recognition": True
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize Zeus configuration.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.config_file = config_file or self._get_default_config_path()
        self.config = self._load_config()
        self._validate_config()
    
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
            logger.info("Configuration file not found, using defaults")
        
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
            'ZEUS_LOG_LEVEL': ['logging', 'level']
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
                    if final_key in ['days_to_analyze', 'min_unique_tokens', 'initial_token_sample', 'max_token_sample']:
                        current[final_key] = int(env_value)
                    elif final_key in ['composite_score_threshold', 'exit_quality_threshold']:
                        current[final_key] = float(env_value)
                    else:
                        current[final_key] = env_value
                    
                    logger.info(f"Applied environment override: {env_var} = {env_value}")
                except ValueError:
                    logger.warning(f"Invalid value for {env_var}: {env_value}")
        
        return config
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        try:
            # Validate analysis settings
            analysis = self.config.get('analysis', {})
            
            if not (1 <= analysis.get('days_to_analyze', 30) <= 365):
                logger.warning("days_to_analyze should be between 1 and 365")
            
            if not (1 <= analysis.get('min_unique_tokens', 6) <= 100):
                logger.warning("min_unique_tokens should be between 1 and 100")
            
            if analysis.get('initial_token_sample', 5) > analysis.get('max_token_sample', 10):
                logger.warning("initial_token_sample should not exceed max_token_sample")
            
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
            
            logger.info("Configuration validation completed")
            
        except Exception as e:
            logger.error(f"Error validating configuration: {str(e)}")
    
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
        """Get API configuration."""
        return self.config.get('api_keys', {})
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration."""
        return self.config.get('analysis', {})
    
    def get_scoring_config(self) -> Dict[str, Any]:
        """Get scoring configuration."""
        return self.config.get('scoring', {})
    
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
    
    def update_api_key(self, api_name: str, api_key: str) -> None:
        """
        Update API key in configuration.
        
        Args:
            api_name: Name of the API (birdeye, cielo, helius)
            api_key: API key value
        """
        api_keys = self.config.setdefault('api_keys', {})
        api_keys[f'{api_name}_api_key'] = api_key
        logger.info(f"Updated {api_name} API key")
    
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
    
    def migrate_config(self, from_version: str = "1.0") -> bool:
        """
        Migrate configuration from older version.
        
        Args:
            from_version: Version to migrate from
            
        Returns:
            bool: True if migration successful
        """
        try:
            logger.info(f"Migrating configuration from version {from_version}")
            
            # Add any missing default keys
            updated = False
            for section, defaults in self.DEFAULT_CONFIG.items():
                if section not in self.config:
                    self.config[section] = defaults.copy()
                    updated = True
                elif isinstance(defaults, dict):
                    for key, default_value in defaults.items():
                        if key not in self.config[section]:
                            self.config[section][key] = default_value
                            updated = True
            
            if updated:
                self.save_config()
                logger.info("Configuration migration completed")
                return True
            else:
                logger.info("No migration needed")
                return True
                
        except Exception as e:
            logger.error(f"Error migrating configuration: {str(e)}")
            return False
    
    def export_config_template(self, output_path: str) -> bool:
        """
        Export configuration template with comments.
        
        Args:
            output_path: Path to save template
            
        Returns:
            bool: True if successful
        """
        try:
            template = {
                "_comment": "Zeus Wallet Analysis System Configuration",
                "_version": "1.0",
                "_description": {
                    "api_keys": "API keys for external services (Cielo required, others optional)",
                    "analysis": "Analysis parameters and thresholds",
                    "scoring": "Scoring system weights and thresholds",
                    "output": "Output format and file settings",
                    "logging": "Logging configuration",
                    "performance": "Performance and rate limiting settings",
                    "features": "Feature toggles"
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
        
        return f"ZeusConfig(APIs: {', '.join(configured_apis)}, File: {self.config_file})"

def load_zeus_config(config_file: Optional[str] = None) -> ZeusConfig:
    """
    Load Zeus configuration.
    
    Args:
        config_file: Optional configuration file path
        
    Returns:
        ZeusConfig instance
    """
    return ZeusConfig(config_file)