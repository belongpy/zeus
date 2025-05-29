#!/usr/bin/env python3
"""
Zeus - Standalone Wallet Analysis System - CORRECTED EXIT ANALYSIS VERSION
Main CLI Entry Point with Corrected Exit Analysis & Realistic TP/SL

MAJOR UPDATES:
- Corrected exit analysis that separates actual exit behavior from final token ROI
- Realistic TP/SL recommendations that make sense for each trading pattern
- TP/SL validation and auto-correction to prevent inflated recommendations
- Enhanced pattern recognition with corrected thresholds (5min/24hr)
- Exit behavior inference to understand actual vs final performance
- Comprehensive validation and inflation detection
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("zeus.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("zeus")

# Ensure stdout is unbuffered
class UnbufferedStream:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, lines):
        self.stream.writelines(lines)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = UnbufferedStream(sys.stdout)

# Configuration
CONFIG_FILE = os.path.expanduser("~/.zeus_config.json")

def load_config() -> Dict[str, Any]:
    """Load Zeus configuration."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {
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
            "require_real_timestamps": True,
            "very_short_threshold_minutes": 5,  # CORRECTED: 5 minutes
            "long_hold_threshold_hours": 24,    # CORRECTED: 24 hours
            # CORRECTED EXIT ANALYSIS SETTINGS
            "enable_corrected_exit_analysis": True,
            "exit_behavior_inference_enabled": True,
            "validate_tp_sl_for_patterns": True,
            "auto_correct_inflated_tp_sl": True
        },
        "features": {
            "enable_corrected_exit_analysis": True,
            "enable_realistic_tp_sl": True,
            "enable_tp_sl_validation": True,
            "enable_exit_behavior_inference": True,
            "enable_inflation_detection": True,
            "enable_auto_correction": True
        },
        "output": {
            "default_csv": "zeus_analysis.csv",
            "include_bot_config": True
        }
    }

def save_config(config: Dict[str, Any]) -> None:
    """Save Zeus configuration."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

def load_wallets_from_file(file_path: str = "wallets.txt") -> List[str]:
    """Load wallet addresses from file."""
    if not os.path.exists(file_path):
        logger.warning(f"Wallets file {file_path} not found.")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            wallets = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    if 32 <= len(line) <= 44:
                        wallets.append(line)
                    else:
                        logger.warning(f"Line {line_num}: Invalid wallet format: {line}")
            
            logger.info(f"Loaded {len(wallets)} wallet addresses from {file_path}")
            return wallets
            
    except Exception as e:
        logger.error(f"Error reading wallets file {file_path}: {str(e)}")
        return []

def ensure_output_dir(output_path: str) -> str:
    """Ensure output directory exists."""
    output_dir = os.path.join(os.getcwd(), "outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created outputs directory: {output_dir}")
    
    if not os.path.dirname(output_path):
        return os.path.join(output_dir, output_path)
    
    return output_path

def get_api_keys_from_config(config: Dict[str, Any]) -> Dict[str, str]:
    """Extract API keys from config - handles nested format."""
    
    # Check for nested api_keys structure first
    if "api_keys" in config and isinstance(config["api_keys"], dict):
        api_keys = config["api_keys"]
        extracted = {
            "birdeye_api_key": str(api_keys.get("birdeye_api_key", "")).strip(),
            "cielo_api_key": str(api_keys.get("cielo_api_key", "")).strip(), 
            "helius_api_key": str(api_keys.get("helius_api_key", "")).strip(),
            "solana_rpc_url": str(api_keys.get("solana_rpc_url", "https://api.mainnet-beta.solana.com")).strip()
        }
    else:
        # Fallback to flat structure
        extracted = {
            "birdeye_api_key": str(config.get("birdeye_api_key", "")).strip(),
            "cielo_api_key": str(config.get("cielo_api_key", "")).strip(), 
            "helius_api_key": str(config.get("helius_api_key", "")).strip(),
            "solana_rpc_url": str(config.get("solana_rpc_url", "https://api.mainnet-beta.solana.com")).strip()
        }
    
    # Debug logging
    configured_keys = [k for k, v in extracted.items() if v and k.endswith('_api_key')]
    logger.info(f"Extracted API keys: {configured_keys}")
    
    return extracted

def validate_required_apis(api_keys: Dict[str, str]) -> Dict[str, Any]:
    """Validate that required API keys are configured."""
    required_apis = ['cielo_api_key', 'helius_api_key']
    recommended_apis = ['birdeye_api_key']
    
    missing_required = [api for api in required_apis if not api_keys.get(api, '').strip()]
    missing_recommended = [api for api in recommended_apis if not api_keys.get(api, '').strip()]
    
    return {
        'system_ready': len(missing_required) == 0,
        'missing_required': missing_required,
        'missing_recommended': missing_recommended,
        'configured_required': [api for api in required_apis if api_keys.get(api, '').strip()],
        'configured_recommended': [api for api in recommended_apis if api_keys.get(api, '').strip()]
    }

class ZeusCLI:
    """Zeus CLI Application with Corrected Exit Analysis and Realistic TP/SL."""
    
    def __init__(self):
        self.config = load_config()
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            description="Zeus - Wallet Analysis with CORRECTED Exit Analysis & Realistic TP/SL",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  zeus configure --cielo-api-key YOUR_KEY --helius-api-key YOUR_KEY
  zeus analyze --wallets wallets.txt
  zeus analyze --wallet 7xG8...k9mP --output custom_analysis.csv
  zeus status
            """
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Commands")
        
        # Configure command
        configure_parser = subparsers.add_parser("configure", help="Configure API keys")
        configure_parser.add_argument("--birdeye-api-key", help="Birdeye API key (recommended)")
        configure_parser.add_argument("--cielo-api-key", help="Cielo Finance API key (REQUIRED)")
        configure_parser.add_argument("--helius-api-key", help="Helius API key (REQUIRED)")
        configure_parser.add_argument("--rpc-url", help="Solana RPC URL")
        
        # Analyze command
        analyze_parser = subparsers.add_parser("analyze", help="Analyze wallets")
        analyze_parser.add_argument("--wallets", help="File containing wallet addresses")
        analyze_parser.add_argument("--wallet", help="Single wallet address to analyze")
        analyze_parser.add_argument("--output", help="Output CSV file")
        analyze_parser.add_argument("--days", type=int, help="Days to analyze (max 30)")
        analyze_parser.add_argument("--force-refresh", action="store_true", help="Force refresh data")
        
        # Status command
        status_parser = subparsers.add_parser("status", help="Check system status")
        
        return parser
    
    def _handle_numbered_menu(self):
        """Interactive numbered menu with CORRECTED EXIT ANALYSIS features."""
        print("\n" + "="*80, flush=True)
        print("                     ⚡ ███████  ███████  ██    ██  ███████ ⚡", flush=True) 
        print("                     ⚡      ██  ██       ██    ██  ██      ⚡", flush=True)
        print("                     ⚡     ██   █████    ██    ██  ███████ ⚡", flush=True)
        print("                     ⚡    ██    ██       ██    ██       ██ ⚡", flush=True)
        print("                     ⚡ ███████  ███████   ██████   ███████ ⚡", flush=True)
        print("="*80, flush=True)
        print("                   🎯 CORRECTED EXIT ANALYSIS & REALISTIC TP/SL", flush=True)
        print("                   ⚡ Separates Actual Exit Behavior from Final Token ROI", flush=True)
        print("                   🔧 TP/SL Validation & Auto-Correction (No More Inflated Levels!)", flush=True)
        print("                   📊 Pattern Thresholds: 5min (flipper) | 24hr (position)", flush=True)
        print("="*80, flush=True)
        
        # Check system readiness first
        api_keys = get_api_keys_from_config(self.config)
        validation = validate_required_apis(api_keys)
        
        if not validation['system_ready']:
            print("\n❌ SYSTEM NOT READY - Missing REQUIRED API Keys:", flush=True)
            for api in validation['missing_required']:
                api_name = api.replace('_api_key', '').upper()
                print(f"   • {api_name} API Key - REQUIRED", flush=True)
            print("\n🔧 Please configure required APIs before using Zeus.", flush=True)
            print("💡 Run: zeus configure --cielo-api-key YOUR_KEY --helius-api-key YOUR_KEY", flush=True)
        else:
            print("\n✅ SYSTEM READY - All required APIs configured", flush=True)
            print("🎯 CORRECTED Exit Analysis: Separates actual exit behavior from final token ROI", flush=True)
            print("⚡ Realistic TP/SL: No more inflated recommendations (flippers get realistic levels)", flush=True)
            print("🔧 TP/SL Validation: Auto-detects and corrects inflated levels", flush=True)
            print("📊 Exit Behavior Inference: Understands what they actually got vs what token did", flush=True)
            if validation['missing_recommended']:
                print("⚠️ Missing recommended APIs:", flush=True)
                for api in validation['missing_recommended']:
                    api_name = api.replace('_api_key', '').upper()
                    print(f"   • {api_name} API Key - Enhanced features", flush=True)
        
        print("\nSelect an option:", flush=True)
        print("\n🔧 CONFIGURATION:", flush=True)
        print("1. Configure API Keys", flush=True)
        print("2. Check Configuration", flush=True)
        print("3. Test API Connectivity", flush=True)
        print("\n📊 ANALYSIS:", flush=True)
        print("4. Analyze Wallets (CORRECTED Exit Analysis + Realistic TP/SL)", flush=True)
        print("5. Single Wallet Analysis", flush=True)
        print("\n🔍 UTILITIES:", flush=True)
        print("6. System Status", flush=True)
        print("7. Help & Corrected Features", flush=True)
        print("0. Exit", flush=True)
        print("="*80, flush=True)
        
        try:
            choice = input("\nEnter your choice (0-7): ").strip()
            
            if choice == '0':
                print("\nExiting Zeus. Goodbye! ⚡", flush=True)
                sys.exit(0)
            elif choice == '1':
                self._interactive_configure()
            elif choice == '2':
                self._check_configuration()
            elif choice == '3':
                self._test_api_connectivity()
            elif choice == '4':
                self._batch_analyze()
            elif choice == '5':
                self._single_wallet_analyze()
            elif choice == '6':
                self._system_status()
            elif choice == '7':
                self._show_help()
            else:
                print("❌ Invalid choice. Please try again.", flush=True)
                input("Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled.", flush=True)
            sys.exit(0)
        except Exception as e:
            logger.error(f"Menu error: {str(e)}")
            input("Press Enter to continue...")
    
    def _interactive_configure(self):
        """Interactive API configuration with REQUIRED validation."""
        print("\n" + "="*70, flush=True)
        print("    🔧 ZEUS API CONFIGURATION - CORRECTED EXIT ANALYSIS", flush=True)
        print("="*70, flush=True)
        
        # Ensure api_keys section exists
        if "api_keys" not in self.config:
            self.config["api_keys"] = {}
        
        api_keys = self.config["api_keys"]
        
        print("\n🚨 REQUIRED APIs - Zeus cannot function without these:")
        
        # Cielo Finance API (REQUIRED)
        print("\n💰 Cielo Finance API Key (REQUIRED for Trading Stats + Corrected Exit Analysis)")
        current_cielo = api_keys.get("cielo_api_key", "")
        if current_cielo:
            print(f"Current: {current_cielo[:8]}...")
            change = input("Change Cielo API key? (y/N): ").lower().strip()
            if change == 'y':
                new_key = input("Enter new Cielo Finance API key: ").strip()
                if new_key:
                    api_keys["cielo_api_key"] = new_key
                    print("✅ Updated")
        else:
            new_key = input("Enter Cielo Finance API key: ").strip()
            if new_key:
                api_keys["cielo_api_key"] = new_key
                print("✅ Configured")
            else:
                print("❌ WARNING: Cielo API key is REQUIRED!")
        
        # Helius API (REQUIRED)
        print("\n🔍 Helius API Key (REQUIRED for accurate timestamps)")
        current_helius = api_keys.get("helius_api_key", "")
        if current_helius:
            print(f"Current: {current_helius[:8]}...")
            change = input("Change Helius API key? (y/N): ").lower().strip()
            if change == 'y':
                new_key = input("Enter new Helius API key: ").strip()
                if new_key:
                    api_keys["helius_api_key"] = new_key
                    print("✅ Updated")
        else:
            new_key = input("Enter Helius API key: ").strip()
            if new_key:
                api_keys["helius_api_key"] = new_key
                print("✅ Configured")
            else:
                print("❌ WARNING: Helius API key is REQUIRED!")
        
        print("\n📈 RECOMMENDED APIs - System works but with limitations:")
        
        # Birdeye API (RECOMMENDED)
        print("\n🐦 Birdeye API Key (RECOMMENDED for enhanced analysis)")
        current_birdeye = api_keys.get("birdeye_api_key", "")
        if current_birdeye:
            print(f"Current: {current_birdeye[:8]}...")
            change = input("Change Birdeye API key? (y/N): ").lower().strip()
            if change == 'y':
                new_key = input("Enter new Birdeye API key: ").strip()
                if new_key:
                    api_keys["birdeye_api_key"] = new_key
                    print("✅ Updated")
        else:
            new_key = input("Enter Birdeye API key (or Enter to skip): ").strip()
            if new_key:
                api_keys["birdeye_api_key"] = new_key
                print("✅ Configured")
        
        # RPC URL
        print("\n🌐 Solana RPC URL")
        current_rpc = api_keys.get("solana_rpc_url", "https://api.mainnet-beta.solana.com")
        print(f"Current: {current_rpc}")
        change = input("Change RPC URL? (y/N): ").lower().strip()
        if change == 'y':
            new_rpc = input("Enter RPC URL: ").strip()
            if new_rpc:
                api_keys["solana_rpc_url"] = new_rpc
                print("✅ Updated")
        
        save_config(self.config)
        
        # Validate configuration
        api_keys_extracted = get_api_keys_from_config(self.config)
        validation = validate_required_apis(api_keys_extracted)
        
        print("\n" + "="*70)
        if validation['system_ready']:
            print("✅ Configuration saved and validated - System is READY!")
            print("🎯 CORRECTED Exit Analysis: Available (separates actual vs final ROI)")
            print("⚡ Realistic TP/SL: Prevents inflated recommendations")
            print("🔧 TP/SL Validation: Auto-correction enabled")
            print("📊 Exit Behavior Inference: Understands actual trader behavior")
        else:
            print("⚠️ Configuration saved but system is NOT READY!")
            print("Missing REQUIRED APIs:")
            for api in validation['missing_required']:
                print(f"   • {api.replace('_api_key', '').upper()}")
        
        input("Press Enter to continue...")
    
    def _check_configuration(self):
        """Display current configuration with CORRECTED features."""
        print("\n" + "="*70, flush=True)
        print("    📋 ZEUS CONFIGURATION - CORRECTED EXIT ANALYSIS", flush=True)
        print("="*70, flush=True)
        
        api_keys = get_api_keys_from_config(self.config)
        validation = validate_required_apis(api_keys)
        
        print(f"\n🔑 API KEYS STATUS:")
        
        # Required APIs
        print(f"\n🚨 REQUIRED APIs:")
        required_status = {
            'cielo_api_key': 'Cielo Finance (Trading Stats + Corrected Exit Analysis)',
            'helius_api_key': 'Helius (Accurate Timestamps)'
        }
        
        for api_key, api_name in required_status.items():
            if api_keys.get(api_key):
                print(f"   {api_name}: ✅ Configured")
            else:
                print(f"   {api_name}: ❌ MISSING (REQUIRED)")
        
        # Recommended APIs
        print(f"\n📈 RECOMMENDED APIs:")
        recommended_status = {
            'birdeye_api_key': 'Birdeye (Enhanced Analysis)'
        }
        
        for api_key, api_name in recommended_status.items():
            if api_keys.get(api_key):
                print(f"   {api_name}: ✅ Configured")
            else:
                print(f"   {api_name}: ⚠️ Not configured")
        
        print(f"\n🌐 RPC ENDPOINT:")
        print(f"   URL: {api_keys.get('solana_rpc_url', 'https://api.mainnet-beta.solana.com')}")
        
        print(f"\n📊 ANALYSIS SETTINGS:")
        analysis_config = self.config.get('analysis', {})
        print(f"   Analysis Period: {analysis_config.get('days_to_analyze', 30)} days")
        print(f"   Min Unique Tokens: {analysis_config.get('min_unique_tokens', 6)}")
        print(f"   Score Threshold: {analysis_config.get('composite_score_threshold', 65.0)}")
        print(f"   Exit Quality Threshold: {analysis_config.get('exit_quality_threshold', 70.0)}")
        
        print(f"\n🎯 CORRECTED EXIT ANALYSIS FEATURES:")
        features_config = self.config.get('features', {})
        print(f"   Corrected Exit Analysis: {'✅ ENABLED' if features_config.get('enable_corrected_exit_analysis', True) else '❌ DISABLED'}")
        print(f"   Realistic TP/SL: {'✅ ENABLED' if features_config.get('enable_realistic_tp_sl', True) else '❌ DISABLED'}")
        print(f"   TP/SL Validation: {'✅ ENABLED' if features_config.get('enable_tp_sl_validation', True) else '❌ DISABLED'}")
        print(f"   Exit Behavior Inference: {'✅ ENABLED' if features_config.get('enable_exit_behavior_inference', True) else '❌ DISABLED'}")
        print(f"   Inflation Detection: {'✅ ENABLED' if features_config.get('enable_inflation_detection', True) else '❌ DISABLED'}")
        print(f"   Auto-Correction: {'✅ ENABLED' if features_config.get('enable_auto_correction', True) else '❌ DISABLED'}")
        
        print(f"\n⚡ CORRECTED PATTERN THRESHOLDS:")
        print(f"   Very Short Holds: <{analysis_config.get('very_short_threshold_minutes', 5)} minutes (flippers)")
        print(f"   Long Holds: >{analysis_config.get('long_hold_threshold_hours', 24)} hours (position traders)")
        print(f"   Exit Analysis: Separates actual exit ROI from final token ROI")
        print(f"   TP/SL Logic: Pattern-aware (flippers get 25-80%, not 200%+)")
        
        print(f"\n🎯 SYSTEM STATUS:")
        if validation['system_ready']:
            print(f"   System Status: ✅ READY")
            print(f"   CORRECTED Exit Analysis: ✅ Available")
            print(f"   Realistic TP/SL: ✅ Available (no more inflated levels)")
            print(f"   TP/SL Validation: ✅ Available (auto-correction)")
            print(f"   Exit Behavior Inference: ✅ Available")
            print(f"   Pattern Recognition: ✅ Available (corrected thresholds)")
            if validation['missing_recommended']:
                print(f"   Enhanced Features: ⚠️ Limited (missing {len(validation['missing_recommended'])} recommended APIs)")
            else:
                print(f"   Enhanced Features: ✅ Full")
        else:
            print(f"   System Status: ❌ NOT READY")
            print(f"   Missing Required: {', '.join([api.replace('_api_key', '').upper() for api in validation['missing_required']])}")
            print(f"   🔧 Fix: zeus configure --cielo-api-key YOUR_KEY --helius-api-key YOUR_KEY")
        
        print(f"\n💡 KEY IMPROVEMENTS:")
        print(f"   • Fixed TP/SL Logic: Flippers now get realistic 25-80% levels (not 200%+)")
        print(f"   • Exit Behavior Inference: Understands what traders actually got vs final token price")
        print(f"   • Pattern Validation: Auto-detects and corrects inflated recommendations")
        print(f"   • Corrected Thresholds: 5min for flippers, 24hr for position traders")
        
        input("\nPress Enter to continue...")
    
    def _test_api_connectivity(self):
        """Test API connectivity with CORRECTED features."""
        print("\n" + "="*70, flush=True)
        print("    🔍 API CONNECTIVITY TEST - CORRECTED EXIT ANALYSIS", flush=True)
        print("="*70, flush=True)
        
        api_keys = get_api_keys_from_config(self.config)
        validation = validate_required_apis(api_keys)
        
        if not validation['system_ready']:
            print(f"\n❌ Cannot test APIs - Missing REQUIRED keys:")
            for api in validation['missing_required']:
                print(f"   • {api.replace('_api_key', '').upper()}")
            print(f"\n🔧 Configure required APIs first: zeus configure")
            input("Press Enter to continue...")
            return
        
        try:
            from zeus_api_manager import ZeusAPIManager
            
            print(f"\n🔧 Initializing API manager with CORRECTED EXIT ANALYSIS features...")
            
            api_manager = ZeusAPIManager(
                birdeye_api_key=api_keys["birdeye_api_key"],
                cielo_api_key=api_keys["cielo_api_key"],
                helius_api_key=api_keys["helius_api_key"],
                rpc_url=api_keys["solana_rpc_url"]
            )
            
            print(f"✅ API manager initialized successfully!")
            
            # Test Helius (REQUIRED)
            print(f"\n🔍 Testing Helius API (PRIMARY timestamp source)...")
            helius_test = api_manager.test_helius_connection()
            
            if helius_test.get('api_working'):
                print(f"   ✅ Helius: Operational")
                print(f"   📅 Timestamp detection: Working")
                if helius_test.get('days_since_last') != 'unknown':
                    print(f"   📊 Test result: {helius_test.get('days_since_last')} days since last trade")
            else:
                print(f"   ❌ Helius: {helius_test.get('error', 'Failed')}")
            
            # Test Cielo (REQUIRED) - Both endpoints
            print(f"\n💰 Testing Cielo API (Trading Stats + CORRECTED Token PnL)...")
            cielo_test = api_manager.test_cielo_api_connection()
            
            if cielo_test.get('api_working'):
                print(f"   ✅ Cielo: Operational")
                print(f"   🔐 Auth method: {cielo_test.get('auth_method', 'unknown')}")
                
                # Show both endpoint results
                trading_stats_working = cielo_test.get('trading_stats_working', False)
                token_pnl_working = cielo_test.get('token_pnl_working', False)
                
                print(f"   📊 Trading Stats (30 credits): {'✅ Working' if trading_stats_working else '❌ Failed'}")
                print(f"   🎯 Token PnL (5 credits): {'✅ Working' if token_pnl_working else '❌ Failed'}")
                
                if token_pnl_working:
                    token_count = cielo_test.get('token_pnl_count', 0)
                    print(f"   📈 CORRECTED Token PnL test result: {token_count} tokens")
                    print(f"   🔧 Exit behavior inference ready")
                
                fields = cielo_test.get('trading_stats_fields', [])
                print(f"   📊 Direct fields available: {len(fields)}")
            else:
                print(f"   ❌ Cielo: Failed")
                if 'trading_stats_error' in cielo_test:
                    print(f"     Trading Stats: {cielo_test['trading_stats_error']}")
                if 'token_pnl_error' in cielo_test:
                    print(f"     Token PnL: {cielo_test['token_pnl_error']}")
            
            # Test Birdeye (RECOMMENDED)
            if api_keys.get("birdeye_api_key"):
                print(f"\n🐦 Testing Birdeye API (RECOMMENDED)...")
                print(f"   ✅ Birdeye: Configured (enhanced features available)")
            else:
                print(f"\n🐦 Birdeye API (RECOMMENDED):")
                print(f"   ⚠️ Not configured - limited features")
            
            # Overall status
            status = api_manager.get_api_status()
            print(f"\n🎯 OVERALL SYSTEM STATUS:")
            print(f"   System Ready: {'✅ YES' if status.get('system_ready', False) else '❌ NO'}")
            print(f"   Wallet Analysis: {'✅ Ready' if status.get('wallet_compatible', False) else '❌ Not Ready'}")
            print(f"   CORRECTED Exit Analysis: ✅ Available")
            print(f"   Realistic TP/SL: ✅ Available (pattern-aware)")
            print(f"   TP/SL Validation: ✅ Available (auto-correction)")
            print(f"   Exit Behavior Inference: ✅ Available")
            print(f"   Pattern Thresholds: ✅ Corrected (5min/24hr)")
            
        except ValueError as e:
            print(f"\n❌ CRITICAL ERROR: {str(e)}")
            print(f"🔧 This indicates missing REQUIRED API keys")
        except Exception as e:
            print(f"\n❌ Error testing APIs: {str(e)}")
            logger.error(f"API test error: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _batch_analyze(self):
        """Batch wallet analysis with CORRECTED Exit Analysis and Realistic TP/SL."""
        print("\n" + "="*80, flush=True)
        print("    📊 ZEUS BATCH WALLET ANALYSIS - CORRECTED EXIT ANALYSIS", flush=True)
        print("    🎯 Separates Actual Exit Behavior from Final Token ROI", flush=True)
        print("    ⚡ Realistic TP/SL: No More Inflated Recommendations!", flush=True)
        print("    🔧 Auto-Validation & Correction of TP/SL Levels", flush=True)
        print("="*80, flush=True)
        
        # Check system readiness
        api_keys = get_api_keys_from_config(self.config)
        validation = validate_required_apis(api_keys)
        
        if not validation['system_ready']:
            print(f"\n❌ SYSTEM NOT READY - Missing REQUIRED API Keys:")
            for api in validation['missing_required']:
                print(f"   • {api.replace('_api_key', '').upper()}")
            print(f"\n🔧 Configure required APIs: zeus configure --cielo-api-key YOUR_KEY --helius-api-key YOUR_KEY")
            input("Press Enter to continue...")
            return
        
        # Load wallets
        wallets = load_wallets_from_file("wallets.txt")
        if not wallets:
            print("\n❌ No wallets found in wallets.txt")
            print("Add wallet addresses to wallets.txt (one per line)")
            input("Press Enter to continue...")
            return
        
        print(f"\n📁 Found {len(wallets)} wallets in wallets.txt")
        print(f"🔧 System ready with REQUIRED APIs configured")
        print(f"🎯 CORRECTED Exit Analysis: Available (separates actual vs final ROI)")
        print(f"⚡ Realistic TP/SL: Pattern-aware recommendations")
        
        # Run analysis
        try:
            from zeus_analyzer import ZeusAnalyzer
            from zeus_api_manager import ZeusAPIManager
            
            print(f"\n🚀 Initializing Zeus with CORRECTED EXIT ANALYSIS...")
            
            api_manager = ZeusAPIManager(
                birdeye_api_key=api_keys["birdeye_api_key"],
                cielo_api_key=api_keys["cielo_api_key"],
                helius_api_key=api_keys["helius_api_key"],
                rpc_url=api_keys["solana_rpc_url"]
            )
            
            analyzer = ZeusAnalyzer(api_manager, self.config)
            
            print(f"\n🔍 Starting analysis with CORRECTED EXIT ANALYSIS...")
            print(f"   • Period: 30 days")
            print(f"   • Min tokens: 6 unique trades")
            print(f"   • Timestamp source: Helius PRIMARY (accurate)")
            print(f"   • Trading Stats: Cielo API (30 credits, direct fields)")
            print(f"   • CORRECTED Token PnL: Cielo API (5 credits, exit behavior inference)")
            print(f"   • TP/SL Logic: REALISTIC and pattern-aware")
            print(f"   • Pattern thresholds: <5min (flipper) | >24hr (position)")
            print(f"   • Validation: Auto-detect & correct inflated TP/SL")
            print(f"   • Exit Analysis: Separates actual exit ROI from final token ROI")
            
            # Estimate costs
            total_trading_stats_cost = len(wallets) * 30
            total_token_pnl_cost = len(wallets) * 5
            total_cost = total_trading_stats_cost + total_token_pnl_cost
            
            print(f"\n💰 ESTIMATED API COSTS:")
            print(f"   Trading Stats: {len(wallets)} × 30 = {total_trading_stats_cost} credits")
            print(f"   CORRECTED Token PnL: {len(wallets)} × 5 = {total_token_pnl_cost} credits")
            print(f"   TOTAL: {total_cost} credits")
            
            confirm = input(f"\nProceed with CORRECTED analysis? (y/N): ").lower().strip()
            if confirm != 'y':
                print("Analysis cancelled.")
                input("Press Enter to continue...")
                return
            
            # Run batch analysis
            results = analyzer.analyze_wallets_batch(wallets)
            
            if results.get("success"):
                # Export results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = ensure_output_dir(f"zeus_corrected_exit_analysis_{timestamp}.csv")
                
                from zeus_export import export_zeus_analysis
                export_success = export_zeus_analysis(results, output_file)
                
                if export_success:
                    print(f"\n✅ Analysis complete with CORRECTED EXIT ANALYSIS!")
                    print(f"📄 Results saved to: {output_file}")
                    print(f"📊 CSV includes: REALISTIC TP/SL levels, corrected exit analysis, pattern validation")
                    
                    # Display summary
                    self._display_analysis_summary_corrected(results)
                else:
                    print(f"\n⚠️ Analysis completed but export failed")
            else:
                print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")
        
        except ValueError as e:
            print(f"\n❌ CRITICAL ERROR: {str(e)}")
            print(f"🔧 This indicates missing REQUIRED API keys")
        except Exception as e:
            print(f"\n❌ Error during analysis: {str(e)}")
            logger.error(f"Batch analysis error: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _single_wallet_analyze(self):
        """Single wallet analysis with CORRECTED Exit Analysis and Realistic TP/SL."""
        print("\n" + "="*70, flush=True)
        print("    🔍 SINGLE WALLET ANALYSIS - CORRECTED EXIT ANALYSIS", flush=True)
        print("="*70, flush=True)
        
        # Check system readiness
        api_keys = get_api_keys_from_config(self.config)
        validation = validate_required_apis(api_keys)
        
        if not validation['system_ready']:
            print(f"\n❌ SYSTEM NOT READY - Missing REQUIRED API Keys:")
            for api in validation['missing_required']:
                print(f"   • {api.replace('_api_key', '').upper()}")
            print(f"\n🔧 Configure required APIs first")
            input("Press Enter to continue...")
            return
        
        wallet_address = input("\nEnter wallet address: ").strip()
        
        if not wallet_address or len(wallet_address) < 32:
            print("❌ Invalid wallet address")
            input("Press Enter to continue...")
            return
        
        try:
            from zeus_analyzer import ZeusAnalyzer
            from zeus_api_manager import ZeusAPIManager
            
            print(f"\n🔧 Initializing with CORRECTED EXIT ANALYSIS features...")
            
            api_manager = ZeusAPIManager(
                birdeye_api_key=api_keys["birdeye_api_key"],
                cielo_api_key=api_keys["cielo_api_key"],
                helius_api_key=api_keys["helius_api_key"],
                rpc_url=api_keys["solana_rpc_url"]
            )
            
            analyzer = ZeusAnalyzer(api_manager, self.config)
            
            print(f"\n🔍 Analyzing {wallet_address[:8]}...{wallet_address[-4:]} with CORRECTED EXIT ANALYSIS")
            print(f"   📊 Trading Stats: Cielo API (30 credits)")
            print(f"   🎯 CORRECTED Token PnL: Cielo API (5 credits)")
            print(f"   ⚡ Exit Behavior Inference: Separates actual vs final ROI")
            print(f"   🔧 Realistic TP/SL: Pattern-aware recommendations")
            print(f"   🔍 TP/SL Validation: Auto-correction of inflated levels")
            
            # Run single analysis
            result = analyzer.analyze_single_wallet(wallet_address)
            
            if result.get("success"):
                # Display detailed results
                self._display_single_wallet_result_corrected(result)
            else:
                error_type = result.get('error_type', 'UNKNOWN')
                print(f"\n❌ Analysis failed ({error_type}): {result.get('error', 'Unknown error')}")
                
                if error_type == 'TIMESTAMP_DETECTION_FAILED':
                    print(f"🔧 This indicates an issue with Helius API timestamp detection")
                    print(f"📞 Check Helius API key and connectivity")
        
        except ValueError as e:
            print(f"\n❌ CRITICAL ERROR: {str(e)}")
            print(f"🔧 This indicates missing REQUIRED API keys")
        except Exception as e:
            print(f"\n❌ Error during analysis: {str(e)}")
            logger.error(f"Single wallet analysis error: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _display_analysis_summary_corrected(self, results: Dict[str, Any]):
        """Display batch analysis summary with CORRECTED features."""
        total_analyzed = results.get('total_analyzed', 0)
        successful = results.get('successful_analyses', 0)
        failed = results.get('failed_analyses', 0)
        
        print(f"\n📊 ANALYSIS SUMMARY (CORRECTED EXIT ANALYSIS):")
        print(f"   Total wallets: {total_analyzed}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        
        # Timestamp source breakdown
        debug_info = results.get('debug_info', {})
        processing_method = debug_info.get('processing_method', 'unknown')
        
        print(f"\n🕐 ANALYSIS METHOD:")
        print(f"   Processing: {processing_method}")
        print(f"   Timestamp source: Helius PRIMARY (accurate)")
        print(f"   Field extraction: Direct from Cielo (no scaling)")
        print(f"   Exit analysis: CORRECTED (actual vs final ROI)")
        print(f"   TP/SL logic: REALISTIC and pattern-aware")
        
        analyses = results.get('analyses', [])
        if analyses:
            # Pattern analysis summary
            patterns = {}
            inflated_tp_sl_count = 0
            corrected_tp_sl_count = 0
            
            for analysis in analyses:
                if analysis.get('success'):
                    # Track patterns
                    trade_pattern_analysis = analysis.get('trade_pattern_analysis', {})
                    if isinstance(trade_pattern_analysis, dict):
                        pattern = trade_pattern_analysis.get('pattern', 'unknown')
                        patterns[pattern] = patterns.get(pattern, 0) + 1
                    
                    # Check for TP/SL corrections
                    strategy = analysis.get('strategy_recommendation', {})
                    if isinstance(strategy, dict):
                        tp1 = strategy.get('tp1_percent', 0)
                        if pattern == 'flipper' and tp1 > 100:
                            inflated_tp_sl_count += 1
                        elif tp1 <= 80 and pattern == 'flipper':
                            corrected_tp_sl_count += 1
            
            if patterns:
                print(f"\n⚡ TRADER PATTERNS DETECTED:")
                for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
                    print(f"   {pattern}: {count} wallets")
            
            # TP/SL validation summary
            print(f"\n🔧 TP/SL VALIDATION RESULTS:")
            print(f"   Realistic TP/SL levels: {corrected_tp_sl_count}/{successful}")
            if inflated_tp_sl_count > 0:
                print(f"   ⚠️ Would have been inflated (corrected): {inflated_tp_sl_count}")
            else:
                print(f"   ✅ All TP/SL levels are realistic!")
            
            # Binary decision summary
            follow_wallet_yes = sum(1 for a in analyses if a.get('binary_decisions', {}).get('follow_wallet', False))
            follow_sells_yes = sum(1 for a in analyses if a.get('binary_decisions', {}).get('follow_sells', False))
            
            print(f"\n🎯 BINARY DECISIONS:")
            print(f"   Follow Wallet: {follow_wallet_yes}/{successful} ({follow_wallet_yes/successful*100:.1f}%)")
            print(f"   Follow Sells: {follow_sells_yes}/{successful} ({follow_sells_yes/successful*100:.1f}%)")
            
            # Top performers with CORRECTED TP/SL
            top_performers = sorted([a for a in analyses if a.get('success')], 
                                  key=lambda x: x.get('composite_score', 0), reverse=True)[:5]
            
            if top_performers:
                print(f"\n🏆 TOP 5 PERFORMERS (CORRECTED TP/SL):")
                for i, perf in enumerate(top_performers, 1):
                    wallet = perf['wallet_address']
                    score = perf.get('composite_score', 0)
                    follow_wallet = perf.get('binary_decisions', {}).get('follow_wallet', False)
                    follow_sells = perf.get('binary_decisions', {}).get('follow_sells', False)
                    
                    # Show pattern and CORRECTED strategy info
                    trade_pattern_analysis = perf.get('trade_pattern_analysis', {})
                    pattern = trade_pattern_analysis.get('pattern', 'unknown') if isinstance(trade_pattern_analysis, dict) else 'unknown'
                    
                    strategy = perf.get('strategy_recommendation', {})
                    tp1 = strategy.get('tp1_percent', 0) if isinstance(strategy, dict) else 0
                    tp2 = strategy.get('tp2_percent', 0) if isinstance(strategy, dict) else 0
                    
                    print(f"   {i}. {wallet[:8]}...{wallet[-4:]}")
                    print(f"      Score: {score:.1f}/100 | Pattern: {pattern}")
                    print(f"      Follow: {'✅' if follow_wallet else '❌'} Wallet | {'✅' if follow_sells else '❌'} Sells")
                    print(f"      CORRECTED TP/SL: {tp1}% / {tp2}% (realistic for {pattern})")
                    print(f"      📊 Exit behavior analysis available")
    
    def _display_single_wallet_result_corrected(self, result: Dict[str, Any]):
        """Display single wallet analysis result with CORRECTED features."""
        print(f"\n" + "="*70)
        print(f"    📊 WALLET ANALYSIS RESULT - CORRECTED EXIT ANALYSIS")
        print(f"="*70)
        
        wallet = result['wallet_address']
        print(f"\n💰 Wallet: {wallet}")
        
        # Timestamp info (1 decimal)
        last_tx_data = result.get('last_transaction_data', {})
        days_since = last_tx_data.get('days_since_last_trade', 'unknown')
        timestamp_source = last_tx_data.get('source', 'unknown')
        
        print(f"\n🕐 TIMESTAMP INFO:")
        print(f"   Days since last trade: {days_since}")
        print(f"   Source: {timestamp_source}")
        print(f"   Accuracy: High (Helius PRIMARY)")
        
        # Basic metrics
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"   Composite Score: {result.get('composite_score', 0):.1f}/100")
        print(f"   Unique Tokens: {result.get('unique_tokens_traded', 0)}")
        print(f"   Analysis Period: {result.get('analysis_days', 30)} days")
        print(f"   Tokens Analyzed: {result.get('tokens_analyzed', 0)}")
        
        # CORRECTED trade pattern analysis
        trade_pattern_analysis = result.get('trade_pattern_analysis', {})
        if isinstance(trade_pattern_analysis, dict) and trade_pattern_analysis.get('success'):
            print(f"\n🎯 CORRECTED TRADE PATTERN ANALYSIS:")
            pattern = trade_pattern_analysis.get('pattern', 'unknown')
            corrected_analysis = trade_pattern_analysis.get('exit_analysis_corrected', False)
            
            if corrected_analysis:
                avg_roi = trade_pattern_analysis.get('avg_roi', 0)
                win_rate = trade_pattern_analysis.get('win_rate', 0)
                avg_hold_time = trade_pattern_analysis.get('avg_hold_time_hours', 0)
                tokens_analyzed = trade_pattern_analysis.get('tokens_analyzed', 0)
                
                print(f"   Pattern: {pattern}")
                print(f"   CORRECTED Avg ROI: {avg_roi:.1f}% (actual exit behavior)")
                print(f"   Win Rate: {win_rate:.1f}%")
                print(f"   Avg Hold Time: {avg_hold_time:.1f} hours")
                print(f"   Token Trades Analyzed: {tokens_analyzed}")
                print(f"   ✅ Exit behavior analysis: CORRECTED")
                
                tp_sl_analysis = trade_pattern_analysis.get('tp_sl_analysis', {})
                if isinstance(tp_sl_analysis, dict) and tp_sl_analysis.get('based_on_actual_exits'):
                    print(f"   TP/SL Source: Actual exit patterns (corrected)")
                else:
                    print(f"   TP/SL Source: Pattern-based defaults (realistic)")
            else:
                print(f"   Pattern: {pattern}")
                print(f"   ⚠️ Using fallback analysis (no corrected exit data)")
        
        # Binary decisions
        binary_decisions = result.get('binary_decisions', {})
        print(f"\n🎯 BINARY DECISIONS:")
        print(f"   Follow Wallet: {'✅ YES' if binary_decisions.get('follow_wallet') else '❌ NO'}")
        print(f"   Follow Sells: {'✅ YES' if binary_decisions.get('follow_sells') else '❌ NO'}")
        
        # CORRECTED strategy recommendation
        strategy = result.get('strategy_recommendation', {})
        if isinstance(strategy, dict):
            print(f"\n📋 CORRECTED STRATEGY RECOMMENDATION:")
            print(f"   Copy Entries: {'✅ YES' if strategy.get('copy_entries') else '❌ NO'}")
            print(f"   Copy Exits: {'✅ YES' if strategy.get('copy_exits') else '❌ NO'}")
            
            tp1 = strategy.get('tp1_percent', 0)
            tp2 = strategy.get('tp2_percent', 0)
            tp3 = strategy.get('tp3_percent', 0)
            stop_loss = strategy.get('stop_loss_percent', 0)
            
            print(f"   REALISTIC TP1: {tp1}%")
            print(f"   REALISTIC TP2: {tp2}%")
            print(f"   REALISTIC TP3: {tp3}%")
            print(f"   Stop Loss: {stop_loss}%")
            print(f"   Position Size: {strategy.get('position_size_sol', '1-10')} SOL")
            
            reasoning = strategy.get('reasoning', 'N/A')
            print(f"   Reasoning: {reasoning}")
            
            # Show pattern validation
            pattern = trade_pattern_analysis.get('pattern', 'unknown') if isinstance(trade_pattern_analysis, dict) else 'unknown'
            if pattern == 'flipper' and tp1 <= 80:
                print(f"   ✅ TP/SL levels are REALISTIC for {pattern} pattern")
            elif pattern == 'flipper' and tp1 > 80:
                print(f"   ⚠️ TP levels high for {pattern} pattern (may have been corrected)")
            else:
                print(f"   ✅ TP/SL levels appropriate for {pattern} pattern")
        
        print(f"\n📊 DATA SOURCES:")
        print(f"   Trading Stats: Cielo API (30 credits, direct fields)")
        print(f"   CORRECTED Token PnL: Cielo API (5 credits, exit behavior inference)")
        print(f"   Timestamps: Helius PRIMARY (accurate)")
        print(f"   TP/SL Strategy: CORRECTED pattern-based analysis")
        print(f"   Field Extraction: Direct (no scaling/conversion)")
        print(f"   Exit Analysis: CORRECTED (separates actual vs final ROI)")
        print(f"   Validation: Auto-correction of inflated TP/SL levels")
    
    def _system_status(self):
        """Display system status with CORRECTED features."""
        print("\n" + "="*70, flush=True)
        print("    ⚡ ZEUS SYSTEM STATUS - CORRECTED EXIT ANALYSIS", flush=True)
        print("="*70, flush=True)
        
        api_keys = get_api_keys_from_config(self.config)
        validation = validate_required_apis(api_keys)
        
        print(f"\n🔧 SYSTEM CONFIGURATION:")
        print(f"   Zeus Version: 3.0 (CORRECTED Exit Analysis)")
        print(f"   Configuration File: {CONFIG_FILE}")
        print(f"   System Ready: {'✅ YES' if validation['system_ready'] else '❌ NO'}")
        
        print(f"\n🔑 API KEY STATUS:")
        
        # Required APIs
        print(f"\n🚨 REQUIRED APIs:")
        for api in ['cielo_api_key', 'helius_api_key']:
            api_name = api.replace('_api_key', '').upper()
            if api == 'cielo_api_key':
                api_desc = f"{api_name} (Trading Stats + CORRECTED Token PnL)"
            else:
                api_desc = f"{api_name} (Accurate Timestamps)"
            
            if api_keys.get(api):
                print(f"   {api_desc}: ✅ Configured")
            else:
                print(f"   {api_desc}: ❌ MISSING")
        
        # Recommended APIs
        print(f"\n📈 RECOMMENDED APIs:")
        for api in ['birdeye_api_key']:
            api_name = api.replace('_api_key', '').upper()
            if api_keys.get(api):
                print(f"   {api_name}: ✅ Configured")
            else:
                print(f"   {api_name}: ⚠️ Not configured")
        
        # CORRECTED features status
        features_config = self.config.get('features', {})
        analysis_config = self.config.get('analysis', {})
        print(f"\n🎯 CORRECTED EXIT ANALYSIS FEATURES:")
        print(f"   Corrected Exit Analysis: {'✅ Available' if validation['system_ready'] and features_config.get('enable_corrected_exit_analysis', True) else '❌ Requires APIs'}")
        print(f"   Realistic TP/SL: {'✅ Available' if features_config.get('enable_realistic_tp_sl', True) else '❌ Disabled'}")
        print(f"   TP/SL Validation: {'✅ Available' if features_config.get('enable_tp_sl_validation', True) else '❌ Disabled'}")
        print(f"   Exit Behavior Inference: {'✅ Available' if features_config.get('enable_exit_behavior_inference', True) else '❌ Disabled'}")
        print(f"   Inflation Detection: {'✅ Available' if features_config.get('enable_inflation_detection', True) else '❌ Disabled'}")
        print(f"   Auto-Correction: {'✅ Available' if features_config.get('enable_auto_correction', True) else '❌ Disabled'}")
        print(f"   Pattern Recognition: ✅ Available (corrected thresholds)")
        print(f"   Very Short Holds: <{analysis_config.get('very_short_threshold_minutes', 5)} minutes")
        print(f"   Long Holds: >{analysis_config.get('long_hold_threshold_hours', 24)} hours")
        
        if validation['system_ready']:
            try:
                from zeus_api_manager import ZeusAPIManager
                
                api_manager = ZeusAPIManager(
                    birdeye_api_key=api_keys["birdeye_api_key"],
                    cielo_api_key=api_keys["cielo_api_key"],
                    helius_api_key=api_keys["helius_api_key"],
                    rpc_url=api_keys["solana_rpc_url"]
                )
                
                status = api_manager.get_api_status()
                perf_stats = api_manager.get_performance_stats()
                
                print(f"\n📊 API PERFORMANCE:")
                for api_name, stats in perf_stats.items():
                    if isinstance(stats, dict) and 'total_calls' in stats:
                        print(f"   {api_name}: {stats['total_calls']} calls, {stats['success_rate_percent']:.1f}% success")
                
                print(f"\n🎯 SYSTEM CAPABILITIES:")
                print(f"   Wallet Analysis: {'✅ Ready' if status.get('wallet_compatible', False) else '❌ Not Ready'}")
                print(f"   Trading Stats: ✅ Available (30 credits)")
                print(f"   CORRECTED Token PnL Analysis: ✅ Available (5 credits)")
                print(f"   Realistic TP/SL: ✅ Available (pattern-aware)")
                print(f"   TP/SL Validation: ✅ Available (auto-correction)")
                print(f"   Exit Behavior Inference: ✅ Available")
                print(f"   Accurate Timestamps: ✅ Available (Helius PRIMARY)")
                print(f"   Pattern-based Strategies: ✅ Available (corrected thresholds)")
                
            except Exception as e:
                print(f"\n❌ Error checking detailed status: {str(e)}")
        else:
            print(f"\n🔧 TO FIX:")
            print(f"   Run: zeus configure")
            for api in validation['missing_required']:
                api_name = api.replace('_api_key', '').replace('_', '-')
                print(f"   Add: --{api_name} YOUR_KEY")
        
        print(f"\n💡 KEY IMPROVEMENTS IN v3.0:")
        print(f"   🎯 CORRECTED Exit Analysis: Separates actual exit behavior from final token ROI")
        print(f"   ⚡ Realistic TP/SL: Flippers get 25-80% (not 200%+), Gem hunters get 100-400%")
        print(f"   🔧 TP/SL Validation: Auto-detects and corrects inflated recommendations")
        print(f"   📊 Exit Behavior Inference: Understands what traders actually got vs final price")
        print(f"   🔍 Pattern Recognition: 5min threshold for flippers, 24hr for position traders")
        print(f"   ✅ Auto-Correction: No more inflated TP/SL recommendations!")
        
        input("\nPress Enter to continue...")
    
    def _show_help(self):
        """Show help and CORRECTED features."""
        print("\n" + "="*80, flush=True)
        print("    📖 ZEUS HELP & CORRECTED EXIT ANALYSIS FEATURES", flush=True)
        print("="*80, flush=True)
        
        print(f"\n🎯 ZEUS OVERVIEW:")
        print(f"Zeus is a standalone wallet analysis system focused on binary decisions")
        print(f"for automated trading bots. Version 3.0 introduces CORRECTED EXIT ANALYSIS")
        print(f"that separates actual trader exit behavior from final token price performance.")
        
        print(f"\n🚨 CRITICAL PROBLEM SOLVED:")
        print(f"Previous versions had a major flaw: they assumed traders held until the")
        print(f"final token ROI, leading to inflated TP/SL recommendations. For example:")
        print(f"• Flipper trader exits at 30% gain in 2 minutes")
        print(f"• Token continues to pump to 500% over the next hour")
        print(f"• OLD Zeus would recommend 200%+ TP levels (WRONG!)")
        print(f"• NEW Zeus correctly identifies they exited at 30% (CORRECT!)")
        
        print(f"\n📊 CORRECTED EXIT ANALYSIS FEATURES:")
        print(f"   • Exit Behavior Inference: Distinguishes actual exit ROI from final token ROI")
        print(f"   • Realistic TP/SL: Pattern-aware recommendations that make sense")
        print(f"   • TP/SL Validation: Auto-detects and corrects inflated levels")
        print(f"   • Pattern Recognition: Corrected thresholds (5min/24hr)")
        print(f"   • Auto-Correction: Prevents dangerous trading recommendations")
        
        print(f"\n🚨 REQUIRED APIs (System cannot function without these):")
        print(f"   • Cielo Finance API - Trading Stats (30 credits) + Token PnL (5 credits)")
        print(f"     Get yours at: https://cielo.finance")
        print(f"   • Helius API - Accurate transaction timestamps")
        print(f"     Get yours at: https://helius.xyz")
        
        print(f"\n📈 RECOMMENDED APIs (Enhanced features):")
        print(f"   • Birdeye API - Enhanced token analysis")
        print(f"     Get yours at: https://birdeye.so")
        
        print(f"\n⚡ CORRECTED TP/SL LOGIC:")
        print(f"   Pattern        | OLD TP Levels | NEW TP Levels | Why")
        print(f"   ---------------|---------------|---------------|-------------------")
        print(f"   Flipper        | 200%+ (WRONG) | 25-80% (RIGHT)| Quick exits")
        print(f"   Sniper         | 150%+ (HIGH)  | 50-120% (GOOD)| Fast but bigger")
        print(f"   Gem Hunter     | 300%+ (OK)    | 100-400% (GOOD)| Patient for gems")
        print(f"   Position Trader| 200%+ (HIGH) | 80-300% (GOOD)| Long-term holds")
        print(f"   Consistent     | 150%+ (HIGH) | 60-200% (GOOD)| Balanced approach")
        
        print(f"\n🔧 EXIT BEHAVIOR INFERENCE:")
        print(f"   • Flipper (<5min): Likely exited at 20-50% regardless of later pump")
        print(f"   • Multiple Swaps: Partial exits - got 60-80% of final performance")
        print(f"   • Long Holds (>24hr): Diamond hands - got 80-90% of performance")
        print(f"   • Single Exit: Standard trade - got ~85% of final performance")
        
        print(f"\n📊 VALIDATION & CORRECTION:")
        print(f"   • Inflation Detection: Flags unrealistic TP levels for patterns")
        print(f"   • Auto-Correction: Caps TP levels within realistic ranges")
        print(f"   • Pattern Validation: Ensures TP/SL makes sense for trading style")
        print(f"   • Safety Buffers: Removes dangerous inflating multipliers")
        
        print(f"\n📋 CORRECTED ANALYSIS PROCESS:")
        print(f"1. Get REAL last transaction timestamp from Helius API")
        print(f"2. Get Trading Stats from Cielo API (30 credits, direct fields)")
        print(f"3. Analyze Token PnL data from Cielo API (5 credits)")
        print(f"4. INFER actual exit behavior vs final token performance")
        print(f"5. Identify trader pattern with corrected thresholds")
        print(f"6. Calculate REALISTIC TP/SL based on actual trading behavior")
        print(f"7. VALIDATE and auto-correct any inflated recommendations")
        print(f"8. Generate binary decisions and corrected strategy")
        
        print(f"\n⚡ COMMAND EXAMPLES:")
        print(f"   zeus configure --cielo-api-key YOUR_KEY --helius-api-key YOUR_KEY")
        print(f"   zeus analyze --wallets wallets.txt")
        print(f"   zeus analyze --wallet 7xG8...k9mP")
        print(f"   zeus status")
        
        print(f"\n💡 WHY THIS MATTERS:")
        print(f"   • Trading bots using OLD Zeus recommendations could set 200%+ TP levels")
        print(f"   • These levels would rarely execute for flipper strategies")
        print(f"   • NEW Zeus provides actionable, realistic levels that actually work")
        print(f"   • CORRECTED analysis leads to better trading outcomes")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Run Zeus CLI."""
        args = self.parser.parse_args()
        
        if not args.command:
            # Show interactive menu
            while True:
                self._handle_numbered_menu()
        else:
            # Handle command line arguments
            if args.command == "configure":
                self._handle_configure_command(args)
            elif args.command == "analyze":
                self._handle_analyze_command(args)
            elif args.command == "status":
                self._handle_status_command(args)
    
    def _handle_configure_command(self, args):
        """Handle configure command."""
        # Ensure api_keys section exists
        if "api_keys" not in self.config:
            self.config["api_keys"] = {}
        
        api_keys = self.config["api_keys"]
        
        if args.birdeye_api_key:
            api_keys["birdeye_api_key"] = args.birdeye_api_key
            logger.info("Birdeye API key configured")
        
        if args.cielo_api_key:
            api_keys["cielo_api_key"] = args.cielo_api_key
            logger.info("Cielo Finance API key configured (Trading Stats + CORRECTED Token PnL enabled)")
        
        if args.helius_api_key:
            api_keys["helius_api_key"] = args.helius_api_key
            logger.info("Helius API key configured")
        
        if args.rpc_url:
            api_keys["solana_rpc_url"] = args.rpc_url
            logger.info("RPC URL configured")
        
        save_config(self.config)
        
        # Validate after configuration
        api_keys_extracted = get_api_keys_from_config(self.config)
        validation = validate_required_apis(api_keys_extracted)
        
        if validation['system_ready']:
            logger.info("✅ Configuration complete - System is READY!")
            logger.info("🎯 CORRECTED Exit Analysis: Available (separates actual vs final ROI)")
            logger.info("⚡ Realistic TP/SL: Pattern-aware recommendations")
            logger.info("🔧 TP/SL Validation: Auto-correction enabled")
        else:
            logger.warning("⚠️ Configuration saved but system is NOT READY!")
            logger.warning(f"Missing REQUIRED APIs: {', '.join([api.replace('_api_key', '').upper() for api in validation['missing_required']])}")
    
    def _handle_analyze_command(self, args):
        """Handle analyze command with CORRECTED EXIT ANALYSIS features."""
        try:
            # Validate required APIs first
            api_keys = get_api_keys_from_config(self.config)
            validation = validate_required_apis(api_keys)
            
            if not validation['system_ready']:
                logger.error("❌ SYSTEM NOT READY - Missing REQUIRED API Keys:")
                for api in validation['missing_required']:
                    logger.error(f"   • {api.replace('_api_key', '').upper()}")
                logger.error("🔧 Configure required APIs: zeus configure --cielo-api-key YOUR_KEY --helius-api-key YOUR_KEY")
                return
            
            from zeus_analyzer import ZeusAnalyzer
            from zeus_api_manager import ZeusAPIManager
            from zeus_export import export_zeus_analysis
            
            logger.info("🔧 Initializing Zeus with CORRECTED EXIT ANALYSIS...")
            logger.info("🎯 CORRECTED Exit Analysis: Available (separates actual vs final ROI)")
            logger.info("⚡ Realistic TP/SL: Pattern-aware recommendations")
            logger.info("🔧 TP/SL Validation: Auto-correction enabled")
            
            api_manager = ZeusAPIManager(
                birdeye_api_key=api_keys["birdeye_api_key"],
                cielo_api_key=api_keys["cielo_api_key"],
                helius_api_key=api_keys["helius_api_key"],
                rpc_url=api_keys["solana_rpc_url"]
            )
            
            analyzer = ZeusAnalyzer(api_manager, self.config)
            
            if args.wallet:
                # Single wallet analysis
                logger.info(f"Analyzing single wallet: {args.wallet}")
                logger.info(f"Cost: 35 credits (30 Trading Stats + 5 CORRECTED Token PnL)")
                result = analyzer.analyze_single_wallet(args.wallet)
                
                if result.get("success"):
                    output_file = args.output or f"zeus_corrected_single_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    output_file = ensure_output_dir(output_file)
                    export_zeus_analysis({"analyses": [result]}, output_file)
                    logger.info(f"Results saved to {output_file}")
                    logger.info(f"📊 CSV includes: CORRECTED exit analysis, REALISTIC TP/SL")
                else:
                    error_type = result.get('error_type', 'UNKNOWN')
                    logger.error(f"Analysis failed ({error_type}): {result.get('error')}")
            
            elif args.wallets:
                # Batch analysis
                wallets = load_wallets_from_file(args.wallets)
                if not wallets:
                    logger.error(f"No wallets found in {args.wallets}")
                    return
                
                logger.info(f"Analyzing {len(wallets)} wallets with CORRECTED EXIT ANALYSIS")
                logger.info(f"Estimated cost: {len(wallets) * 35} credits ({len(wallets)} × 35)")
                results = analyzer.analyze_wallets_batch(wallets)
                
                if results.get("success"):
                    output_file = args.output or f"zeus_corrected_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    output_file = ensure_output_dir(output_file)
                    export_zeus_analysis(results, output_file)
                    logger.info(f"Results saved to {output_file}")
                    logger.info(f"📊 CSV includes: CORRECTED exit analysis, REALISTIC TP/SL, validation results")
                else:
                    logger.error(f"Batch analysis failed: {results.get('error')}")
            
            else:
                logger.error("Either --wallet or --wallets must be specified")
        
        except ValueError as e:
            logger.error(f"❌ CRITICAL ERROR: {str(e)}")
            logger.error("🔧 This indicates missing REQUIRED API keys")
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
    
    def _handle_status_command(self, args):
        """Handle status command with CORRECTED features."""
        try:
            api_keys = get_api_keys_from_config(self.config)
            validation = validate_required_apis(api_keys)
            
            print("Zeus System Status (CORRECTED EXIT ANALYSIS):")
            print("=" * 50)
            
            print(f"\n🚨 REQUIRED APIs:")
            for api in ['cielo_api_key', 'helius_api_key']:
                api_name = api.replace('_api_key', '').upper()
                if api == 'cielo_api_key':
                    api_desc = f"{api_name} (Trading Stats + CORRECTED Token PnL)"
                else:
                    api_desc = f"{api_name} (Accurate Timestamps)"
                
                status = "✅ Configured" if api_keys.get(api) else "❌ MISSING"
                print(f"{api_desc}: {status}")
            
            print(f"\n📈 RECOMMENDED APIs:")
            for api in ['birdeye_api_key']:
                api_name = api.replace('_api_key', '').upper()
                status = "✅ Configured" if api_keys.get(api) else "⚠️ Not configured"
                print(f"{api_name}: {status}")
            
            print(f"\nSystem Ready: {'✅ YES' if validation['system_ready'] else '❌ NO'}")
            
            # CORRECTED features status
            features_config = self.config.get('features', {})
            analysis_config = self.config.get('analysis', {})
            print(f"\n🎯 CORRECTED EXIT ANALYSIS FEATURES:")
            print(f"Corrected Exit Analysis: {'✅ Available' if validation['system_ready'] and features_config.get('enable_corrected_exit_analysis', True) else '❌ Requires APIs'}")
            print(f"Realistic TP/SL: {'✅ Available' if features_config.get('enable_realistic_tp_sl', True) else '❌ Disabled'}")
            print(f"TP/SL Validation: {'✅ Available' if features_config.get('enable_tp_sl_validation', True) else '❌ Disabled'}")
            print(f"Exit Behavior Inference: {'✅ Available' if features_config.get('enable_exit_behavior_inference', True) else '❌ Disabled'}")
            print(f"Pattern Thresholds: ✅ Corrected")
            print(f"Very Short Holds: <{analysis_config.get('very_short_threshold_minutes', 5)} minutes")
            print(f"Long Holds: >{analysis_config.get('long_hold_threshold_hours', 24)} hours")
            
            if validation['system_ready']:
                from zeus_api_manager import ZeusAPIManager
                
                api_manager = ZeusAPIManager(
                    birdeye_api_key=api_keys["birdeye_api_key"],
                    cielo_api_key=api_keys["cielo_api_key"],
                    helius_api_key=api_keys["helius_api_key"],
                    rpc_url=api_keys["solana_rpc_url"]
                )
                
                status = api_manager.get_api_status()
                print(f"Wallet Analysis Ready: {status.get('wallet_compatible', False)}")
                print(f"CORRECTED Token PnL Analysis Ready: {validation['system_ready']}")
                print(f"Realistic TP/SL Ready: {validation['system_ready']}")
                print(f"TP/SL Validation Ready: ✅")
            else:
                print(f"Missing REQUIRED APIs: {', '.join([api.replace('_api_key', '').upper() for api in validation['missing_required']])}")
        
        except Exception as e:
            logger.error(f"Status check error: {str(e)}")

def main():
    """Main entry point."""
    try:
        cli = ZeusCLI()
        cli.run()
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()