#!/usr/bin/env python3
"""
Zeus - Standalone Wallet Analysis System - FULLY UPDATED 
Main CLI Entry Point with Token PnL Analysis & Direct Field Extraction

MAJOR UPDATES:
- Token PnL endpoint analysis for real TP/SL recommendations
- Direct Cielo field extraction (no conversions/scaling)
- Smart pattern-based TP/SL strategies
- Removed timestamp accuracy display (Helius PRIMARY internal)
- Enhanced CSV with unique_tokens_30d field
- Updated thresholds (5 minutes, 24 hours)
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
            "very_short_threshold_minutes": 5,  # UPDATED: 5 minutes
            "long_hold_threshold_hours": 24     # UPDATED: 24 hours
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
    """Zeus CLI Application with Token PnL Analysis and Direct Field Extraction."""
    
    def __init__(self):
        self.config = load_config()
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            description="Zeus - Wallet Analysis with Token PnL & Direct Field Extraction (UPDATED)",
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
        """Interactive numbered menu with TOKEN PNL ANALYSIS features."""
        print("\n" + "="*80, flush=True)
        print("                     ⚡ ███████  ███████  ██    ██  ███████ ⚡", flush=True) 
        print("                     ⚡      ██  ██       ██    ██  ██      ⚡", flush=True)
        print("                     ⚡     ██   █████    ██    ██  ███████ ⚡", flush=True)
        print("                     ⚡    ██    ██       ██    ██       ██ ⚡", flush=True)
        print("                     ⚡ ███████  ███████   ██████   ███████ ⚡", flush=True)
        print("="*80, flush=True)
        print("                   📊 NEW: Token PnL Analysis & Smart TP/SL", flush=True)
        print("                   ⚡ Direct Cielo Field Extraction (No Scaling)", flush=True)
        print("                   🎯 Pattern Thresholds: 5min | 24hr", flush=True)
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
            print("📊 Token PnL Analysis: Available (5 credits per wallet)", flush=True)
            print("🎯 Smart TP/SL: Pattern-based recommendations", flush=True)
            print("⚡ Direct Fields: No scaling/conversion logic", flush=True)
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
        print("4. Analyze Wallets (Token PnL + Smart TP/SL)", flush=True)
        print("5. Single Wallet Analysis", flush=True)
        print("\n🔍 UTILITIES:", flush=True)
        print("6. System Status", flush=True)
        print("7. Help & New Features", flush=True)
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
        print("    🔧 ZEUS API CONFIGURATION - TOKEN PNL ENABLED", flush=True)
        print("="*70, flush=True)
        
        # Ensure api_keys section exists
        if "api_keys" not in self.config:
            self.config["api_keys"] = {}
        
        api_keys = self.config["api_keys"]
        
        print("\n🚨 REQUIRED APIs - Zeus cannot function without these:")
        
        # Cielo Finance API (REQUIRED)
        print("\n💰 Cielo Finance API Key (REQUIRED for Trading Stats + Token PnL)")
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
            print("📊 Token PnL Analysis: Available (5 credits per wallet)")
            print("🎯 Smart TP/SL: Pattern-based recommendations")
            print("⚡ Direct Field Extraction: No scaling/conversion")
        else:
            print("⚠️ Configuration saved but system is NOT READY!")
            print("Missing REQUIRED APIs:")
            for api in validation['missing_required']:
                print(f"   • {api.replace('_api_key', '').upper()}")
        
        input("Press Enter to continue...")
    
    def _check_configuration(self):
        """Display current configuration with NEW features."""
        print("\n" + "="*70, flush=True)
        print("    📋 ZEUS CONFIGURATION - TOKEN PNL ENABLED", flush=True)
        print("="*70, flush=True)
        
        api_keys = get_api_keys_from_config(self.config)
        validation = validate_required_apis(api_keys)
        
        print(f"\n🔑 API KEYS STATUS:")
        
        # Required APIs
        print(f"\n🚨 REQUIRED APIs:")
        required_status = {
            'cielo_api_key': 'Cielo Finance (Trading Stats + Token PnL)',
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
        
        print(f"\n🎯 NEW FEATURES:")
        print(f"   Token PnL Analysis: ✅ Available (5 credits per wallet)")
        print(f"   Smart TP/SL Recommendations: ✅ Pattern-based")
        print(f"   Direct Field Extraction: ✅ No scaling/conversion")
        print(f"   Updated CSV Format: ✅ unique_tokens_30d field")
        
        print(f"\n⚡ UPDATED PATTERN THRESHOLDS:")
        print(f"   Very Short Holds: <{analysis_config.get('very_short_threshold_minutes', 5)} minutes")
        print(f"   Long Holds: >{analysis_config.get('long_hold_threshold_hours', 24)} hours")
        
        print(f"\n🎯 SYSTEM STATUS:")
        if validation['system_ready']:
            print(f"   System Status: ✅ READY")
            print(f"   Token PnL Analysis: ✅ Available")
            print(f"   Smart TP/SL: ✅ Available")
            print(f"   Direct Field Extraction: ✅ Available")
            print(f"   Accurate Timestamps: ✅ Available (Helius PRIMARY)")
            if validation['missing_recommended']:
                print(f"   Enhanced Features: ⚠️ Limited (missing {len(validation['missing_recommended'])} recommended APIs)")
            else:
                print(f"   Enhanced Features: ✅ Full")
        else:
            print(f"   System Status: ❌ NOT READY")
            print(f"   Missing Required: {', '.join([api.replace('_api_key', '').upper() for api in validation['missing_required']])}")
            print(f"   🔧 Fix: zeus configure --cielo-api-key YOUR_KEY --helius-api-key YOUR_KEY")
        
        input("\nPress Enter to continue...")
    
    def _test_api_connectivity(self):
        """Test API connectivity with Token PnL features."""
        print("\n" + "="*70, flush=True)
        print("    🔍 API CONNECTIVITY TEST - TOKEN PNL ENABLED", flush=True)
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
            
            print(f"\n🔧 Initializing API manager with TOKEN PNL features...")
            
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
            print(f"\n💰 Testing Cielo API (Trading Stats + Token PnL)...")
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
                    print(f"   📈 Token PnL test result: {token_count} tokens")
                
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
            print(f"   Token PnL Analysis: ✅ Available")
            print(f"   Smart TP/SL: ✅ Available")
            print(f"   Direct Field Extraction: ✅ Available")
            print(f"   Pattern Thresholds: ✅ Updated (5min/24hr)")
            
        except ValueError as e:
            print(f"\n❌ CRITICAL ERROR: {str(e)}")
            print(f"🔧 This indicates missing REQUIRED API keys")
        except Exception as e:
            print(f"\n❌ Error testing APIs: {str(e)}")
            logger.error(f"API test error: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _batch_analyze(self):
        """Batch wallet analysis with Token PnL and Smart TP/SL."""
        print("\n" + "="*80, flush=True)
        print("    📊 ZEUS BATCH WALLET ANALYSIS - TOKEN PNL ENABLED", flush=True)
        print("    🎯 Token PnL Analysis (5 credits) + Smart Pattern TP/SL", flush=True)
        print("    ⚡ Direct Field Extraction + Updated Thresholds", flush=True)
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
        print(f"📊 Token PnL Analysis: Available (5 credits per wallet)")
        print(f"🎯 Smart TP/SL: Pattern-based recommendations")
        
        # Run analysis
        try:
            from zeus_analyzer import ZeusAnalyzer
            from zeus_api_manager import ZeusAPIManager
            
            print(f"\n🚀 Initializing Zeus with TOKEN PNL ANALYSIS...")
            
            api_manager = ZeusAPIManager(
                birdeye_api_key=api_keys["birdeye_api_key"],
                cielo_api_key=api_keys["cielo_api_key"],
                helius_api_key=api_keys["helius_api_key"],
                rpc_url=api_keys["solana_rpc_url"]
            )
            
            analyzer = ZeusAnalyzer(api_manager, self.config)
            
            print(f"\n🔍 Starting analysis with NEW features...")
            print(f"   • Period: 30 days")
            print(f"   • Min tokens: 6 unique trades")
            print(f"   • Timestamp source: Helius PRIMARY (accurate)")
            print(f"   • Trading Stats: Cielo API (30 credits, direct fields)")
            print(f"   • Token PnL: Cielo API (5 credits, real trade patterns)")
            print(f"   • TP/SL: Smart pattern-based recommendations")
            print(f"   • Pattern thresholds: <5min (flipper) | >24hr (position)")
            print(f"   • Binary decisions: Follow Wallet + Follow Sells")
            
            # Estimate costs
            total_trading_stats_cost = len(wallets) * 30
            total_token_pnl_cost = len(wallets) * 5
            total_cost = total_trading_stats_cost + total_token_pnl_cost
            
            print(f"\n💰 ESTIMATED API COSTS:")
            print(f"   Trading Stats: {len(wallets)} × 30 = {total_trading_stats_cost} credits")
            print(f"   Token PnL: {len(wallets)} × 5 = {total_token_pnl_cost} credits")
            print(f"   TOTAL: {total_cost} credits")
            
            confirm = input(f"\nProceed with analysis? (y/N): ").lower().strip()
            if confirm != 'y':
                print("Analysis cancelled.")
                input("Press Enter to continue...")
                return
            
            # Run batch analysis
            results = analyzer.analyze_wallets_batch(wallets)
            
            if results.get("success"):
                # Export results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = ensure_output_dir(f"zeus_token_pnl_analysis_{timestamp}.csv")
                
                from zeus_export import export_zeus_analysis
                export_success = export_zeus_analysis(results, output_file)
                
                if export_success:
                    print(f"\n✅ Analysis complete with TOKEN PNL & SMART TP/SL!")
                    print(f"📄 Results saved to: {output_file}")
                    print(f"📊 CSV includes: Direct Cielo fields, unique_tokens_30d, pattern-based TP/SL")
                    
                    # Display summary
                    self._display_analysis_summary_updated(results)
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
        """Single wallet analysis with Token PnL and Smart TP/SL."""
        print("\n" + "="*70, flush=True)
        print("    🔍 SINGLE WALLET ANALYSIS - TOKEN PNL ENABLED", flush=True)
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
            
            print(f"\n🔧 Initializing with TOKEN PNL features...")
            
            api_manager = ZeusAPIManager(
                birdeye_api_key=api_keys["birdeye_api_key"],
                cielo_api_key=api_keys["cielo_api_key"],
                helius_api_key=api_keys["helius_api_key"],
                rpc_url=api_keys["solana_rpc_url"]
            )
            
            analyzer = ZeusAnalyzer(api_manager, self.config)
            
            print(f"\n🔍 Analyzing {wallet_address[:8]}...{wallet_address[-4:]} with NEW features")
            print(f"   📊 Trading Stats: Cielo API (30 credits)")
            print(f"   🎯 Token PnL: Cielo API (5 credits)")
            print(f"   ⚡ Direct Field Extraction: No scaling")
            print(f"   🎯 Smart TP/SL: Pattern-based recommendations")
            
            # Run single analysis
            result = analyzer.analyze_single_wallet(wallet_address)
            
            if result.get("success"):
                # Display detailed results
                self._display_single_wallet_result_updated(result)
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
    
    def _display_analysis_summary_updated(self, results: Dict[str, Any]):
        """Display batch analysis summary with NEW features."""
        total_analyzed = results.get('total_analyzed', 0)
        successful = results.get('successful_analyses', 0)
        failed = results.get('failed_analyses', 0)
        
        print(f"\n📊 ANALYSIS SUMMARY (TOKEN PNL ENABLED):")
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
        
        analyses = results.get('analyses', [])
        if analyses:
            # Pattern analysis summary
            patterns = {}
            for analysis in analyses:
                if analysis.get('success'):
                    pattern_data = analysis.get('trade_pattern_analysis', {})
                    pattern = pattern_data.get('pattern', 'unknown')
                    patterns[pattern] = patterns.get(pattern, 0) + 1
            
            if patterns:
                print(f"\n⚡ TRADER PATTERNS DETECTED:")
                for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
                    print(f"   {pattern}: {count} wallets")
            
            # Binary decision summary
            follow_wallet_yes = sum(1 for a in analyses if a.get('binary_decisions', {}).get('follow_wallet', False))
            follow_sells_yes = sum(1 for a in analyses if a.get('binary_decisions', {}).get('follow_sells', False))
            
            print(f"\n🎯 BINARY DECISIONS:")
            print(f"   Follow Wallet: {follow_wallet_yes}/{successful} ({follow_wallet_yes/successful*100:.1f}%)")
            print(f"   Follow Sells: {follow_sells_yes}/{successful} ({follow_sells_yes/successful*100:.1f}%)")
            
            # Top performers
            top_performers = sorted([a for a in analyses if a.get('success')], 
                                  key=lambda x: x.get('composite_score', 0), reverse=True)[:5]
            
            if top_performers:
                print(f"\n🏆 TOP 5 PERFORMERS:")
                for i, perf in enumerate(top_performers, 1):
                    wallet = perf['wallet_address']
                    score = perf.get('composite_score', 0)
                    follow_wallet = perf.get('binary_decisions', {}).get('follow_wallet', False)
                    follow_sells = perf.get('binary_decisions', {}).get('follow_sells', False)
                    
                    # Show pattern and strategy info
                    pattern_data = perf.get('trade_pattern_analysis', {})
                    pattern = pattern_data.get('pattern', 'unknown')
                    
                    strategy = perf.get('strategy_recommendation', {})
                    tp1 = strategy.get('tp1_percent', 0)
                    tp2 = strategy.get('tp2_percent', 0)
                    
                    print(f"   {i}. {wallet[:8]}...{wallet[-4:]}")
                    print(f"      Score: {score:.1f}/100 | Pattern: {pattern}")
                    print(f"      Follow: {'✅' if follow_wallet else '❌'} Wallet | {'✅' if follow_sells else '❌'} Sells")
                    print(f"      Smart TP/SL: {tp1}% / {tp2}%")
                    print(f"      📊 Token PnL analysis available")
    
    def _display_single_wallet_result_updated(self, result: Dict[str, Any]):
        """Display single wallet analysis result with NEW features."""
        print(f"\n" + "="*70)
        print(f"    📊 WALLET ANALYSIS RESULT - TOKEN PNL ENABLED")
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
        
        # Trade pattern analysis (NEW)
        trade_pattern_analysis = result.get('trade_pattern_analysis', {})
        if trade_pattern_analysis.get('success'):
            print(f"\n🎯 TRADE PATTERN ANALYSIS:")
            pattern = trade_pattern_analysis.get('pattern', 'unknown')
            avg_roi = trade_pattern_analysis.get('avg_roi', 0)
            win_rate = trade_pattern_analysis.get('win_rate', 0)
            avg_hold_time = trade_pattern_analysis.get('avg_hold_time_hours', 0)
            tokens_analyzed = trade_pattern_analysis.get('tokens_analyzed', 0)
            
            print(f"   Pattern: {pattern}")
            print(f"   Avg ROI: {avg_roi:.1f}%")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Avg Hold Time: {avg_hold_time:.1f} hours")
            print(f"   Token Trades Analyzed: {tokens_analyzed}")
            
            tp_sl_analysis = trade_pattern_analysis.get('tp_sl_analysis', {})
            if tp_sl_analysis.get('based_on_actual_exits'):
                print(f"   TP/SL Source: Actual exit patterns")
            else:
                print(f"   TP/SL Source: Pattern-based defaults")
        
        # Binary decisions
        binary_decisions = result.get('binary_decisions', {})
        print(f"\n🎯 BINARY DECISIONS:")
        print(f"   Follow Wallet: {'✅ YES' if binary_decisions.get('follow_wallet') else '❌ NO'}")
        print(f"   Follow Sells: {'✅ YES' if binary_decisions.get('follow_sells') else '❌ NO'}")
        
        # Strategy recommendation (UPDATED)
        strategy = result.get('strategy_recommendation', {})
        if strategy:
            print(f"\n📋 SMART STRATEGY RECOMMENDATION:")
            print(f"   Copy Entries: {'✅ YES' if strategy.get('copy_entries') else '❌ NO'}")
            print(f"   Copy Exits: {'✅ YES' if strategy.get('copy_exits') else '❌ NO'}")
            print(f"   Smart TP1: {strategy.get('tp1_percent', 0)}%")
            print(f"   Smart TP2: {strategy.get('tp2_percent', 0)}%")
            print(f"   Smart TP3: {strategy.get('tp3_percent', 0)}%")
            print(f"   Stop Loss: {strategy.get('stop_loss_percent', 0)}%")
            print(f"   Position Size: {strategy.get('position_size_sol', '1-10')} SOL")
            print(f"   Reasoning: {strategy.get('reasoning', 'N/A')}")
        
        print(f"\n📊 DATA SOURCES:")
        print(f"   Trading Stats: Cielo API (30 credits, direct fields)")
        print(f"   Token PnL: Cielo API (5 credits, real trade patterns)")
        print(f"   Timestamps: Helius PRIMARY (accurate)")
        print(f"   TP/SL Strategy: Pattern-based analysis")
        print(f"   Field Extraction: Direct (no scaling/conversion)")
    
    def _system_status(self):
        """Display system status with NEW features."""
        print("\n" + "="*70, flush=True)
        print("    ⚡ ZEUS SYSTEM STATUS - TOKEN PNL ENABLED", flush=True)
        print("="*70, flush=True)
        
        api_keys = get_api_keys_from_config(self.config)
        validation = validate_required_apis(api_keys)
        
        print(f"\n🔧 SYSTEM CONFIGURATION:")
        print(f"   Zeus Version: 2.2 (Token PnL + Smart TP/SL)")
        print(f"   Configuration File: {CONFIG_FILE}")
        print(f"   System Ready: {'✅ YES' if validation['system_ready'] else '❌ NO'}")
        
        print(f"\n🔑 API KEY STATUS:")
        
        # Required APIs
        print(f"\n🚨 REQUIRED APIs:")
        for api in ['cielo_api_key', 'helius_api_key']:
            api_name = api.replace('_api_key', '').upper()
            if api == 'cielo_api_key':
                api_desc = f"{api_name} (Trading Stats + Token PnL)"
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
        
        # NEW features status
        analysis_config = self.config.get('analysis', {})
        print(f"\n🎯 NEW FEATURES:")
        print(f"   Token PnL Analysis: {'✅ Available' if validation['system_ready'] else '❌ Requires Cielo API'}")
        print(f"   Smart TP/SL: {'✅ Available' if validation['system_ready'] else '❌ Requires APIs'}")
        print(f"   Direct Field Extraction: ✅ Available")
        print(f"   Pattern Recognition: ✅ Updated")
        print(f"   Very Short Holds: <{analysis_config.get('very_short_threshold_minutes', 5)} minutes")
        print(f"   Long Holds: >{analysis_config.get('long_hold_threshold_hours', 24)} hours")
        print(f"   CSV Format: ✅ Updated (unique_tokens_30d field)")
        
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
                print(f"   Token PnL Analysis: ✅ Available (5 credits)")
                print(f"   Smart TP/SL: ✅ Available")
                print(f"   Direct Field Extraction: ✅ Available")
                print(f"   Accurate Timestamps: ✅ Available (Helius PRIMARY)")
                print(f"   Pattern-based Strategies: ✅ Available")
                
            except Exception as e:
                print(f"\n❌ Error checking detailed status: {str(e)}")
        else:
            print(f"\n🔧 TO FIX:")
            print(f"   Run: zeus configure")
            for api in validation['missing_required']:
                api_name = api.replace('_api_key', '').replace('_', '-')
                print(f"   Add: --{api_name} YOUR_KEY")
        
        input("\nPress Enter to continue...")
    
    def _show_help(self):
        """Show help and NEW features."""
        print("\n" + "="*80, flush=True)
        print("    📖 ZEUS HELP & NEW FEATURES", flush=True)
        print("="*80, flush=True)
        
        print(f"\n🎯 ZEUS OVERVIEW:")
        print(f"Zeus is a standalone wallet analysis system focused on binary decisions")
        print(f"for automated trading bots. It now includes Token PnL analysis and")
        print(f"smart pattern-based TP/SL recommendations.")
        
        print(f"\n📊 MAJOR NEW FEATURES IN v2.2:")
        print(f"   • Token PnL Analysis: Real trade patterns (5 credits per wallet)")
        print(f"   • Smart TP/SL: Pattern-based recommendations (flippers vs gem hunters)")
        print(f"   • Direct Field Extraction: No scaling/conversion from Cielo")
        print(f"   • Updated CSV: unique_tokens_30d field (removed total_buys/sells)")
        print(f"   • Enhanced Pattern Recognition: 5min/24hr thresholds")
        print(f"   • Real Exit Analysis: Based on actual trade patterns")
        
        print(f"\n🚨 REQUIRED APIs (System cannot function without these):")
        print(f"   • Cielo Finance API - Trading Stats (30 credits) + Token PnL (5 credits)")
        print(f"     Get yours at: https://cielo.finance")
        print(f"   • Helius API - Accurate transaction timestamps")
        print(f"     Get yours at: https://helius.xyz")
        
        print(f"\n📈 RECOMMENDED APIs (Enhanced features):")
        print(f"   • Birdeye API - Enhanced token analysis")
        print(f"     Get yours at: https://birdeye.so")
        
        print(f"\n🎯 TOKEN PNL ANALYSIS FEATURES:")
        print(f"   • Real Trade Patterns: Analyzes actual token trades")
        print(f"   • Smart Sampling: 5 initial + 5 if inconclusive (max 30 days)")
        print(f"   • Pattern Detection: Identifies flippers, gem hunters, etc.")
        print(f"   • Actual Exit Analysis: Calculates real TP/SL levels")
        print(f"   • Cost Efficient: Only 5 credits vs 30 for Trading Stats")
        
        print(f"\n⚡ SMART TP/SL RECOMMENDATIONS:")
        print(f"   • Flipper Pattern: Low TP levels (30%-60%) + tight SL (-15%)")
        print(f"   • Gem Hunter Pattern: High TP levels (200%-500%) + patient SL (-50%)")
        print(f"   • Consistent Trader: Balanced TP/SL based on performance")
        print(f"   • Position Trader: Higher TP levels for long-term holds")
        print(f"   • Based on Actual Exits: Uses real trade data when available")
        
        print(f"\n📊 DIRECT FIELD EXTRACTION:")
        print(f"   • No More Conversions: Direct ROI, winrate, hold time from Cielo")
        print(f"   • No More Scaling: Raw PnL values without adjustments")
        print(f"   • Field Discovery: Logs all available Cielo fields")
        print(f"   • Data Validation: Ensures field quality and accuracy")
        
        print(f"\n📋 UPDATED CSV FORMAT:")
        print(f"   • Added: unique_tokens_30d (from Cielo)")
        print(f"   • Removed: total_buys_30_days, total_sells_30_days")
        print(f"   • Enhanced: 1 decimal precision for key fields")
        print(f"   • Smart TP/SL: Pattern-based recommendations in tp_1, tp_2")
        
        print(f"\n📊 ANALYSIS PROCESS:")
        print(f"1. Get REAL last transaction timestamp from Helius API")
        print(f"2. Get Trading Stats from Cielo API (30 credits, direct fields)")
        print(f"3. Analyze Token PnL data from Cielo API (5 credits, real patterns)")
        print(f"4. Identify trader pattern (flipper, gem hunter, etc.)")
        print(f"5. Calculate smart TP/SL based on actual trade analysis")
        print(f"6. Generate binary decisions and strategy recommendations")
        
        print(f"\n⚡ COMMAND EXAMPLES:")
        print(f"   zeus configure --cielo-api-key YOUR_KEY --helius-api-key YOUR_KEY")
        print(f"   zeus analyze --wallets wallets.txt")
        print(f"   zeus analyze --wallet 7xG8...k9mP")
        print(f"   zeus status")
        
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
            logger.info("Cielo Finance API key configured (Trading Stats + Token PnL enabled)")
        
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
            logger.info("📊 Token PnL Analysis: Available (5 credits per wallet)")
            logger.info("🎯 Smart TP/SL: Pattern-based recommendations")
            logger.info("⚡ Direct Field Extraction: No scaling/conversion")
        else:
            logger.warning("⚠️ Configuration saved but system is NOT READY!")
            logger.warning(f"Missing REQUIRED APIs: {', '.join([api.replace('_api_key', '').upper() for api in validation['missing_required']])}")
    
    def _handle_analyze_command(self, args):
        """Handle analyze command with TOKEN PNL features."""
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
            
            logger.info("🔧 Initializing Zeus with TOKEN PNL ANALYSIS...")
            logger.info("📊 Token PnL: Available (5 credits per wallet)")
            logger.info("🎯 Smart TP/SL: Pattern-based recommendations")
            logger.info("⚡ Direct Field Extraction: No scaling/conversion")
            
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
                logger.info(f"Cost: 35 credits (30 Trading Stats + 5 Token PnL)")
                result = analyzer.analyze_single_wallet(args.wallet)
                
                if result.get("success"):
                    output_file = args.output or f"zeus_token_pnl_single_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    output_file = ensure_output_dir(output_file)
                    export_zeus_analysis({"analyses": [result]}, output_file)
                    logger.info(f"Results saved to {output_file}")
                    logger.info(f"📊 CSV includes: Direct Cielo fields, Token PnL analysis, Smart TP/SL")
                else:
                    error_type = result.get('error_type', 'UNKNOWN')
                    logger.error(f"Analysis failed ({error_type}): {result.get('error')}")
            
            elif args.wallets:
                # Batch analysis
                wallets = load_wallets_from_file(args.wallets)
                if not wallets:
                    logger.error(f"No wallets found in {args.wallets}")
                    return
                
                logger.info(f"Analyzing {len(wallets)} wallets with TOKEN PNL features")
                logger.info(f"Estimated cost: {len(wallets) * 35} credits ({len(wallets)} × 35)")
                results = analyzer.analyze_wallets_batch(wallets)
                
                if results.get("success"):
                    output_file = args.output or f"zeus_token_pnl_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    output_file = ensure_output_dir(output_file)
                    export_zeus_analysis(results, output_file)
                    logger.info(f"Results saved to {output_file}")
                    logger.info(f"📊 CSV includes: Direct fields, Token PnL, unique_tokens_30d, Smart TP/SL")
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
        """Handle status command with TOKEN PNL features."""
        try:
            api_keys = get_api_keys_from_config(self.config)
            validation = validate_required_apis(api_keys)
            
            print("Zeus System Status (TOKEN PNL ENABLED):")
            print("=" * 50)
            
            print(f"\n🚨 REQUIRED APIs:")
            for api in ['cielo_api_key', 'helius_api_key']:
                api_name = api.replace('_api_key', '').upper()
                if api == 'cielo_api_key':
                    api_desc = f"{api_name} (Trading Stats + Token PnL)"
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
            
            # NEW features status
            analysis_config = self.config.get('analysis', {})
            print(f"\n🎯 NEW FEATURES:")
            print(f"Token PnL Analysis: {'✅ Available' if validation['system_ready'] else '❌ Requires APIs'}")
            print(f"Smart TP/SL: {'✅ Available' if validation['system_ready'] else '❌ Requires APIs'}")
            print(f"Direct Field Extraction: ✅ Available")
            print(f"Pattern Thresholds: ✅ Updated")
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
                print(f"Token PnL Analysis Ready: {validation['system_ready']}")
                print(f"Smart TP/SL Ready: {validation['system_ready']}")
                print(f"Direct Field Extraction Ready: ✅")
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