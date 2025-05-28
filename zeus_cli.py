#!/usr/bin/env python3
"""
Zeus - Standalone Wallet Analysis System
Main CLI Entry Point with Binary Decision System

Features:
- 30-day analysis window
- Minimum 6 unique token trades requirement
- Smart token sampling (5 → 10 if inconclusive)
- Binary decisions (Follow Wallet/Follow Sells)
- New scoring system with volume qualifier
- Bot-friendly CSV output
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
        "birdeye_api_key": "",
        "cielo_api_key": "",
        "helius_api_key": "",
        "solana_rpc_url": "https://api.mainnet-beta.solana.com",
        "analysis": {
            "days_to_analyze": 30,
            "min_unique_tokens": 6,
            "initial_token_sample": 5,
            "max_token_sample": 10,
            "composite_score_threshold": 65.0,
            "exit_quality_threshold": 70.0
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

class ZeusCLI:
    """Zeus CLI Application with binary decision system."""
    
    def __init__(self):
        self.config = load_config()
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            description="Zeus - Standalone Wallet Analysis System with Binary Decisions",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  zeus configure --cielo-api-key YOUR_KEY
  zeus analyze --wallets wallets.txt
  zeus analyze --wallet 7xG8...k9mP --output custom_analysis.csv
  zeus status
            """
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Commands")
        
        # Configure command
        configure_parser = subparsers.add_parser("configure", help="Configure API keys")
        configure_parser.add_argument("--birdeye-api-key", help="Birdeye API key")
        configure_parser.add_argument("--cielo-api-key", help="Cielo Finance API key")
        configure_parser.add_argument("--helius-api-key", help="Helius API key")
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
        """Interactive numbered menu."""
        print("\n" + "="*80, flush=True)
        print("ZEUS - Standalone Wallet Analysis System", flush=True)
        print("🎯 Binary Decision System with 30-Day Analysis", flush=True)
        print(f"📅 Current Date: {datetime.now().strftime('%Y-%m-%d')}", flush=True)
        print("="*80, flush=True)
        print("\nSelect an option:", flush=True)
        print("\n🔧 CONFIGURATION:", flush=True)
        print("1. Configure API Keys", flush=True)
        print("2. Check Configuration", flush=True)
        print("3. Test API Connectivity", flush=True)
        print("\n📊 ANALYSIS:", flush=True)
        print("4. Analyze Wallets (Binary Decisions)", flush=True)
        print("5. Single Wallet Analysis", flush=True)
        print("\n🔍 UTILITIES:", flush=True)
        print("6. System Status", flush=True)
        print("7. Help & Scoring Guide", flush=True)
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
        """Interactive API configuration."""
        print("\n" + "="*70, flush=True)
        print("    🔧 ZEUS API CONFIGURATION", flush=True)
        print("="*70, flush=True)
        
        # Ensure api_keys section exists
        if "api_keys" not in self.config:
            self.config["api_keys"] = {}
        
        api_keys = self.config["api_keys"]
        
        # Cielo Finance API (REQUIRED)
        print("\n💰 Cielo Finance API Key (REQUIRED for wallet analysis)")
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
        
        # Birdeye API (RECOMMENDED)
        print("\n🔍 Birdeye API Key (RECOMMENDED for token analysis)")
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
        
        # Helius API (OPTIONAL)
        print("\n🚀 Helius API Key (OPTIONAL for enhanced analysis)")
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
            new_key = input("Enter Helius API key (or Enter to skip): ").strip()
            if new_key:
                api_keys["helius_api_key"] = new_key
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
        print("\n✅ Configuration saved!")
        input("Press Enter to continue...")
    
    def _check_configuration(self):
        """Display current configuration."""
        print("\n" + "="*70, flush=True)
        print("    📋 ZEUS CONFIGURATION", flush=True)
        print("="*70, flush=True)
        
        print(f"\n🔑 API KEYS:")
        api_keys = self.config.get("api_keys", {})
        print(f"   Cielo Finance: {'✅ Configured' if api_keys.get('cielo_api_key') else '❌ Not configured'}")
        print(f"   Birdeye: {'✅ Configured' if api_keys.get('birdeye_api_key') else '⚠️ Not configured'}")
        print(f"   Helius: {'✅ Configured' if api_keys.get('helius_api_key') else '⚠️ Not configured'}")
        
        print(f"\n🌐 RPC ENDPOINT:")
        print(f"   URL: {api_keys.get('solana_rpc_url', 'https://api.mainnet-beta.solana.com')}")
        
        print(f"\n📊 ANALYSIS SETTINGS:")
        analysis_config = self.config.get('analysis', {})
        print(f"   Analysis Period: {analysis_config.get('days_to_analyze', 30)} days")
        print(f"   Min Unique Tokens: {analysis_config.get('min_unique_tokens', 6)}")
        print(f"   Initial Sample: {analysis_config.get('initial_token_sample', 5)} tokens")
        print(f"   Max Sample: {analysis_config.get('max_token_sample', 10)} tokens")
        print(f"   Score Threshold: {analysis_config.get('composite_score_threshold', 65.0)}")
        print(f"   Exit Quality Threshold: {analysis_config.get('exit_quality_threshold', 70.0)}")
        
        print(f"\n🎯 BINARY DECISION SYSTEM:")
        print(f"   Follow Wallet Threshold: ≥{analysis_config.get('composite_score_threshold', 65.0)} score")
        print(f"   Follow Sells Threshold: ≥{analysis_config.get('exit_quality_threshold', 70.0)}% exit quality")
        print(f"   Volume Qualifier: ≥{analysis_config.get('min_unique_tokens', 6)} unique tokens")
        
        # Check system readiness
        api_keys = self.config.get("api_keys", {})
        cielo_ok = bool(api_keys.get("cielo_api_key"))
        print(f"\n🎯 SYSTEM READINESS:")
        print(f"   Core Analysis: {'✅ Ready' if cielo_ok else '❌ Need Cielo API'}")
        print(f"   Enhanced Analysis: {'✅ Ready' if api_keys.get('birdeye_api_key') else '⚠️ Limited'}")
        
        input("\nPress Enter to continue...")
    
    def _test_api_connectivity(self):
        """Test API connectivity."""
        print("\n" + "="*70, flush=True)
        print("    🔍 API CONNECTIVITY TEST", flush=True)
        print("="*70, flush=True)
        
        try:
            from zeus_api_manager import ZeusAPIManager
            
            # Initialize - read API keys from config file
            api_keys = self.config.get("api_keys", {})
            api_manager = ZeusAPIManager(
                api_keys.get("birdeye_api_key", ""),
                api_keys.get("cielo_api_key", ""),
                api_keys.get("helius_api_key", ""),
                api_keys.get("solana_rpc_url", "https://api.mainnet-beta.solana.com")
            )
            
            status = api_manager.get_api_status()
            
            print(f"\n📊 API STATUS:")
            for api_name, api_status in status.get('api_status', {}).items():
                if api_status == "operational":
                    print(f"   {api_name}: ✅ Operational")
                elif api_status == "not_configured":
                    print(f"   {api_name}: ⚠️ Not configured")
                else:
                    print(f"   {api_name}: ❌ {api_status}")
            
            print(f"\n🎯 ZEUS COMPATIBILITY:")
            print(f"   Wallet Analysis: {'✅ Ready' if status.get('wallet_compatible', False) else '❌ Missing APIs'}")
            print(f"   Token Analysis: {'✅ Enhanced' if status.get('token_analysis_ready', False) else '⚠️ Limited'}")
            
        except Exception as e:
            print(f"❌ Error testing APIs: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _batch_analyze(self):
        """Batch wallet analysis."""
        print("\n" + "="*80, flush=True)
        print("    📊 ZEUS BATCH WALLET ANALYSIS", flush=True)
        print("    🎯 30-Day Analysis with Binary Decisions", flush=True)
        print("="*80, flush=True)
        
        # Load wallets
        wallets = load_wallets_from_file("wallets.txt")
        if not wallets:
            print("\n❌ No wallets found in wallets.txt")
            print("Add wallet addresses to wallets.txt (one per line)")
            input("Press Enter to continue...")
            return
        
        print(f"\n📁 Found {len(wallets)} wallets in wallets.txt")
        
        # Run analysis
        try:
            from zeus_analyzer import ZeusAnalyzer
            from zeus_api_manager import ZeusAPIManager
            
            # Initialize - get API keys from nested config structure
            api_config = self.config.get("api_keys", {})
            api_manager = ZeusAPIManager(
                api_config.get("birdeye_api_key", ""),
                api_config.get("cielo_api_key", ""),
                api_config.get("helius_api_key", ""),
                api_config.get("solana_rpc_url", "https://api.mainnet-beta.solana.com")
            )
            
            analyzer = ZeusAnalyzer(api_manager, self.config)
            
            print(f"\n🚀 Starting analysis...")
            print(f"   • Period: 30 days")
            print(f"   • Min tokens: 6 unique trades")
            print(f"   • Smart sampling: 5 → 10 tokens if needed")
            print(f"   • Binary decisions: Follow Wallet + Follow Sells")
            
            # Run batch analysis
            results = analyzer.analyze_wallets_batch(wallets)
            
            if results.get("success"):
                # Export results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = ensure_output_dir(f"zeus_analysis_{timestamp}.csv")
                
                from zeus_export import export_zeus_analysis
                export_success = export_zeus_analysis(results, output_file)
                
                if export_success:
                    print(f"\n✅ Analysis complete!")
                    print(f"📄 Results saved to: {output_file}")
                    
                    # Display summary
                    self._display_analysis_summary(results)
                else:
                    print(f"\n⚠️ Analysis completed but export failed")
            else:
                print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")
        
        except Exception as e:
            print(f"\n❌ Error during analysis: {str(e)}")
            logger.error(f"Batch analysis error: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _single_wallet_analyze(self):
        """Single wallet analysis."""
        print("\n" + "="*70, flush=True)
        print("    🔍 SINGLE WALLET ANALYSIS", flush=True)
        print("="*70, flush=True)
        
        wallet_address = input("\nEnter wallet address: ").strip()
        
        if not wallet_address or len(wallet_address) < 32:
            print("❌ Invalid wallet address")
            input("Press Enter to continue...")
            return
        
        try:
            from zeus_analyzer import ZeusAnalyzer
            from zeus_api_manager import ZeusAPIManager
            
            # Initialize - read API keys from config file
            api_keys = self.config.get("api_keys", {})
            api_manager = ZeusAPIManager(
                api_keys.get("birdeye_api_key", ""),
                api_keys.get("cielo_api_key", ""),
                api_keys.get("helius_api_key", ""),
                api_keys.get("solana_rpc_url", "https://api.mainnet-beta.solana.com")
            )
            
            analyzer = ZeusAnalyzer(api_manager, self.config)
            
            print(f"\n🔍 Analyzing {wallet_address[:8]}...{wallet_address[-4:]}")
            
            # Run single analysis
            result = analyzer.analyze_single_wallet(wallet_address)
            
            if result.get("success"):
                # Display detailed results
                self._display_single_wallet_result(result)
            else:
                print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            print(f"\n❌ Error during analysis: {str(e)}")
            logger.error(f"Single wallet analysis error: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _display_analysis_summary(self, results: Dict[str, Any]):
        """Display batch analysis summary."""
        total_analyzed = results.get('total_analyzed', 0)
        successful = results.get('successful_analyses', 0)
        failed = results.get('failed_analyses', 0)
        
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"   Total wallets: {total_analyzed}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        
        analyses = results.get('analyses', [])
        if analyses:
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
                    
                    print(f"   {i}. {wallet[:8]}...{wallet[-4:]}")
                    print(f"      Score: {score:.1f}/100")
                    print(f"      Follow: {'✅' if follow_wallet else '❌'} Wallet | {'✅' if follow_sells else '❌'} Sells")
    
    def _display_single_wallet_result(self, result: Dict[str, Any]):
        """Display single wallet analysis result."""
        print(f"\n" + "="*70)
        print(f"    📊 WALLET ANALYSIS RESULT")
        print(f"="*70)
        
        wallet = result['wallet_address']
        print(f"\n💰 Wallet: {wallet}")
        
        # Basic metrics
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"   Composite Score: {result.get('composite_score', 0):.1f}/100")
        print(f"   Unique Tokens: {result.get('unique_tokens_traded', 0)}")
        print(f"   Analysis Period: {result.get('analysis_days', 30)} days")
        print(f"   Tokens Analyzed: {result.get('tokens_analyzed', 0)}")
        
        # Binary decisions
        binary_decisions = result.get('binary_decisions', {})
        print(f"\n🎯 BINARY DECISIONS:")
        print(f"   Follow Wallet: {'✅ YES' if binary_decisions.get('follow_wallet') else '❌ NO'}")
        print(f"   Follow Sells: {'✅ YES' if binary_decisions.get('follow_sells') else '❌ NO'}")
        
        # Strategy recommendation
        strategy = result.get('strategy_recommendation', {})
        if strategy:
            print(f"\n📋 STRATEGY RECOMMENDATION:")
            print(f"   Copy Entries: {'✅ YES' if strategy.get('copy_entries') else '❌ NO'}")
            print(f"   Copy Exits: {'✅ YES' if strategy.get('copy_exits') else '❌ NO'}")
            print(f"   Take Profit 1: {strategy.get('tp1_percent', 0)}%")
            print(f"   Take Profit 2: {strategy.get('tp2_percent', 0)}%")
            print(f"   Take Profit 3: {strategy.get('tp3_percent', 0)}%")
            print(f"   Stop Loss: {strategy.get('stop_loss_percent', 0)}%")
            print(f"   Position Size: {strategy.get('position_size_sol', '1-10')} SOL")
            print(f"   Reasoning: {strategy.get('reasoning', 'N/A')}")
        
        # Scoring breakdown
        scoring = result.get('scoring_breakdown', {})
        if scoring:
            print(f"\n🔢 SCORING BREAKDOWN:")
            print(f"   Risk-Adjusted Performance: {scoring.get('risk_adjusted_score', 0):.1f}/30")
            print(f"   Distribution Quality: {scoring.get('distribution_score', 0):.1f}/25") 
            print(f"   Trading Discipline: {scoring.get('discipline_score', 0):.1f}/20")
            print(f"   Market Impact Awareness: {scoring.get('market_impact_score', 0):.1f}/15")
            print(f"   Consistency & Reliability: {scoring.get('consistency_score', 0):.1f}/10")
    
    def _system_status(self):
        """Display system status."""
        print("\n" + "="*70, flush=True)
        print("    ⚡ ZEUS SYSTEM STATUS", flush=True)
        print("="*70, flush=True)
        
        try:
            from zeus_api_manager import ZeusAPIManager
            
            # Read API keys from config file
            api_keys = self.config.get("api_keys", {})
            api_manager = ZeusAPIManager(
                api_keys.get("birdeye_api_key", ""),
                api_keys.get("cielo_api_key", ""),
                api_keys.get("helius_api_key", ""),
                api_keys.get("solana_rpc_url", "https://api.mainnet-beta.solana.com")
            )
            
            status = api_manager.get_api_status()
            perf_stats = api_manager.get_performance_stats()
            
            print(f"\n🔌 API STATUS:")
            for api_name, api_status in status.get('api_status', {}).items():
                print(f"   {api_name}: {api_status}")
            
            print(f"\n📊 PERFORMANCE STATS:")
            for api_name, stats in perf_stats.items():
                if isinstance(stats, dict) and 'total_calls' in stats:
                    print(f"   {api_name}: {stats['total_calls']} calls, {stats['success_rate_percent']:.1f}% success")
            
            print(f"\n🎯 SYSTEM CAPABILITIES:")
            print(f"   30-Day Analysis: ✅ Ready")
            print(f"   Binary Decisions: ✅ Ready") 
            print(f"   Smart Token Sampling: ✅ Ready")
            print(f"   Volume Qualifier: ✅ Ready (≥6 tokens)")
            print(f"   TP/SL Strategy Matrix: ✅ Ready")
            
        except Exception as e:
            print(f"❌ Error checking system status: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _show_help(self):
        """Show help and scoring guide."""
        print("\n" + "="*80, flush=True)
        print("    📖 ZEUS HELP & SCORING GUIDE", flush=True)
        print("="*80, flush=True)
        
        print(f"\n🎯 ZEUS OVERVIEW:")
        print(f"Zeus is a standalone wallet analysis system focused on binary decisions")
        print(f"for automated trading bots. It analyzes wallets over a 30-day period")
        print(f"and provides clear YES/NO decisions on whether to follow their trades.")
        
        print(f"\n📊 ANALYSIS PROCESS:")
        print(f"1. Check minimum 6 unique token trades in 30 days")
        print(f"2. Analyze 5 most recent token trades")
        print(f"3. If decision is unclear, expand to 10 tokens")
        print(f"4. Calculate composite score (0-100)")
        print(f"5. Generate binary decisions and TP/SL strategy")
        
        print(f"\n🔢 SCORING SYSTEM (0-100 points):")
        print(f"   • Risk-Adjusted Performance (30%)")
        print(f"     - Median ROI, Standard Deviation, Volume Qualifier")
        print(f"   • Distribution Quality (25%)")
        print(f"     - Moonshot rate, Big wins, Loss management")
        print(f"   • Trading Discipline (20%)")
        print(f"     - Loss cutting, Exit behavior, Flipper detection")
        print(f"   • Market Impact Awareness (15%)")
        print(f"     - Bet sizing, Market cap strategy")
        print(f"   • Consistency & Reliability (10%)")
        print(f"     - Activity pattern, Red flag detection")
        
        print(f"\n🎯 BINARY DECISIONS:")
        print(f"   Follow Wallet: ≥65 score + ≥6 tokens + No bot behavior")
        print(f"   Follow Sells: ≥70% exit quality + Low dump rate")
        
        print(f"\n📋 STRATEGY MATRIX:")
        print(f"   Mirror Strategy (Follow Sells = YES):")
        print(f"   - Copy their exits with safety buffer")
        print(f"   Custom Strategy (Follow Sells = NO):")
        print(f"   - Use fixed TP/SL based on trader pattern")
        
        print(f"\n⚡ COMMAND EXAMPLES:")
        print(f"   zeus configure --cielo-api-key YOUR_KEY")
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
            logger.info("Cielo Finance API key configured")
        
        if args.helius_api_key:
            api_keys["helius_api_key"] = args.helius_api_key
            logger.info("Helius API key configured")
        
        if args.rpc_url:
            api_keys["solana_rpc_url"] = args.rpc_url
            logger.info("RPC URL configured")
        
        save_config(self.config)
        logger.info("Configuration saved")
    
    def _handle_analyze_command(self, args):
        """Handle analyze command."""
        try:
            from zeus_analyzer import ZeusAnalyzer
            from zeus_api_manager import ZeusAPIManager
            from zeus_export import export_zeus_analysis
            
            # Initialize - read API keys from config file
            api_keys = self.config.get("api_keys", {})
            api_manager = ZeusAPIManager(
                api_keys.get("birdeye_api_key", ""),
                api_keys.get("cielo_api_key", ""),
                api_keys.get("helius_api_key", ""),
                api_keys.get("solana_rpc_url", "https://api.mainnet-beta.solana.com")
            )
            
            analyzer = ZeusAnalyzer(api_manager, self.config)
            
            if args.wallet:
                # Single wallet analysis
                logger.info(f"Analyzing single wallet: {args.wallet}")
                result = analyzer.analyze_single_wallet(args.wallet)
                
                if result.get("success"):
                    output_file = args.output or f"zeus_single_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    output_file = ensure_output_dir(output_file)
                    export_zeus_analysis({"analyses": [result]}, output_file)
                    logger.info(f"Results saved to {output_file}")
                else:
                    logger.error(f"Analysis failed: {result.get('error')}")
            
            elif args.wallets:
                # Batch analysis
                wallets = load_wallets_from_file(args.wallets)
                if not wallets:
                    logger.error(f"No wallets found in {args.wallets}")
                    return
                
                logger.info(f"Analyzing {len(wallets)} wallets")
                results = analyzer.analyze_wallets_batch(wallets)
                
                if results.get("success"):
                    output_file = args.output or f"zeus_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    output_file = ensure_output_dir(output_file)
                    export_zeus_analysis(results, output_file)
                    logger.info(f"Results saved to {output_file}")
                else:
                    logger.error(f"Batch analysis failed: {results.get('error')}")
            
            else:
                logger.error("Either --wallet or --wallets must be specified")
        
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
    
    def _handle_status_command(self, args):
        """Handle status command."""
        try:
            from zeus_api_manager import ZeusAPIManager
            
            # Read API keys from config file  
            api_keys = self.config.get("api_keys", {})
            api_manager = ZeusAPIManager(
                api_keys.get("birdeye_api_key", ""),
                api_keys.get("cielo_api_key", ""),
                api_keys.get("helius_api_key", ""),
                api_keys.get("solana_rpc_url", "https://api.mainnet-beta.solana.com")
            )
            
            status = api_manager.get_api_status()
            
            print("Zeus System Status:")
            print("=" * 50)
            for api_name, api_status in status.get('api_status', {}).items():
                print(f"{api_name}: {api_status}")
            
            print(f"\nWallet Analysis Ready: {status.get('wallet_compatible', False)}")
            print(f"Token Analysis Ready: {status.get('token_analysis_ready', False)}")
        
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