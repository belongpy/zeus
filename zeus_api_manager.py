"""
Zeus API Manager - Enhanced for Corrected Exit Analysis with Performance Monitoring
ENHANCEMENTS:
- Enhanced Token PnL data extraction for better exit analysis
- Improved transaction data processing for exit pattern inference
- Added better logging for exit analysis debugging
- Performance monitoring for API costs and response times
- Real-time cost tracking and alerts
- Response time optimization monitoring
- Preserved all existing core functionality
"""

import logging
import time
import requests
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import dateutil.parser

logger = logging.getLogger("zeus.api_manager")

class ZeusAPIManager:
    """Zeus API manager with ENHANCED Token PnL analysis for corrected exit analysis and performance monitoring."""
    
    def __init__(self, birdeye_api_key: str = "", cielo_api_key: str = "", 
                 helius_api_key: str = "", rpc_url: str = "https://api.mainnet-beta.solana.com",
                 performance_monitor=None):
        """Initialize with REQUIRED API key validation and performance monitoring."""
        
        # Store API keys
        self.birdeye_api_key = birdeye_api_key.strip() if birdeye_api_key else ""
        self.cielo_api_key = cielo_api_key.strip() if cielo_api_key else ""
        self.helius_api_key = helius_api_key.strip() if helius_api_key else ""
        self.rpc_url = rpc_url
        
        # Performance monitoring
        self.performance_monitor = performance_monitor
        self.enable_monitoring = performance_monitor is not None
        
        # VALIDATE REQUIRED APIS
        if not self.cielo_api_key:
            raise ValueError("Cielo Finance API key is REQUIRED")
        if not self.helius_api_key:
            raise ValueError("Helius API key is REQUIRED for accurate timestamps")
        
        # API endpoints
        self.cielo_base_url = "https://feed-api.cielo.finance/api/v1"
        self.birdeye_base_url = "https://public-api.birdeye.so"
        self.helius_base_url = f"https://api.helius.xyz/v0"
        
        # API performance tracking
        self.api_stats = {
            'birdeye': {'calls': 0, 'success': 0, 'errors': 0},
            'helius': {'calls': 0, 'success': 0, 'errors': 0},
            'rpc': {'calls': 0, 'success': 0, 'errors': 0},
            'cielo_trading_stats': {'calls': 0, 'success': 0, 'errors': 0},
            'cielo_token_pnl': {'calls': 0, 'success': 0, 'errors': 0}
        }
        
        # Request session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Zeus-Wallet-Analyzer/3.0-Performance-Monitoring',
            'Accept': 'application/json'
        })
        
        self._initialize_apis()
    
    def _initialize_apis(self):
        """Initialize API configurations with validation."""
        logger.info("🔧 Initializing Zeus API Manager with ENHANCED EXIT ANALYSIS and PERFORMANCE MONITORING...")
        
        if self.cielo_api_key:
            logger.info("✅ Cielo Finance API key configured (Trading Stats + ENHANCED Token PnL)")
        else:
            logger.error("❌ CRITICAL: Cielo Finance API key missing")
            raise ValueError("Cielo Finance API key is REQUIRED")
        
        if self.helius_api_key:
            logger.info("✅ Helius API key configured (PRIMARY timestamp source)")
        else:
            logger.error("❌ CRITICAL: Helius API key missing")
            raise ValueError("Helius API key is REQUIRED for accurate timestamps")
        
        if self.birdeye_api_key:
            logger.info("✅ Birdeye API key configured (enhanced features)")
        else:
            logger.info("⚠️ Birdeye API key not provided (limited features)")
        
        if self.enable_monitoring:
            logger.info("📈 Performance monitoring: ENABLED")
        else:
            logger.info("📈 Performance monitoring: DISABLED")
    
    def _record_api_call(self, api_name: str, cost: int, response_time: float, success: bool):
        """Record API call for performance monitoring."""
        # Update internal stats
        if api_name in self.api_stats:
            self.api_stats[api_name]['calls'] += 1
            if success:
                self.api_stats[api_name]['success'] += 1
            else:
                self.api_stats[api_name]['errors'] += 1
        
        # Record in external performance monitor if available
        if self.enable_monitoring and self.performance_monitor:
            self.performance_monitor.record_api_call(api_name, cost, response_time, success)
            
            # Check for cost alerts
            if hasattr(self.performance_monitor, 'total_cost'):
                if self.performance_monitor.total_cost > 1000:  # Alert threshold
                    logger.warning(f"⚠️ High API cost detected: {self.performance_monitor.total_cost} credits")
    
    def get_last_transaction_timestamp(self, wallet_address: str) -> Dict[str, Any]:
        """Get the REAL last transaction timestamp using Helius API with ENHANCED transaction analysis and performance monitoring."""
        start_time = time.time()
        
        try:
            logger.info(f"🕐 HELIUS PRIMARY: Getting real last transaction timestamp for {wallet_address[:8]}...")
            
            self.api_stats['helius']['calls'] += 1
            
            # Use correct Helius API endpoint and parameters
            url = f"{self.helius_base_url}/addresses/{wallet_address}/transactions"
            params = {
                'api-key': self.helius_api_key,
                'limit': 15,  # ENHANCED: Get more transactions for better analysis
                'commitment': 'confirmed'
            }
            
            logger.info(f"🔍 Helius API call: {url}")
            logger.debug(f"Parameters: {params}")
            
            response = self.session.get(url, params=params, timeout=30)
            response_time = time.time() - start_time
            
            logger.debug(f"Helius response: HTTP {response.status_code} in {response_time:.2f}s")
            
            if response.status_code == 200:
                transactions = response.json()
                self.api_stats['helius']['success'] += 1
                self._record_api_call('helius_timestamp', 0, response_time, True)
                
                logger.info(f"✅ Helius API success - received {len(transactions)} transactions in {response_time:.2f}s")
                
                if transactions and len(transactions) > 0:
                    # ENHANCED: Process transactions for better trading activity detection
                    trading_analysis = self._analyze_recent_trading_activity(transactions)
                    
                    if trading_analysis['latest_trade_timestamp']:
                        current_time = int(time.time())
                        days_since_last = max(0, (current_time - trading_analysis['latest_trade_timestamp']) / 86400)
                        
                        logger.info(f"✅ HELIUS PRIMARY: Found real last trade timestamp")
                        logger.info(f"   Timestamp: {trading_analysis['latest_trade_timestamp']}")
                        logger.info(f"   Date: {datetime.fromtimestamp(trading_analysis['latest_trade_timestamp'])}")
                        logger.info(f"   Days ago: {days_since_last:.2f}")
                        logger.info(f"   Trading activity: {trading_analysis['trading_activity_summary']}")
                        logger.info(f"   Response time: {response_time:.2f}s")
                        
                        return {
                            'success': True,
                            'last_transaction_timestamp': trading_analysis['latest_trade_timestamp'],
                            'days_since_last_trade': round(days_since_last, 1),  # 1 decimal precision
                            'transaction_count': len(transactions),
                            'source': 'helius_primary',
                            'method': 'helius_transactions_api',
                            'wallet_address': wallet_address,
                            'timestamp_accuracy': 'high',
                            'trading_analysis': trading_analysis,  # ENHANCED: Include trading activity analysis
                            'performance': {
                                'response_time': response_time,
                                'cost': 0
                            }
                        }
                    else:
                        logger.warning(f"⚠️ HELIUS: No trading transactions found in recent history")
                        self._record_api_call('helius_timestamp', 0, response_time, False)
                        return {
                            'success': False,
                            'error': 'No trading transactions found in recent history',
                            'transaction_count': len(transactions),
                            'source': 'helius_primary',
                            'method': 'helius_transactions_api',
                            'trading_analysis': trading_analysis,
                            'performance': {
                                'response_time': response_time,
                                'cost': 0
                            }
                        }
                else:
                    logger.warning(f"⚠️ HELIUS: No transactions found for wallet {wallet_address[:8]}")
                    self._record_api_call('helius_timestamp', 0, response_time, False)
                    return {
                        'success': False,
                        'error': 'No transactions found for wallet',
                        'source': 'helius_primary',
                        'method': 'helius_transactions_api',
                        'performance': {
                            'response_time': response_time,
                            'cost': 0
                        }
                    }
            
            elif response.status_code == 401:
                error_msg = "Helius API authentication failed - check API key"
                logger.error(f"❌ {error_msg}")
                self.api_stats['helius']['errors'] += 1
                self._record_api_call('helius_timestamp', 0, response_time, False)
                return {
                    'success': False,
                    'error': error_msg,
                    'error_code': 401,
                    'source': 'helius_primary',
                    'performance': {
                        'response_time': response_time,
                        'cost': 0
                    }
                }
            
            elif response.status_code == 429:
                error_msg = "Helius API rate limited"
                logger.warning(f"⚠️ {error_msg}")
                self.api_stats['helius']['errors'] += 1
                self._record_api_call('helius_timestamp', 0, response_time, False)
                return {
                    'success': False,
                    'error': error_msg,
                    'error_code': 429,
                    'source': 'helius_primary',
                    'performance': {
                        'response_time': response_time,
                        'cost': 0
                    }
                }
            
            else:
                error_msg = f"Helius API error: HTTP {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text[:200]}"
                
                logger.error(f"❌ {error_msg}")
                self.api_stats['helius']['errors'] += 1
                self._record_api_call('helius_timestamp', 0, response_time, False)
                return {
                    'success': False,
                    'error': error_msg,
                    'error_code': response.status_code,
                    'source': 'helius_primary',
                    'performance': {
                        'response_time': response_time,
                        'cost': 0
                    }
                }
            
        except requests.exceptions.Timeout:
            response_time = time.time() - start_time
            error_msg = "Helius API timeout"
            logger.error(f"❌ {error_msg} after {response_time:.2f}s")
            self.api_stats['helius']['errors'] += 1
            self._record_api_call('helius_timestamp', 0, response_time, False)
            return {
                'success': False,
                'error': error_msg,
                'source': 'helius_primary',
                'performance': {
                    'response_time': response_time,
                    'cost': 0
                }
            }
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Helius API unexpected error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.api_stats['helius']['errors'] += 1
            self._record_api_call('helius_timestamp', 0, response_time, False)
            return {
                'success': False,
                'error': error_msg,
                'source': 'helius_primary',
                'performance': {
                    'response_time': response_time,
                    'cost': 0
                }
            }
    
    def _analyze_recent_trading_activity(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        ENHANCED: Analyze recent trading activity for better exit pattern inference.
        """
        try:
            analysis = {
                'latest_trade_timestamp': None,
                'total_trades': 0,
                'swap_transactions': 0,
                'buy_transactions': 0,
                'sell_transactions': 0,
                'trading_activity_summary': 'inactive',
                'recent_activity_pattern': 'unknown'
            }
            
            trade_timestamps = []
            
            for tx in transactions:
                # Extract timestamp from transaction
                tx_timestamp = None
                
                # Try different timestamp fields
                if 'timestamp' in tx:
                    tx_timestamp = tx['timestamp']
                elif 'blockTime' in tx:
                    tx_timestamp = tx['blockTime']
                
                if not tx_timestamp:
                    continue
                
                # ENHANCED: Classify transaction type
                tx_classification = self._classify_transaction_enhanced(tx)
                
                if tx_classification['is_trading']:
                    analysis['total_trades'] += 1
                    trade_timestamps.append(tx_timestamp)
                    
                    if tx_classification['tx_type'] == 'swap':
                        analysis['swap_transactions'] += 1
                    elif tx_classification['tx_type'] == 'buy':
                        analysis['buy_transactions'] += 1
                    elif tx_classification['tx_type'] == 'sell':
                        analysis['sell_transactions'] += 1
                    
                    # Track latest trade timestamp
                    if analysis['latest_trade_timestamp'] is None or tx_timestamp > analysis['latest_trade_timestamp']:
                        analysis['latest_trade_timestamp'] = tx_timestamp
            
            # ENHANCED: Analyze trading activity pattern
            if analysis['total_trades'] > 0:
                current_time = int(time.time())
                hours_since_last = (current_time - analysis['latest_trade_timestamp']) / 3600
                
                if hours_since_last < 1:
                    analysis['trading_activity_summary'] = 'very_active'
                elif hours_since_last < 24:
                    analysis['trading_activity_summary'] = 'active'
                elif hours_since_last < 168:  # 1 week
                    analysis['trading_activity_summary'] = 'recently_active'
                else:
                    analysis['trading_activity_summary'] = 'inactive'
                
                # Analyze recent activity pattern
                if len(trade_timestamps) >= 3:
                    # Calculate time intervals between trades
                    intervals = []
                    sorted_timestamps = sorted(trade_timestamps, reverse=True)
                    for i in range(len(sorted_timestamps) - 1):
                        interval = sorted_timestamps[i] - sorted_timestamps[i + 1]
                        intervals.append(interval)
                    
                    if intervals:
                        avg_interval = sum(intervals) / len(intervals)
                        if avg_interval < 300:  # Less than 5 minutes
                            analysis['recent_activity_pattern'] = 'rapid_trading'
                        elif avg_interval < 3600:  # Less than 1 hour
                            analysis['recent_activity_pattern'] = 'frequent_trading'
                        elif avg_interval < 86400:  # Less than 1 day
                            analysis['recent_activity_pattern'] = 'regular_trading'
                        else:
                            analysis['recent_activity_pattern'] = 'occasional_trading'
            
            logger.info(f"📊 ENHANCED trading activity analysis:")
            logger.info(f"  Total trades: {analysis['total_trades']}")
            logger.info(f"  Activity level: {analysis['trading_activity_summary']}")
            logger.info(f"  Pattern: {analysis['recent_activity_pattern']}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing recent trading activity: {str(e)}")
            return {
                'latest_trade_timestamp': None,
                'total_trades': 0,
                'trading_activity_summary': 'error',
                'recent_activity_pattern': 'unknown'
            }
    
    def _classify_transaction_enhanced(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        ENHANCED: Classify transaction type for better exit analysis.
        """
        try:
            classification = {
                'is_trading': False,
                'tx_type': 'unknown',
                'confidence': 'low'
            }
            
            # Check transaction type
            tx_type = transaction.get('type', '').lower()
            if 'swap' in tx_type:
                classification.update({
                    'is_trading': True,
                    'tx_type': 'swap',
                    'confidence': 'high'
                })
                return classification
            
            # ENHANCED: Check for known DEX programs and instructions
            dex_programs = {
                '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8': 'raydium',
                'JUP2jxvXaqu7NQY1GmNF4m1vodw12LVXYxbFL2uJvfo': 'jupiter',
                '9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP': 'orca',
                'EUqojwWA2rd19FZrzeBncJsm38Jm1hEhE3zsmX3bRc2o': 'serum',
                'SSwpkEEyHdRBUNHuTw2qJC5jbTMfKu3Y5qMVBhFdHCR': 'saber'
            }
            
            # Check for known DEX programs in instructions
            if 'instructions' in transaction:
                for instruction in transaction['instructions']:
                    if isinstance(instruction, dict):
                        program_id = instruction.get('programId', '')
                        if program_id in dex_programs:
                            classification.update({
                                'is_trading': True,
                                'tx_type': 'swap',
                                'confidence': 'high',
                                'dex': dex_programs[program_id]
                            })
                            return classification
            
            # Check account data for trading indicators
            if 'accountData' in transaction:
                for account in transaction['accountData']:
                    account_str = str(account).lower()
                    if any(dex in account_str for dex in ['raydium', 'jupiter', 'orca', 'serum', 'saber']):
                        classification.update({
                            'is_trading': True,
                            'tx_type': 'swap',
                            'confidence': 'medium'
                        })
                        return classification
            
            # ENHANCED: Try to determine buy vs sell based on token flow
            # This is a simplified heuristic - in practice, this would require more sophisticated analysis
            if 'tokenTransfers' in transaction:
                token_transfers = transaction['tokenTransfers']
                if isinstance(token_transfers, list) and len(token_transfers) > 0:
                    classification.update({
                        'is_trading': True,
                        'tx_type': 'swap',  # Default to swap for now
                        'confidence': 'medium'
                    })
                    return classification
            
            # Default: assume it's a trading transaction if we can't determine otherwise
            # This is to be inclusive rather than miss trading activity
            classification.update({
                'is_trading': True,
                'tx_type': 'unknown',
                'confidence': 'low'
            })
            
            return classification
            
        except Exception as e:
            logger.debug(f"Error classifying transaction: {str(e)}")
            return {
                'is_trading': True,  # Default to True to be inclusive
                'tx_type': 'unknown',
                'confidence': 'low'
            }
    
    def get_wallet_trading_stats(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get wallet trading statistics from Cielo Finance Trading Stats API with SAFE FIELD VALIDATION and performance monitoring.
        Cost: 30 credits
        """
        start_time = time.time()
        
        try:
            if not self.cielo_api_key:
                return {
                    'success': False,
                    'error': 'Cielo Finance API key not configured',
                    'source': 'cielo_trading_stats'
                }
            
            self.api_stats['cielo_trading_stats']['calls'] += 1
            
            # Use Trading Stats endpoint (30 credits)
            url = f"{self.cielo_base_url}/{wallet_address}/trading-stats"
            
            logger.info(f"🏦 CIELO TRADING STATS: {url} (30 credits)")
            logger.debug(f"Using API key: {self.cielo_api_key[:12]}...")
            
            # Try different authentication methods for Cielo API
            auth_methods = [
                {'X-API-KEY': self.cielo_api_key},
                {'Authorization': f'Bearer {self.cielo_api_key}'},
                {'api-key': self.cielo_api_key},
                {'apikey': self.cielo_api_key}
            ]
            
            base_headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            }
            
            # Try each authentication method
            for i, auth_header in enumerate(auth_methods, 1):
                auth_method = list(auth_header.keys())[0]
                
                logger.debug(f"Trying Cielo auth method {i}/{len(auth_methods)}: {auth_method}")
                
                try:
                    headers = {**base_headers, **auth_header}
                    response = self.session.get(url, headers=headers, timeout=30)
                    response_time = time.time() - start_time
                    
                    logger.debug(f"Cielo Trading Stats response: HTTP {response.status_code} in {response_time:.2f}s")
                    
                    if response.status_code == 200:
                        # SUCCESS - Extract complete response data with SAFE FIELD VALIDATION
                        try:
                            response_data = response.json()
                            self.api_stats['cielo_trading_stats']['success'] += 1
                            self._record_api_call('cielo_trading_stats', 30, response_time, True)
                            
                            logger.info(f"✅ Cielo Trading Stats API success with {auth_method} in {response_time:.2f}s!")
                            
                            # SAFE FIELD DISCOVERY: Log the exact API response structure
                            logger.info(f"🔍 SAFE CIELO TRADING STATS FIELDS for {wallet_address[:8]}:")
                            logger.info(f"  Response type: {type(response_data)}")
                            
                            if isinstance(response_data, dict):
                                logger.info(f"  Top-level keys: {list(response_data.keys())}")
                                
                                # Extract the actual trading data from nested structure
                                actual_trading_data = response_data.get('data', {}) if 'data' in response_data else response_data
                                
                                if isinstance(actual_trading_data, dict):
                                    logger.info(f"  Trading data keys: {list(actual_trading_data.keys())}")
                                    logger.info(f"  Field count: {len(actual_trading_data)}")
                                    
                                    # SAFE FIELD LISTING for exact mapping
                                    logger.info(f"🗂️ SAFE FIELD LISTING WITH VALUES:")
                                    for field, value in actual_trading_data.items():
                                        logger.info(f"    {field}: {value} ({type(value).__name__})")
                                    
                                    # FIELD VALIDATION with SAFE type checking
                                    validation_result = self._validate_trading_stats_fields_safe(actual_trading_data)
                                    
                                    return {
                                        'success': True,
                                        'data': actual_trading_data,  # CORRECT API response - no modifications
                                        'source': 'cielo_trading_stats_safe',
                                        'auth_method_used': auth_method,
                                        'api_endpoint': 'trading-stats',
                                        'wallet_address': wallet_address,
                                        'response_timestamp': int(time.time()),
                                        'raw_response': response_data,
                                        'field_validation': validation_result,
                                        'credit_cost': 30,
                                        'field_extraction_method': 'safe_direct_mapping',
                                        'performance': {
                                            'response_time': response_time,
                                            'cost': 30
                                        }
                                    }
                                else:
                                    logger.warning(f"⚠️ Trading data is not a dict: {type(actual_trading_data)}")
                            
                            # Fallback return
                            return {
                                'success': True,
                                'data': response_data,
                                'source': 'cielo_trading_stats_safe',
                                'auth_method_used': auth_method,
                                'api_endpoint': 'trading-stats',
                                'wallet_address': wallet_address,
                                'response_timestamp': int(time.time()),
                                'raw_response': response_data,
                                'credit_cost': 30,
                                'performance': {
                                    'response_time': response_time,
                                    'cost': 30
                                }
                            }
                            
                        except ValueError as json_error:
                            logger.error(f"Failed to parse Cielo Trading Stats JSON response: {json_error}")
                            logger.debug(f"Raw response: {response.text[:500]}")
                            continue
                    
                    elif response.status_code == 404:
                        logger.warning(f"⚠️ Wallet {wallet_address[:8]}... not found in Cielo Trading Stats database")
                        self.api_stats['cielo_trading_stats']['errors'] += 1
                        self._record_api_call('cielo_trading_stats', 30, response_time, False)
                        return {
                            'success': False,
                            'error': f'Wallet not found in Cielo Trading Stats database',
                            'error_code': 404,
                            'auth_method_used': auth_method,
                            'source': 'cielo_trading_stats_safe',
                            'performance': {
                                'response_time': response_time,
                                'cost': 30
                            }
                        }
                    
                    elif response.status_code in [401, 403]:
                        logger.debug(f"Auth failed ({response.status_code}) with {auth_method} - trying next method")
                        continue
                    
                    elif response.status_code == 429:
                        logger.warning(f"⚠️ Cielo Trading Stats API rate limited - waiting before retry")
                        time.sleep(2)
                        continue
                    
                    else:
                        logger.debug(f"HTTP {response.status_code} with {auth_method}: {response.text[:200]}")
                        continue
                
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Request error with {auth_method}: {str(e)}")
                    continue
            
            # If we get here, all auth methods failed
            response_time = time.time() - start_time
            error_msg = f"All Cielo Trading Stats authentication methods failed. API key may be invalid."
            logger.error(f"❌ {error_msg}")
            self.api_stats['cielo_trading_stats']['errors'] += 1
            self._record_api_call('cielo_trading_stats', 30, response_time, False)
            
            return {
                'success': False,
                'error': error_msg,
                'attempted_auth_methods': len(auth_methods),
                'source': 'cielo_trading_stats_safe',
                'performance': {
                    'response_time': response_time,
                    'cost': 30
                }
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Cielo Trading Stats API error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.api_stats['cielo_trading_stats']['errors'] += 1
            self._record_api_call('cielo_trading_stats', 30, response_time, False)
            return {
                'success': False,
                'error': error_msg,
                'source': 'cielo_trading_stats_safe',
                'performance': {
                    'response_time': response_time,
                    'cost': 30
                }
            }
    
    def get_token_pnl(self, wallet_address: str, limit: int = 5) -> Dict[str, Any]:
        """
        Get individual token PnL data with ENHANCED structure parsing for exit analysis and performance monitoring.
        Cost: 5 credits (much cheaper than Trading Stats)
        """
        start_time = time.time()
        
        try:
            if not self.cielo_api_key:
                return {
                    'success': False,
                    'error': 'Cielo Finance API key not configured',
                    'source': 'cielo_token_pnl_enhanced'
                }
            
            self.api_stats['cielo_token_pnl']['calls'] += 1
            
            # Use Token PnL endpoint (5 credits)
            url = f"{self.cielo_base_url}/{wallet_address}/pnl/tokens"
            params = {'limit': limit}
            
            logger.info(f"📊 CIELO TOKEN PNL: {url} (5 credits, limit={limit})")
            logger.info(f"🔍 ENHANCED for exit analysis: data.items[]")
            
            # Try different authentication methods
            auth_methods = [
                {'X-API-KEY': self.cielo_api_key},
                {'Authorization': f'Bearer {self.cielo_api_key}'},
                {'api-key': self.cielo_api_key}
            ]
            
            base_headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            }
            
            for i, auth_header in enumerate(auth_methods, 1):
                headers = {**base_headers, **auth_header}
                auth_method = list(auth_header.keys())[0]
                
                try:
                    response = self.session.get(url, headers=headers, params=params, timeout=30)
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        self.api_stats['cielo_token_pnl']['success'] += 1
                        self._record_api_call('cielo_token_pnl', 5, response_time, True)
                        
                        logger.info(f"✅ Cielo Token PnL API success with {auth_method} in {response_time:.2f}s!")
                        
                        # ENHANCED: Extract tokens with better field analysis
                        tokens_data = self._extract_tokens_with_enhanced_analysis(response_data)
                        
                        logger.info(f"📊 ENHANCED extraction - Retrieved {len(tokens_data)} token PnL records")
                        
                        # ENHANCED: Log detailed token structure for exit analysis
                        if tokens_data and len(tokens_data) > 0:
                            self._log_enhanced_token_structure(tokens_data[0])
                        
                        return {
                            'success': True,
                            'data': response_data,  # Return full response for structure analysis
                            'tokens_extracted': tokens_data,  # ENHANCED extracted tokens
                            'source': 'cielo_token_pnl_enhanced',
                            'auth_method_used': auth_method,
                            'wallet_address': wallet_address,
                            'tokens_count': len(tokens_data),
                            'credit_cost': 5,
                            'structure_used': 'data.items[]',
                            'extraction_method': 'enhanced_exit_analysis',
                            'performance': {
                                'response_time': response_time,
                                'cost': 5
                            }
                        }
                    
                    elif response.status_code == 404:
                        response_time = time.time() - start_time
                        logger.warning(f"⚠️ No token PnL data found for wallet {wallet_address[:8]}...")
                        self._record_api_call('cielo_token_pnl', 5, response_time, False)
                        return {
                            'success': False,
                            'error': 'No token PnL data found',
                            'error_code': 404,
                            'source': 'cielo_token_pnl_enhanced',
                            'performance': {
                                'response_time': response_time,
                                'cost': 5
                            }
                        }
                    
                    elif response.status_code in [401, 403]:
                        logger.debug(f"Auth failed ({response.status_code}) with {auth_method} - trying next method")
                        continue
                    
                    else:
                        logger.debug(f"HTTP {response.status_code} with {auth_method}")
                        continue
                
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Token PnL request error with {auth_method}: {str(e)}")
                    continue
            
            # All auth methods failed
            response_time = time.time() - start_time
            error_msg = f"All Cielo Token PnL authentication methods failed"
            logger.error(f"❌ {error_msg}")
            self.api_stats['cielo_token_pnl']['errors'] += 1
            self._record_api_call('cielo_token_pnl', 5, response_time, False)
            
            return {
                'success': False,
                'error': error_msg,
                'source': 'cielo_token_pnl_enhanced',
                'performance': {
                    'response_time': response_time,
                    'cost': 5
                }
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Cielo Token PnL API error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.api_stats['cielo_token_pnl']['errors'] += 1
            self._record_api_call('cielo_token_pnl', 5, response_time, False)
            return {
                'success': False,
                'error': error_msg,
                'source': 'cielo_token_pnl_enhanced',
                'performance': {
                    'response_time': response_time,
                    'cost': 5
                }
            }
    
    def _extract_tokens_with_enhanced_analysis(self, response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ENHANCED: Extract tokens with better field analysis for exit patterns."""
        try:
            logger.info(f"🔍 ENHANCED token extraction for exit analysis...")
            
            # Extract using known structure: data.items[]
            if isinstance(response_data, dict):
                if 'data' in response_data:
                    data_section = response_data['data']
                    if isinstance(data_section, dict) and 'items' in data_section:
                        items = data_section['items']
                        if isinstance(items, list):
                            logger.info(f"✅ ENHANCED structure found: data.items[] with {len(items)} tokens")
                            return items
                    elif isinstance(data_section, list):
                        logger.info(f"✅ Found data as direct array with {len(data_section)} tokens")
                        return data_section
                
                # Fallback checks for other possible structures
                for key in ['items', 'tokens', 'results']:
                    if key in response_data and isinstance(response_data[key], list):
                        logger.info(f"✅ Found tokens in {key} with {len(response_data[key])} items")
                        return response_data[key]
            
            logger.warning(f"❌ No tokens found in ENHANCED extraction")
            logger.warning(f"Available keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'not dict'}")
            return []
            
        except Exception as e:
            logger.error(f"Error in ENHANCED token extraction: {str(e)}")
            return []
    
    def _log_enhanced_token_structure(self, sample_token: Dict[str, Any]) -> None:
        """ENHANCED: Log token structure for exit analysis debugging."""
        try:
            logger.info(f"🔍 ENHANCED TOKEN STRUCTURE for exit analysis:")
            
            # Key fields for exit analysis
            exit_analysis_fields = [
                'roi_percentage',
                'total_pnl_usd',
                'holding_time_seconds',
                'num_swaps',
                'entry_price',
                'exit_price',
                'buy_amount',
                'sell_amount',
                'token_symbol',
                'token_address'
            ]
            
            for field in exit_analysis_fields:
                if field in sample_token:
                    value = sample_token[field]
                    logger.info(f"  {field}: {value} ({type(value).__name__}) ✅")
                else:
                    logger.info(f"  {field}: NOT FOUND ❌")
            
            # Log all available fields
            logger.info(f"  All available fields: {list(sample_token.keys())}")
            
            # ENHANCED: Estimate exit analysis capability
            exit_analysis_score = 0
            if 'roi_percentage' in sample_token:
                exit_analysis_score += 25
            if 'holding_time_seconds' in sample_token:
                exit_analysis_score += 25
            if 'num_swaps' in sample_token:
                exit_analysis_score += 30
            if 'total_pnl_usd' in sample_token:
                exit_analysis_score += 20
            
            logger.info(f"  Exit Analysis Capability: {exit_analysis_score}%")
            
        except Exception as e:
            logger.error(f"Error logging enhanced token structure: {str(e)}")
    
    def _validate_trading_stats_fields_safe(self, trading_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        SAFELY validate Trading Stats fields with proper type checking.
        """
        validation = {
            'valid_fields': [],
            'invalid_fields': [],
            'missing_expected': [],
            'field_types': {},
            'safe_mappings': {},
            'validation_errors': []
        }
        
        try:
            # SAFE expected fields with proper type definitions
            expected_fields = {
                'pnl': {'types': (int, float), 'allow_negative': True},
                'winrate': {'types': (int, float), 'min_val': 0, 'max_val': 100},
                'swaps_count': {'types': (int,), 'min_val': 0},
                'buy_count': {'types': (int,), 'min_val': 0},
                'sell_count': {'types': (int,), 'min_val': 0},
                'total_buy_amount_usd': {'types': (int, float), 'min_val': 0},
                'total_sell_amount_usd': {'types': (int, float), 'min_val': 0},
                'average_buy_amount_usd': {'types': (int, float), 'min_val': 0},
                'average_holding_time_sec': {'types': (int, float), 'min_val': 0},
                'consecutive_trading_days': {'types': (int,), 'min_val': 0},
                'holding_distribution': {'types': (dict,), 'skip_range_check': True},
                'roi_distribution': {'types': (dict,), 'skip_range_check': True}
            }
            
            # SAFELY check each field
            for field, field_config in expected_fields.items():
                try:
                    if field in trading_data:
                        value = trading_data[field]
                        expected_types = field_config['types']
                        
                        # SAFE TYPE CHECK
                        if isinstance(value, expected_types):
                            validation['valid_fields'].append(field)
                            validation['field_types'][field] = type(value).__name__
                            
                            # SAFE RANGE VALIDATION - only for numeric types
                            if not field_config.get('skip_range_check', False):
                                range_valid = True
                                range_errors = []
                                
                                # Only check ranges for numeric types
                                if isinstance(value, (int, float)):
                                    if 'min_val' in field_config and value < field_config['min_val']:
                                        range_valid = False
                                        range_errors.append(f'Below minimum: {value} < {field_config["min_val"]}')
                                    
                                    if 'max_val' in field_config and value > field_config['max_val']:
                                        range_valid = False
                                        range_errors.append(f'Above maximum: {value} > {field_config["max_val"]}')
                                
                                if not range_valid:
                                    validation['invalid_fields'].append({
                                        'field': field,
                                        'type': 'range_error',
                                        'value': value,
                                        'errors': range_errors
                                    })
                            
                            # SAFE field mapping
                            zeus_mapping = self._map_to_zeus_field_safe(field, value)
                            if zeus_mapping:
                                validation['safe_mappings'][field] = zeus_mapping
                                
                        else:
                            validation['invalid_fields'].append({
                                'field': field,
                                'type': 'type_error',
                                'expected_types': [t.__name__ for t in expected_types],
                                'actual_type': type(value).__name__,
                                'value': str(value)[:100]  # Truncate long values
                            })
                    else:
                        validation['missing_expected'].append(field)
                        
                except Exception as field_error:
                    validation['validation_errors'].append({
                        'field': field,
                        'error': str(field_error)
                    })
                    logger.debug(f"Error validating field {field}: {field_error}")
            
            return validation
            
        except Exception as e:
            logger.error(f"Critical error in SAFE field validation: {str(e)}")
            return {
                'valid_fields': [],
                'invalid_fields': [],
                'missing_expected': [],
                'field_types': {},
                'safe_mappings': {},
                'validation_errors': [{'general_error': str(e)}],
                'critical_error': True
            }
    
    def _map_to_zeus_field_safe(self, cielo_field: str, value: Any) -> Optional[Dict[str, Any]]:
        """SAFELY map Cielo field to Zeus field with transformation info."""
        try:
            mappings = {
                'winrate': {
                    'zeus_field': '7_day_winrate',
                    'transformation': 'direct' if isinstance(value, (int, float)) and 0 <= value <= 100 else 'needs_validation',
                    'value': value,
                    'safe': True
                },
                'pnl': {
                    'zeus_field': 'roi_7_day',
                    'transformation': 'calculate_roi_from_pnl_and_buy_amount',
                    'value': value,
                    'note': 'Requires total_buy_amount_usd for ROI calculation',
                    'safe': True
                },
                'average_holding_time_sec': {
                    'zeus_field': 'average_holding_time_minutes',
                    'transformation': 'divide_by_60',
                    'value': value,
                    'safe': True
                },
                'holding_distribution': {
                    'zeus_field': 'unique_tokens_30d',
                    'transformation': 'extract_total_tokens',
                    'value': value.get('total_tokens') if isinstance(value, dict) and 'total_tokens' in value else None,
                    'safe': isinstance(value, dict)
                },
                'average_buy_amount_usd': {
                    'zeus_field': 'avg_sol_buy_per_token',
                    'transformation': 'divide_by_sol_price_estimate',
                    'value': value,
                    'safe': isinstance(value, (int, float))
                },
                'buy_count': {
                    'zeus_field': 'avg_buys_per_token',
                    'transformation': 'divide_by_unique_tokens',
                    'value': value,
                    'note': 'Requires unique_tokens for calculation',
                    'safe': isinstance(value, int)
                }
            }
            
            mapping = mappings.get(cielo_field)
            if mapping and mapping.get('safe', False):
                return mapping
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Error in SAFE field mapping for {cielo_field}: {str(e)}")
            return None
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get detailed API status information with REQUIRED validation."""
        status = {
            'apis_configured': [],
            'api_status': {},
            'system_ready': True,
            'wallet_compatible': False,
            'token_analysis_ready': False,
            'timestamp_accuracy': 'none',
            'field_extraction_method': 'safe_direct_mapping',
            'exit_analysis_enhanced': True,
            'performance_monitoring': self.enable_monitoring
        }
        
        # Check REQUIRED APIs
        required_apis = {
            'cielo': self.cielo_api_key,
            'helius': self.helius_api_key
        }
        
        for api_name, api_key in required_apis.items():
            if api_key:
                status['apis_configured'].append(api_name)
                status['api_status'][api_name] = 'operational'
            else:
                status['api_status'][api_name] = 'missing_required'
                status['system_ready'] = False
        
        # Check OPTIONAL APIs
        if self.birdeye_api_key:
            status['apis_configured'].append('birdeye')
            status['api_status']['birdeye'] = 'operational'
            status['token_analysis_ready'] = True
        else:
            status['api_status']['birdeye'] = 'not_configured'
        
        # RPC is always available
        status['apis_configured'].append('rpc')
        status['api_status']['rpc'] = 'operational'
        
        # System capabilities
        if self.cielo_api_key and self.helius_api_key:
            status['wallet_compatible'] = True
            status['timestamp_accuracy'] = 'high'
            status['token_pnl_structure'] = 'data.items[] (ENHANCED)'
            status['field_extraction'] = 'safe_direct_mapping (ENHANCED)'
            status['exit_analysis'] = 'enhanced_pattern_inference'
        elif self.cielo_api_key:
            status['wallet_compatible'] = True
            status['timestamp_accuracy'] = 'low'
        
        return status
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get API performance statistics with enhanced monitoring data."""
        perf_stats = {}
        
        for api_name, stats in self.api_stats.items():
            total_calls = stats['calls']
            success_calls = stats['success']
            error_calls = stats['errors']
            
            success_rate = (success_calls / total_calls * 100) if total_calls > 0 else 0
            
            perf_stats[api_name] = {
                'total_calls': total_calls,
                'successful_calls': success_calls,
                'failed_calls': error_calls,
                'success_rate_percent': round(success_rate, 2),
                'status': 'good' if success_rate >= 80 else 'degraded' if success_rate >= 50 else 'poor'
            }
        
        # Add performance monitoring info if available
        if self.enable_monitoring and self.performance_monitor:
            monitor_report = self.performance_monitor.get_performance_report()
            perf_stats['session_summary'] = {
                'total_cost': monitor_report.get('total_cost', 0),
                'avg_response_time': monitor_report.get('avg_response_time', 0),
                'session_duration': monitor_report.get('session_duration', 0),
                'monitoring_enabled': True
            }
        else:
            perf_stats['session_summary'] = {
                'monitoring_enabled': False
            }
        
        return perf_stats
    
    def test_helius_connection(self, test_wallet: str = "DhDiCRqc4BAojxUDzBonf7KAujejtpUryxDsuqPqGKA9") -> Dict[str, Any]:
        """Test Helius API connection with ENHANCED transaction analysis and performance monitoring."""
        try:
            logger.info(f"🧪 Testing Helius API connection with ENHANCED analysis: {test_wallet[:8]}...")
            
            start_time = time.time()
            result = self.get_last_transaction_timestamp(test_wallet)
            test_duration = time.time() - start_time
            
            if result.get('success'):
                logger.info("✅ Helius API connection test successful!")
                return {
                    'connection_test': 'success',
                    'api_working': True,
                    'timestamp_found': result.get('last_transaction_timestamp') is not None,
                    'days_since_last': result.get('days_since_last_trade', 'unknown'),
                    'transaction_count': result.get('transaction_count', 0),
                    'trading_analysis': result.get('trading_analysis', {}),
                    'enhanced_features': True,
                    'performance': {
                        'test_duration': test_duration,
                        'response_time': result.get('performance', {}).get('response_time', 0)
                    }
                }
            else:
                logger.error(f"❌ Helius API connection test failed: {result.get('error', 'Unknown error')}")
                return {
                    'connection_test': 'failed',
                    'api_working': False,
                    'error': result.get('error', 'Unknown error'),
                    'enhanced_features': False,
                    'performance': {
                        'test_duration': test_duration,
                        'response_time': result.get('performance', {}).get('response_time', 0)
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Helius API connection test error: {str(e)}")
            return {
                'connection_test': 'error',
                'api_working': False,
                'error': str(e),
                'enhanced_features': False
            }
    
    def test_cielo_api_connection(self, test_wallet: str = "DhDiCRqc4BAojxUDzBonf7KAujejtpUryxDsuqPqGKA9") -> Dict[str, Any]:
        """Test Cielo API connection with ENHANCED Token PnL analysis and performance monitoring."""
        try:
            logger.info(f"🧪 Testing Cielo API connection with ENHANCED features: {test_wallet[:8]}...")
            
            start_time = time.time()
            
            # Test Trading Stats (30 credits)
            trading_stats_result = self.get_wallet_trading_stats(test_wallet)
            
            # Test Token PnL with ENHANCED analysis (5 credits)
            token_pnl_result = self.get_token_pnl(test_wallet, limit=2)
            
            test_duration = time.time() - start_time
            
            if trading_stats_result.get('success') or token_pnl_result.get('success'):
                logger.info("✅ Cielo API connection test successful!")
                
                # Calculate total cost and response times
                total_cost = 0
                total_response_time = 0
                
                if trading_stats_result.get('success'):
                    total_cost += trading_stats_result.get('performance', {}).get('cost', 30)
                    total_response_time += trading_stats_result.get('performance', {}).get('response_time', 0)
                
                if token_pnl_result.get('success'):
                    total_cost += token_pnl_result.get('performance', {}).get('cost', 5)
                    total_response_time += token_pnl_result.get('performance', {}).get('response_time', 0)
                
                return {
                    'connection_test': 'success',
                    'api_working': True,
                    'trading_stats_working': trading_stats_result.get('success', False),
                    'token_pnl_working': token_pnl_result.get('success', False),
                    'trading_stats_fields': list(trading_stats_result.get('data', {}).keys()) if trading_stats_result.get('success') else [],
                    'token_pnl_count': token_pnl_result.get('tokens_count', 0) if token_pnl_result.get('success') else 0,
                    'token_pnl_structure': token_pnl_result.get('structure_used', 'unknown'),
                    'auth_method': trading_stats_result.get('auth_method_used', 'unknown'),
                    'field_extraction_method': 'safe_direct_mapping',
                    'validation_safe': True,
                    'enhanced_exit_analysis': True,
                    'performance': {
                        'test_duration': test_duration,
                        'total_cost': total_cost,
                        'total_response_time': total_response_time,
                        'avg_response_time': total_response_time / 2 if total_response_time > 0 else 0
                    }
                }
            else:
                logger.error(f"❌ Cielo API connection test failed")
                return {
                    'connection_test': 'failed',
                    'api_working': False,
                    'trading_stats_error': trading_stats_result.get('error', 'Unknown error'),
                    'token_pnl_error': token_pnl_result.get('error', 'Unknown error'),
                    'enhanced_exit_analysis': False,
                    'performance': {
                        'test_duration': test_duration
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Cielo API connection test error: {str(e)}")
            return {
                'connection_test': 'error',
                'api_working': False,
                'error': str(e),
                'enhanced_exit_analysis': False
            }
    
    def __str__(self) -> str:
        """String representation of API manager."""
        apis = []
        if self.cielo_api_key:
            apis.append("Cielo✅")
        if self.helius_api_key:
            apis.append("Helius✅")
        if self.birdeye_api_key:
            apis.append("Birdeye✅")
        apis.append("RPC✅")
        
        monitoring_status = "📈Monitor✅" if self.enable_monitoring else "📈Monitor❌"
        
        return f"ZeusAPIManager({', '.join(apis)}, ENHANCED Exit Analysis, Token PnL: data.items[], {monitoring_status})"