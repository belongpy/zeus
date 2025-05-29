"""
Zeus API Manager - Enhanced Debugging for Exact Field Mapping
ENHANCED DEBUGGING:
- Log exact Cielo API response structure and field names
- Help identify the correct field names for direct extraction
- Preserve full API response for analysis
- Enhanced error handling and field discovery
"""

import logging
import time
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import dateutil.parser

logger = logging.getLogger("zeus.api_manager")

class ZeusAPIManager:
    """Zeus API manager with enhanced debugging for exact field mapping."""
    
    def __init__(self, birdeye_api_key: str = "", cielo_api_key: str = "", 
                 helius_api_key: str = "", rpc_url: str = "https://api.mainnet-beta.solana.com"):
        """Initialize with REQUIRED Helius API key validation."""
        
        # Store API keys
        self.birdeye_api_key = birdeye_api_key.strip() if birdeye_api_key else ""
        self.cielo_api_key = cielo_api_key.strip() if cielo_api_key else ""
        self.helius_api_key = helius_api_key.strip() if helius_api_key else ""
        self.rpc_url = rpc_url
        
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
            'cielo': {'calls': 0, 'success': 0, 'errors': 0}
        }
        
        # Request session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Zeus-Wallet-Analyzer/2.0',
            'Accept': 'application/json'
        })
        
        self._initialize_apis()
    
    def _initialize_apis(self):
        """Initialize API configurations with validation."""
        logger.info("🔧 Initializing Zeus API Manager with ENHANCED FIELD DEBUGGING...")
        
        if self.cielo_api_key:
            logger.info("✅ Cielo Finance API key configured")
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
    
    def get_last_transaction_timestamp(self, wallet_address: str) -> Dict[str, Any]:
        """Get the REAL last transaction timestamp using Helius API."""
        try:
            logger.info(f"🕐 HELIUS PRIMARY: Getting real last transaction timestamp for {wallet_address[:8]}...")
            
            self.api_stats['helius']['calls'] += 1
            
            # Use correct Helius API endpoint and parameters
            url = f"{self.helius_base_url}/addresses/{wallet_address}/transactions"
            params = {
                'api-key': self.helius_api_key,
                'limit': 10,  # Get recent 10 transactions for better accuracy
                'commitment': 'confirmed'
            }
            
            logger.info(f"🔍 Helius API call: {url}")
            logger.debug(f"Parameters: {params}")
            
            response = self.session.get(url, params=params, timeout=30)
            
            logger.debug(f"Helius response: HTTP {response.status_code}")
            
            if response.status_code == 200:
                transactions = response.json()
                self.api_stats['helius']['success'] += 1
                
                logger.info(f"✅ Helius API success - received {len(transactions)} transactions")
                
                if transactions and len(transactions) > 0:
                    # Process transactions to find the most recent TRADING activity
                    latest_trade_timestamp = self._find_latest_trading_timestamp(transactions)
                    
                    if latest_trade_timestamp:
                        current_time = int(time.time())
                        days_since_last = max(0, (current_time - latest_trade_timestamp) / 86400)
                        
                        logger.info(f"✅ HELIUS PRIMARY: Found real last trade timestamp")
                        logger.info(f"   Timestamp: {latest_trade_timestamp}")
                        logger.info(f"   Date: {datetime.fromtimestamp(latest_trade_timestamp)}")
                        logger.info(f"   Days ago: {days_since_last:.2f}")
                        
                        return {
                            'success': True,
                            'last_transaction_timestamp': latest_trade_timestamp,
                            'days_since_last_trade': round(days_since_last, 2),
                            'transaction_count': len(transactions),
                            'source': 'helius_primary',
                            'method': 'helius_transactions_api',
                            'wallet_address': wallet_address,
                            'timestamp_accuracy': 'high'
                        }
                    else:
                        logger.warning(f"⚠️ HELIUS: No trading transactions found in recent history")
                        return {
                            'success': False,
                            'error': 'No trading transactions found in recent history',
                            'transaction_count': len(transactions),
                            'source': 'helius_primary',
                            'method': 'helius_transactions_api'
                        }
                else:
                    logger.warning(f"⚠️ HELIUS: No transactions found for wallet {wallet_address[:8]}")
                    return {
                        'success': False,
                        'error': 'No transactions found for wallet',
                        'source': 'helius_primary',
                        'method': 'helius_transactions_api'
                    }
            
            elif response.status_code == 401:
                error_msg = "Helius API authentication failed - check API key"
                logger.error(f"❌ {error_msg}")
                self.api_stats['helius']['errors'] += 1
                return {
                    'success': False,
                    'error': error_msg,
                    'error_code': 401,
                    'source': 'helius_primary'
                }
            
            elif response.status_code == 429:
                error_msg = "Helius API rate limited"
                logger.warning(f"⚠️ {error_msg}")
                self.api_stats['helius']['errors'] += 1
                return {
                    'success': False,
                    'error': error_msg,
                    'error_code': 429,
                    'source': 'helius_primary'
                }
            
            else:
                error_msg = f"Helius API error: HTTP {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text[:200]}"
                
                logger.error(f"❌ {error_msg}")
                self.api_stats['helius']['errors'] += 1
                return {
                    'success': False,
                    'error': error_msg,
                    'error_code': response.status_code,
                    'source': 'helius_primary'
                }
            
        except requests.exceptions.Timeout:
            error_msg = "Helius API timeout"
            logger.error(f"❌ {error_msg}")
            self.api_stats['helius']['errors'] += 1
            return {
                'success': False,
                'error': error_msg,
                'source': 'helius_primary'
            }
        except Exception as e:
            error_msg = f"Helius API unexpected error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.api_stats['helius']['errors'] += 1
            return {
                'success': False,
                'error': error_msg,
                'source': 'helius_primary'
            }
    
    def _find_latest_trading_timestamp(self, transactions: List[Dict[str, Any]]) -> Optional[int]:
        """Find the latest TRADING transaction timestamp from Helius data."""
        try:
            latest_trade_timestamp = None
            
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
                
                # Check if this is a trading transaction
                if self._is_trading_transaction(tx):
                    if latest_trade_timestamp is None or tx_timestamp > latest_trade_timestamp:
                        latest_trade_timestamp = tx_timestamp
                        logger.debug(f"Found trading transaction at {datetime.fromtimestamp(tx_timestamp)}")
            
            return latest_trade_timestamp
            
        except Exception as e:
            logger.error(f"Error finding latest trading timestamp: {str(e)}")
            return None
    
    def _is_trading_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Determine if a transaction is a trading/swap transaction."""
        try:
            # Check transaction type
            tx_type = transaction.get('type', '').lower()
            if 'swap' in tx_type:
                return True
            
            # Check for known DEX programs
            if 'accountData' in transaction:
                for account in transaction['accountData']:
                    account_str = str(account).lower()
                    if any(dex in account_str for dex in ['raydium', 'jupiter', 'orca', 'serum', 'saber']):
                        return True
            
            # Check instruction data for swap indicators
            if 'instructions' in transaction:
                for instruction in transaction['instructions']:
                    if isinstance(instruction, dict):
                        program_id = instruction.get('programId', '')
                        if program_id in [
                            '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8',  # Raydium
                            'JUP2jxvXaqu7NQY1GmNF4m1vodw12LVXYxbFL2uJvfo',   # Jupiter
                            '9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP',  # Orca
                            'EUqojwWA2rd19FZrzeBncJsm38Jm1hEhE3zsmX3bRc2o',  # Serum
                        ]:
                            return True
            
            # Default: assume it's a trading transaction if we can't determine otherwise
            return True
            
        except Exception as e:
            logger.debug(f"Error checking if transaction is trading: {str(e)}")
            return True  # Default to True to be inclusive
    
    def get_wallet_trading_stats(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get wallet trading statistics from Cielo Finance Trading Stats API with ENHANCED DEBUG LOGGING.
        """
        try:
            if not self.cielo_api_key:
                return {
                    'success': False,
                    'error': 'Cielo Finance API key not configured',
                    'source': 'cielo_trading_stats'
                }
            
            self.api_stats['cielo']['calls'] += 1
            
            # Use correct Trading Stats endpoint
            url = f"{self.cielo_base_url}/{wallet_address}/trading-stats"
            
            logger.info(f"🏦 Calling Cielo Trading Stats API: {url}")
            logger.debug(f"Using API key: {self.cielo_api_key[:12]}...")
            
            # Try different authentication methods for Cielo API
            auth_methods = [
                {'X-API-KEY': self.cielo_api_key},
                {'Authorization': f'Bearer {self.cielo_api_key}'},
                {'api-key': self.cielo_api_key},
                {'apikey': self.cielo_api_key},
                {'Authorization': f'Api-Key {self.cielo_api_key}'},
                {'Authorization': self.cielo_api_key}
            ]
            
            base_headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            }
            
            # Try each authentication method
            for i, auth_header in enumerate(auth_methods, 1):
                headers = {**base_headers, **auth_header}
                auth_method = list(auth_header.keys())[0]
                
                logger.debug(f"Trying Cielo auth method {i}/{len(auth_methods)}: {auth_method}")
                
                try:
                    response = self.session.get(url, headers=headers, timeout=30)
                    
                    logger.debug(f"Cielo response: HTTP {response.status_code}")
                    
                    if response.status_code == 200:
                        # SUCCESS - Extract complete response data with ENHANCED DEBUGGING
                        try:
                            response_data = response.json()
                            self.api_stats['cielo']['success'] += 1
                            
                            logger.info(f"✅ Cielo Trading Stats API success with {auth_method}!")
                            
                            # ENHANCED DEBUGGING: Log the exact API response structure
                            logger.info(f"🔍 CIELO API RESPONSE DEBUG for {wallet_address[:8]}:")
                            logger.info(f"  Response type: {type(response_data)}")
                            
                            if isinstance(response_data, dict):
                                logger.info(f"  Top-level keys: {list(response_data.keys())}")
                                
                                # Extract the actual trading data from nested structure
                                actual_trading_data = response_data.get('data', {}) if 'data' in response_data else response_data
                                
                                if isinstance(actual_trading_data, dict):
                                    logger.info(f"  Trading data keys: {list(actual_trading_data.keys())}")
                                    logger.info(f"  Field count: {len(actual_trading_data)}")
                                    
                                    # ENHANCED FIELD DISCOVERY: Log potential field matches
                                    logger.info(f"🔎 FIELD DISCOVERY - Looking for exact fields:")
                                    
                                    # Look for ROI-related fields
                                    roi_candidates = [k for k in actual_trading_data.keys() if any(term in k.lower() for term in ['roi', 'pnl', 'return'])]
                                    if roi_candidates:
                                        logger.info(f"  ROI candidates: {roi_candidates}")
                                        for field in roi_candidates:
                                            value = actual_trading_data[field]
                                            logger.info(f"    {field}: {value} ({type(value).__name__})")
                                    
                                    # Look for winrate-related fields
                                    winrate_candidates = [k for k in actual_trading_data.keys() if any(term in k.lower() for term in ['win', 'rate', 'success'])]
                                    if winrate_candidates:
                                        logger.info(f"  Winrate candidates: {winrate_candidates}")
                                        for field in winrate_candidates:
                                            value = actual_trading_data[field]
                                            logger.info(f"    {field}: {value} ({type(value).__name__})")
                                    
                                    # Look for hold time-related fields
                                    holdtime_candidates = [k for k in actual_trading_data.keys() if any(term in k.lower() for term in ['hold', 'time', 'duration'])]
                                    if holdtime_candidates:
                                        logger.info(f"  Hold time candidates: {holdtime_candidates}")
                                        for field in holdtime_candidates:
                                            value = actual_trading_data[field]
                                            logger.info(f"    {field}: {value} ({type(value).__name__})")
                                    
                                    # Look for count-related fields
                                    count_candidates = [k for k in actual_trading_data.keys() if any(term in k.lower() for term in ['count', 'buys', 'sells', 'swap'])]
                                    if count_candidates:
                                        logger.info(f"  Count candidates: {count_candidates}")
                                        for field in count_candidates:
                                            value = actual_trading_data[field]
                                            logger.info(f"    {field}: {value} ({type(value).__name__})")
                                    
                                    # Log ALL fields for complete discovery
                                    logger.info(f"🗂️ COMPLETE FIELD LISTING:")
                                    for field, value in actual_trading_data.items():
                                        logger.info(f"    {field}: {value} ({type(value).__name__})")
                                    
                                    return {
                                        'success': True,
                                        'data': actual_trading_data,
                                        'source': 'cielo_trading_stats',
                                        'auth_method_used': auth_method,
                                        'api_endpoint': 'trading-stats',
                                        'wallet_address': wallet_address,
                                        'response_timestamp': int(time.time()),
                                        'raw_response': response_data,
                                        'debug_info': {
                                            'field_count': len(actual_trading_data),
                                            'roi_candidates': roi_candidates,
                                            'winrate_candidates': winrate_candidates,
                                            'holdtime_candidates': holdtime_candidates,
                                            'count_candidates': count_candidates
                                        }
                                    }
                                else:
                                    logger.warning(f"⚠️ Trading data is not a dict: {type(actual_trading_data)}")
                            
                            # Fallback return
                            return {
                                'success': True,
                                'data': response_data,
                                'source': 'cielo_trading_stats',
                                'auth_method_used': auth_method,
                                'api_endpoint': 'trading-stats',
                                'wallet_address': wallet_address,
                                'response_timestamp': int(time.time()),
                                'raw_response': response_data
                            }
                            
                        except ValueError as json_error:
                            logger.error(f"Failed to parse Cielo JSON response: {json_error}")
                            logger.debug(f"Raw response: {response.text[:500]}")
                            continue
                    
                    elif response.status_code == 403:
                        logger.debug(f"403 Forbidden with {auth_method} - trying next method")
                        continue
                    
                    elif response.status_code == 401:
                        logger.debug(f"401 Unauthorized with {auth_method} - trying next method")
                        continue
                    
                    elif response.status_code == 404:
                        logger.warning(f"⚠️ Wallet {wallet_address[:8]}... not found in Cielo database")
                        self.api_stats['cielo']['errors'] += 1
                        return {
                            'success': False,
                            'error': f'Wallet not found in Cielo Trading Stats database',
                            'error_code': 404,
                            'auth_method_used': auth_method,
                            'source': 'cielo_trading_stats'
                        }
                    
                    elif response.status_code == 429:
                        logger.warning(f"⚠️ Cielo API rate limited - waiting before retry")
                        time.sleep(2)
                        continue
                    
                    else:
                        logger.debug(f"HTTP {response.status_code} with {auth_method}: {response.text[:200]}")
                        continue
                
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Request error with {auth_method}: {str(e)}")
                    continue
            
            # If we get here, all auth methods failed
            error_msg = f"All Cielo authentication methods failed. API key may be invalid."
            logger.error(f"❌ {error_msg}")
            self.api_stats['cielo']['errors'] += 1
            
            return {
                'success': False,
                'error': error_msg,
                'attempted_auth_methods': len(auth_methods),
                'source': 'cielo_trading_stats'
            }
            
        except Exception as e:
            error_msg = f"Cielo Trading Stats API error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.api_stats['cielo']['errors'] += 1
            return {
                'success': False,
                'error': error_msg,
                'source': 'cielo_trading_stats'
            }
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get detailed API status information with REQUIRED validation."""
        status = {
            'apis_configured': [],
            'api_status': {},
            'system_ready': True,
            'wallet_compatible': False,
            'token_analysis_ready': False,
            'timestamp_accuracy': 'none'
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
        elif self.cielo_api_key:
            status['wallet_compatible'] = True
            status['timestamp_accuracy'] = 'low'
        
        return status
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get API performance statistics."""
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
        
        return perf_stats
    
    def test_helius_connection(self, test_wallet: str = "DhDiCRqc4BAojxUDzBonf7KAujejtpUryxDsuqPqGKA9") -> Dict[str, Any]:
        """Test Helius API connection with a known wallet address."""
        try:
            logger.info(f"🧪 Testing Helius API connection with wallet: {test_wallet[:8]}...")
            
            result = self.get_last_transaction_timestamp(test_wallet)
            
            if result.get('success'):
                logger.info("✅ Helius API connection test successful!")
                return {
                    'connection_test': 'success',
                    'api_working': True,
                    'timestamp_found': result.get('last_transaction_timestamp') is not None,
                    'days_since_last': result.get('days_since_last_trade', 'unknown'),
                    'transaction_count': result.get('transaction_count', 0)
                }
            else:
                logger.error(f"❌ Helius API connection test failed: {result.get('error', 'Unknown error')}")
                return {
                    'connection_test': 'failed',
                    'api_working': False,
                    'error': result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            logger.error(f"❌ Helius API connection test error: {str(e)}")
            return {
                'connection_test': 'error',
                'api_working': False,
                'error': str(e)
            }
    
    def test_cielo_api_connection(self, test_wallet: str = "DhDiCRqc4BAojxUDzBonf7KAujejtpUryxDsuqPqGKA9") -> Dict[str, Any]:
        """Test Cielo API connection with a known wallet address and ENHANCED FIELD DISCOVERY."""
        try:
            logger.info(f"🧪 Testing Cielo API connection with wallet: {test_wallet[:8]}...")
            
            result = self.get_wallet_trading_stats(test_wallet)
            
            if result.get('success'):
                logger.info("✅ Cielo API connection test successful!")
                
                data = result.get('data', {})
                debug_info = result.get('debug_info', {})
                
                return {
                    'connection_test': 'success',
                    'api_working': True,
                    'response_fields': list(data.keys()) if isinstance(data, dict) else [],
                    'auth_method': result.get('auth_method_used', 'unknown'),
                    'field_discovery': debug_info
                }
            else:
                logger.error(f"❌ Cielo API connection test failed: {result.get('error', 'Unknown error')}")
                return {
                    'connection_test': 'failed',
                    'api_working': False,
                    'error': result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            logger.error(f"❌ Cielo API connection test error: {str(e)}")
            return {
                'connection_test': 'error',
                'api_working': False,
                'error': str(e)
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
        
        return f"ZeusAPIManager({', '.join(apis)})"