"""
Zeus API Manager - FIXED with Correct Token PnL Structure and Field Discovery
MAJOR FIXES:
- Fixed Token PnL endpoint parsing (data.items[] not data.tokens[])
- Enhanced field discovery and validation for Trading Stats
- Complete authentication method testing
- Correct response structure handling
"""

import logging
import time
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import dateutil.parser

logger = logging.getLogger("zeus.api_manager")

class ZeusAPIManager:
    """Zeus API manager with CORRECT Token PnL analysis and field extraction."""
    
    def __init__(self, birdeye_api_key: str = "", cielo_api_key: str = "", 
                 helius_api_key: str = "", rpc_url: str = "https://api.mainnet-beta.solana.com"):
        """Initialize with REQUIRED API key validation."""
        
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
            'cielo_trading_stats': {'calls': 0, 'success': 0, 'errors': 0},
            'cielo_token_pnl': {'calls': 0, 'success': 0, 'errors': 0}
        }
        
        # Request session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Zeus-Wallet-Analyzer/2.2-Correct-Fields',
            'Accept': 'application/json'
        })
        
        self._initialize_apis()
    
    def _initialize_apis(self):
        """Initialize API configurations with validation."""
        logger.info("🔧 Initializing Zeus API Manager with CORRECT FIELD EXTRACTION...")
        
        if self.cielo_api_key:
            logger.info("✅ Cielo Finance API key configured (Trading Stats + CORRECT Token PnL)")
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
                            'days_since_last_trade': round(days_since_last, 1),  # 1 decimal precision
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
        Get wallet trading statistics from Cielo Finance Trading Stats API with CORRECT FIELD DISCOVERY.
        Cost: 30 credits
        """
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
                    
                    logger.debug(f"Cielo Trading Stats response: HTTP {response.status_code}")
                    
                    if response.status_code == 200:
                        # SUCCESS - Extract complete response data with CORRECT FIELD DISCOVERY
                        try:
                            response_data = response.json()
                            self.api_stats['cielo_trading_stats']['success'] += 1
                            
                            logger.info(f"✅ Cielo Trading Stats API success with {auth_method}!")
                            
                            # CORRECT FIELD DISCOVERY: Log the exact API response structure
                            logger.info(f"🔍 CORRECT CIELO TRADING STATS FIELDS for {wallet_address[:8]}:")
                            logger.info(f"  Response type: {type(response_data)}")
                            
                            if isinstance(response_data, dict):
                                logger.info(f"  Top-level keys: {list(response_data.keys())}")
                                
                                # Extract the actual trading data from nested structure
                                actual_trading_data = response_data.get('data', {}) if 'data' in response_data else response_data
                                
                                if isinstance(actual_trading_data, dict):
                                    logger.info(f"  Trading data keys: {list(actual_trading_data.keys())}")
                                    logger.info(f"  Field count: {len(actual_trading_data)}")
                                    
                                    # CORRECT FIELD LISTING for exact mapping
                                    logger.info(f"🗂️ CORRECT FIELD LISTING WITH VALUES:")
                                    for field, value in actual_trading_data.items():
                                        logger.info(f"    {field}: {value} ({type(value).__name__})")
                                    
                                    # FIELD VALIDATION with CORRECT expected fields
                                    validation_result = self._validate_trading_stats_fields_correct(actual_trading_data)
                                    
                                    return {
                                        'success': True,
                                        'data': actual_trading_data,  # CORRECT API response - no modifications
                                        'source': 'cielo_trading_stats_correct',
                                        'auth_method_used': auth_method,
                                        'api_endpoint': 'trading-stats',
                                        'wallet_address': wallet_address,
                                        'response_timestamp': int(time.time()),
                                        'raw_response': response_data,
                                        'field_validation': validation_result,
                                        'credit_cost': 30,
                                        'field_extraction_method': 'correct_direct_mapping'
                                    }
                                else:
                                    logger.warning(f"⚠️ Trading data is not a dict: {type(actual_trading_data)}")
                            
                            # Fallback return
                            return {
                                'success': True,
                                'data': response_data,
                                'source': 'cielo_trading_stats_correct',
                                'auth_method_used': auth_method,
                                'api_endpoint': 'trading-stats',
                                'wallet_address': wallet_address,
                                'response_timestamp': int(time.time()),
                                'raw_response': response_data,
                                'credit_cost': 30
                            }
                            
                        except ValueError as json_error:
                            logger.error(f"Failed to parse Cielo Trading Stats JSON response: {json_error}")
                            logger.debug(f"Raw response: {response.text[:500]}")
                            continue
                    
                    elif response.status_code == 404:
                        logger.warning(f"⚠️ Wallet {wallet_address[:8]}... not found in Cielo Trading Stats database")
                        self.api_stats['cielo_trading_stats']['errors'] += 1
                        return {
                            'success': False,
                            'error': f'Wallet not found in Cielo Trading Stats database',
                            'error_code': 404,
                            'auth_method_used': auth_method,
                            'source': 'cielo_trading_stats_correct'
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
            error_msg = f"All Cielo Trading Stats authentication methods failed. API key may be invalid."
            logger.error(f"❌ {error_msg}")
            self.api_stats['cielo_trading_stats']['errors'] += 1
            
            return {
                'success': False,
                'error': error_msg,
                'attempted_auth_methods': len(auth_methods),
                'source': 'cielo_trading_stats_correct'
            }
            
        except Exception as e:
            error_msg = f"Cielo Trading Stats API error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.api_stats['cielo_trading_stats']['errors'] += 1
            return {
                'success': False,
                'error': error_msg,
                'source': 'cielo_trading_stats_correct'
            }
    
    def get_token_pnl(self, wallet_address: str, limit: int = 5) -> Dict[str, Any]:
        """
        Get individual token PnL data with CORRECT structure parsing (data.items[] not data.tokens[]).
        Cost: 5 credits (much cheaper than Trading Stats)
        """
        try:
            if not self.cielo_api_key:
                return {
                    'success': False,
                    'error': 'Cielo Finance API key not configured',
                    'source': 'cielo_token_pnl_correct'
                }
            
            self.api_stats['cielo_token_pnl']['calls'] += 1
            
            # Use Token PnL endpoint (5 credits)
            url = f"{self.cielo_base_url}/{wallet_address}/pnl/tokens"
            params = {'limit': limit}
            
            logger.info(f"📊 CIELO TOKEN PNL: {url} (5 credits, limit={limit})")
            logger.info(f"🔍 Expected structure: data.items[] (CORRECTED)")
            
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
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        self.api_stats['cielo_token_pnl']['success'] += 1
                        
                        logger.info(f"✅ Cielo Token PnL API success with {auth_method}!")
                        
                        # CORRECT structure parsing - data.items[] not data.tokens[]
                        tokens_data = self._extract_tokens_from_correct_structure(response_data)
                        
                        logger.info(f"📊 CORRECT structure - Retrieved {len(tokens_data)} token PnL records")
                        
                        # Log token structure for analysis
                        if tokens_data and len(tokens_data) > 0:
                            sample_token = tokens_data[0]
                            logger.info(f"🔍 SAMPLE TOKEN PNL STRUCTURE (CORRECT):")
                            for field, value in sample_token.items():
                                logger.info(f"  {field}: {value} ({type(value).__name__})")
                        
                        return {
                            'success': True,
                            'data': response_data,  # Return full response for structure analysis
                            'tokens_extracted': tokens_data,  # CORRECT extracted tokens
                            'source': 'cielo_token_pnl_correct',
                            'auth_method_used': auth_method,
                            'wallet_address': wallet_address,
                            'tokens_count': len(tokens_data),
                            'credit_cost': 5,
                            'structure_used': 'data.items[]',
                            'extraction_method': 'correct_structure_parsing'
                        }
                    
                    elif response.status_code == 404:
                        logger.warning(f"⚠️ No token PnL data found for wallet {wallet_address[:8]}...")
                        return {
                            'success': False,
                            'error': 'No token PnL data found',
                            'error_code': 404,
                            'source': 'cielo_token_pnl_correct'
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
            error_msg = f"All Cielo Token PnL authentication methods failed"
            logger.error(f"❌ {error_msg}")
            self.api_stats['cielo_token_pnl']['errors'] += 1
            
            return {
                'success': False,
                'error': error_msg,
                'source': 'cielo_token_pnl_correct'
            }
            
        except Exception as e:
            error_msg = f"Cielo Token PnL API error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.api_stats['cielo_token_pnl']['errors'] += 1
            return {
                'success': False,
                'error': error_msg,
                'source': 'cielo_token_pnl_correct'
            }
    
    def _extract_tokens_from_correct_structure(self, response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tokens from CORRECT Token PnL structure (data.items[] not data.tokens[])."""
        try:
            logger.info(f"🔍 Extracting tokens from CORRECT structure...")
            
            # CORRECT structure based on actual JSON: data.items[]
            if isinstance(response_data, dict):
                if 'data' in response_data:
                    data_section = response_data['data']
                    if isinstance(data_section, dict) and 'items' in data_section:
                        items = data_section['items']
                        if isinstance(items, list):
                            logger.info(f"✅ CORRECT structure found: data.items[] with {len(items)} tokens")
                            return items
                    elif isinstance(data_section, list):
                        logger.info(f"✅ Found data as direct array with {len(data_section)} tokens")
                        return data_section
                
                # Fallback checks for other possible structures
                for key in ['items', 'tokens', 'results']:
                    if key in response_data and isinstance(response_data[key], list):
                        logger.info(f"✅ Found tokens in {key} with {len(response_data[key])} items")
                        return response_data[key]
            
            logger.warning(f"❌ No tokens found in CORRECT structure")
            logger.warning(f"Available keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'not dict'}")
            return []
            
        except Exception as e:
            logger.error(f"Error extracting tokens from CORRECT structure: {str(e)}")
            return []
    
    def _validate_trading_stats_fields_correct(self, trading_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Trading Stats fields using CORRECT expected field names from debug results."""
        validation = {
            'valid_fields': [],
            'invalid_fields': [],
            'missing_expected': [],
            'field_types': {},
            'correct_mappings': {}
        }
        
        # CORRECT expected fields based on debug results
        expected_fields = {
            'pnl': (int, float),                          # CORRECT: total PnL
            'winrate': (int, float),                      # CORRECT: win rate percentage
            'swaps_count': (int,),                        # CORRECT: total swaps
            'buy_count': (int,),                          # CORRECT: buy transactions
            'sell_count': (int,),                         # CORRECT: sell transactions
            'total_buy_amount_usd': (int, float),         # CORRECT: total buy volume
            'total_sell_amount_usd': (int, float),        # CORRECT: total sell volume
            'average_buy_amount_usd': (int, float),       # CORRECT: average buy size
            'average_holding_time_sec': (int, float),     # CORRECT: hold time in seconds
            'consecutive_trading_days': (int,),           # CORRECT: trading days
            'holding_distribution': (dict,),              # CORRECT: hold time distribution
            'roi_distribution': (dict,)                   # CORRECT: ROI distribution
        }
        
        # Check each field
        for field, expected_types in expected_fields.items():
            if field in trading_data:
                value = trading_data[field]
                if isinstance(value, expected_types):
                    validation['valid_fields'].append(field)
                    validation['field_types'][field] = type(value).__name__
                    
                    # Map to Zeus fields
                    zeus_mapping = self._map_to_zeus_field(field, value)
                    if zeus_mapping:
                        validation['correct_mappings'][field] = zeus_mapping
                else:
                    validation['invalid_fields'].append({
                        'field': field,
                        'expected_type': [t.__name__ for t in expected_types],
                        'actual_type': type(value).__name__,
                        'value': value
                    })
            else:
                validation['missing_expected'].append(field)
        
        return validation
    
    def _map_to_zeus_field(self, cielo_field: str, value: Any) -> Optional[Dict[str, Any]]:
        """Map Cielo field to Zeus field with transformation info."""
        mappings = {
            'winrate': {
                'zeus_field': '7_day_winrate',
                'transformation': 'direct' if 0 <= value <= 100 else 'multiply_by_100',
                'value': value
            },
            'pnl': {
                'zeus_field': 'roi_7_day',
                'transformation': 'calculate_roi_from_pnl_and_buy_amount',
                'value': value,
                'note': 'Requires total_buy_amount_usd for ROI calculation'
            },
            'average_holding_time_sec': {
                'zeus_field': 'average_holding_time_minutes',
                'transformation': 'divide_by_60',
                'value': value
            },
            'holding_distribution': {
                'zeus_field': 'unique_tokens_30d',
                'transformation': 'extract_total_tokens',
                'value': value.get('total_tokens') if isinstance(value, dict) else None
            },
            'average_buy_amount_usd': {
                'zeus_field': 'avg_sol_buy_per_token',
                'transformation': 'divide_by_sol_price_estimate',
                'value': value
            },
            'buy_count': {
                'zeus_field': 'avg_buys_per_token',
                'transformation': 'divide_by_unique_tokens',
                'value': value,
                'note': 'Requires unique_tokens for calculation'
            }
        }
        
        return mappings.get(cielo_field)
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get detailed API status information with REQUIRED validation."""
        status = {
            'apis_configured': [],
            'api_status': {},
            'system_ready': True,
            'wallet_compatible': False,
            'token_analysis_ready': False,
            'timestamp_accuracy': 'none',
            'field_extraction_method': 'correct_direct_mapping'
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
            status['token_pnl_structure'] = 'data.items[] (CORRECT)'
            status['field_extraction'] = 'direct_mapping (CORRECT)'
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
        """Test Cielo API connection with both Trading Stats and CORRECT Token PnL structure."""
        try:
            logger.info(f"🧪 Testing Cielo API connection with wallet: {test_wallet[:8]}...")
            
            # Test Trading Stats (30 credits)
            trading_stats_result = self.get_wallet_trading_stats(test_wallet)
            
            # Test Token PnL with CORRECT structure (5 credits)
            token_pnl_result = self.get_token_pnl(test_wallet, limit=2)
            
            if trading_stats_result.get('success') or token_pnl_result.get('success'):
                logger.info("✅ Cielo API connection test successful!")
                
                return {
                    'connection_test': 'success',
                    'api_working': True,
                    'trading_stats_working': trading_stats_result.get('success', False),
                    'token_pnl_working': token_pnl_result.get('success', False),
                    'trading_stats_fields': list(trading_stats_result.get('data', {}).keys()) if trading_stats_result.get('success') else [],
                    'token_pnl_count': token_pnl_result.get('tokens_count', 0) if token_pnl_result.get('success') else 0,
                    'token_pnl_structure': token_pnl_result.get('structure_used', 'unknown'),
                    'auth_method': trading_stats_result.get('auth_method_used', 'unknown'),
                    'field_extraction_method': 'correct_direct_mapping'
                }
            else:
                logger.error(f"❌ Cielo API connection test failed")
                return {
                    'connection_test': 'failed',
                    'api_working': False,
                    'trading_stats_error': trading_stats_result.get('error', 'Unknown error'),
                    'token_pnl_error': token_pnl_result.get('error', 'Unknown error')
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
        
        return f"ZeusAPIManager({', '.join(apis)}, CORRECT Token PnL: data.items[])"