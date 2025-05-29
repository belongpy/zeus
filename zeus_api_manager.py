"""
Zeus API Manager - FIXED Cielo Finance Trading Stats API Integration
Based on actual Cielo Finance API documentation at https://developer.cielo.finance/reference/getfeed

FIXES:
- Use correct Cielo Trading Stats API endpoint
- Proper authentication handling for Cielo API
- Return complete API response structure for proper data extraction
- Fixed 403 authentication attempts (now DEBUG level)
- Better error handling and response structure
"""

import logging
import time
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("zeus.api_manager")

class ZeusAPIManager:
    """Zeus API manager with FIXED Cielo Finance Trading Stats API integration."""
    
    def __init__(self, birdeye_api_key: str = "", cielo_api_key: str = "", 
                 helius_api_key: str = "", rpc_url: str = "https://api.mainnet-beta.solana.com"):
        """Initialize with proper authentication."""
        
        # Store API keys
        self.birdeye_api_key = birdeye_api_key.strip() if birdeye_api_key else ""
        self.cielo_api_key = cielo_api_key.strip() if cielo_api_key else ""
        self.helius_api_key = helius_api_key.strip() if helius_api_key else ""
        self.rpc_url = rpc_url
        
        # FIXED: Use correct Cielo Finance API endpoints from documentation
        self.cielo_base_url = "https://feed-api.cielo.finance/api/v1"
        self.birdeye_base_url = "https://public-api.birdeye.so"
        self.helius_base_url = f"https://api.helius.xyz/v0" if self.helius_api_key else ""
        
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
            'User-Agent': 'Zeus-Wallet-Analyzer/1.0',
            'Accept': 'application/json'
        })
        
        self._initialize_apis()
    
    def _initialize_apis(self):
        """Initialize API configurations."""
        if self.cielo_api_key:
            logger.info("✅ Cielo Finance API key configured")
        else:
            logger.warning("❌ Cielo Finance API key not provided")
        
        if self.birdeye_api_key:
            logger.info("✅ Birdeye API key configured")
        else:
            logger.info("ℹ️ Birdeye API key not provided")
        
        if self.helius_api_key:
            logger.info("✅ Helius API key configured")
        else:
            logger.info("ℹ️ Helius API key not provided")
    
    def get_wallet_trading_stats(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get wallet trading statistics from Cielo Finance Trading Stats API.
        FIXED: Uses correct endpoint and returns complete response structure.
        Based on: https://feed-api.cielo.finance/api/v1/{wallet}/trading-stats
        """
        try:
            if not self.cielo_api_key:
                return {
                    'success': False,
                    'error': 'Cielo Finance API key not configured',
                    'source': 'cielo_trading_stats'
                }
            
            self.api_stats['cielo']['calls'] += 1
            
            # FIXED: Use correct Trading Stats endpoint from documentation
            url = f"{self.cielo_base_url}/{wallet_address}/trading-stats"
            
            logger.info(f"🔧 Calling Cielo Trading Stats API: {url}")
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
                
                logger.debug(f"Trying auth method {i}/{len(auth_methods)}: {auth_method}")
                
                try:
                    response = self.session.get(url, headers=headers, timeout=30)
                    
                    logger.debug(f"Response: HTTP {response.status_code}")
                    
                    if response.status_code == 200:
                        # SUCCESS - Extract complete response data
                        try:
                            response_data = response.json()
                            self.api_stats['cielo']['success'] += 1
                            
                            logger.info(f"✅ Cielo Trading Stats API success with {auth_method}!")
                            logger.info(f"Response contains keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Non-dict response'}")
                            
                            # FIXED: Return complete response structure for proper extraction
                            return {
                                'success': True,
                                'data': response_data,  # Complete Cielo Trading Stats response
                                'source': 'cielo_trading_stats',
                                'auth_method_used': auth_method,
                                'api_endpoint': 'trading-stats',
                                'wallet_address': wallet_address,
                                'response_timestamp': int(time.time())
                            }
                            
                        except ValueError as json_error:
                            logger.error(f"Failed to parse JSON response: {json_error}")
                            logger.debug(f"Raw response: {response.text[:500]}")
                            continue
                    
                    elif response.status_code == 403:
                        logger.debug(f"403 Forbidden with {auth_method} - trying next method")
                        if response.text:
                            logger.debug(f"403 response: {response.text[:200]}")
                        continue
                    
                    elif response.status_code == 401:
                        logger.debug(f"401 Unauthorized with {auth_method} - trying next method")
                        continue
                    
                    elif response.status_code == 404:
                        # 404 means wallet not found, but auth worked
                        logger.warning(f"⚠️ Wallet {wallet_address[:8]}... not found in Cielo Trading Stats database")
                        self.api_stats['cielo']['errors'] += 1
                        return {
                            'success': False,
                            'error': f'Wallet not found in Cielo Trading Stats database',
                            'error_code': 404,
                            'auth_method_used': auth_method,
                            'source': 'cielo_trading_stats'
                        }
                    
                    elif response.status_code == 429:
                        # Rate limited
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
            error_msg = f"All authentication methods failed for Cielo Trading Stats API. API key may be invalid."
            logger.error(f"❌ {error_msg}")
            self.api_stats['cielo']['errors'] += 1
            
            return {
                'success': False,
                'error': error_msg,
                'attempted_auth_methods': len(auth_methods),
                'source': 'cielo_trading_stats',
                'api_endpoint': 'trading-stats'
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
    
    def get_wallet_pnl_tokens(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get wallet token PnL from Cielo Finance Token PnL API.
        Based on: https://feed-api.cielo.finance/api/v1/{wallet}/pnl/tokens
        """
        try:
            if not self.cielo_api_key:
                return {
                    'success': False,
                    'error': 'Cielo Finance API key not configured'
                }
            
            self.api_stats['cielo']['calls'] += 1
            
            url = f"{self.cielo_base_url}/{wallet_address}/pnl/tokens"
            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json',
                'X-API-KEY': self.cielo_api_key
            }
            
            logger.info(f"Making Cielo Token PnL API call: {url}")
            
            response = self.session.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                self.api_stats['cielo']['success'] += 1
                
                logger.info(f"✅ Cielo token PnL success for {wallet_address[:8]}...")
                
                return {
                    'success': True,
                    'data': data,
                    'source': 'cielo_token_pnl'
                }
            
            else:
                error_msg = f"Cielo Token PnL API error: HTTP {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text[:200]}"
                
                logger.error(f"❌ {error_msg}")
                self.api_stats['cielo']['errors'] += 1
                return {
                    'success': False,
                    'error': error_msg
                }
            
        except Exception as e:
            error_msg = f"Cielo Token PnL API error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.api_stats['cielo']['errors'] += 1
            return {
                'success': False,
                'error': error_msg
            }
    
    def get_wallet_aggregated_pnl(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get wallet aggregated PnL from Cielo Finance Aggregated Token PnL API.
        Based on: https://feed-api.cielo.finance/api/v1/{wallet}/pnl/total-stats
        """
        try:
            if not self.cielo_api_key:
                return {
                    'success': False,
                    'error': 'Cielo Finance API key not configured'
                }
            
            self.api_stats['cielo']['calls'] += 1
            
            url = f"{self.cielo_base_url}/{wallet_address}/pnl/total-stats"
            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json',
                'X-API-KEY': self.cielo_api_key
            }
            
            logger.info(f"Making Cielo Aggregated PnL API call: {url}")
            
            response = self.session.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                self.api_stats['cielo']['success'] += 1
                
                logger.info(f"✅ Cielo aggregated PnL success for {wallet_address[:8]}...")
                
                return {
                    'success': True,
                    'data': data,
                    'source': 'cielo_aggregated_pnl'
                }
            
            else:
                error_msg = f"Cielo Aggregated PnL API error: HTTP {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text[:200]}"
                
                logger.error(f"❌ {error_msg}")
                self.api_stats['cielo']['errors'] += 1
                return {
                    'success': False,
                    'error': error_msg
                }
            
        except Exception as e:
            error_msg = f"Cielo Aggregated PnL API error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.api_stats['cielo']['errors'] += 1
            return {
                'success': False,
                'error': error_msg
            }
    
    def get_enhanced_transactions(self, wallet_address: str, limit: int = 100) -> Dict[str, Any]:
        """Get enhanced parsed transactions using Helius API."""
        try:
            if not self.helius_api_key:
                return {
                    'success': False,
                    'error': 'Helius API not configured'
                }
            
            self.api_stats['helius']['calls'] += 1
            
            url = f"{self.helius_base_url}/addresses/{wallet_address}/transactions"
            params = {
                'api-key': self.helius_api_key,
                'limit': limit,
                'commitment': 'confirmed'
            }
            
            logger.info(f"Making Helius API call for {wallet_address[:8]}...")
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                self.api_stats['helius']['success'] += 1
                
                logger.info(f"✅ Helius API success - {len(data)} transactions")
                
                return {
                    'success': True,
                    'data': data
                }
            
            else:
                error_msg = f"Helius API error: HTTP {response.status_code}"
                logger.error(f"❌ {error_msg}")
                self.api_stats['helius']['errors'] += 1
                return {
                    'success': False,
                    'error': error_msg
                }
            
        except Exception as e:
            error_msg = f"Helius API error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.api_stats['helius']['errors'] += 1
            return {
                'success': False,
                'error': error_msg
            }
    
    def make_rpc_call(self, method: str, params: List[Any]) -> Dict[str, Any]:
        """Make direct RPC call to Solana node."""
        try:
            self.api_stats['rpc']['calls'] += 1
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params
            }
            
            response = self.session.post(
                self.rpc_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            if "result" in result:
                self.api_stats['rpc']['success'] += 1
                return {
                    'success': True,
                    'data': result["result"]
                }
            else:
                self.api_stats['rpc']['errors'] += 1
                return {
                    'success': False,
                    'error': result.get("error", "Unknown RPC error")
                }
                
        except Exception as e:
            logger.error(f"RPC call error: {str(e)}")
            self.api_stats['rpc']['errors'] += 1
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get detailed API status information."""
        status = {
            'apis_configured': [],
            'api_status': {},
            'zeus_compatible': True,
            'wallet_compatible': False,
            'token_analysis_ready': False
        }
        
        # Check Cielo Finance API
        if self.cielo_api_key:
            status['apis_configured'].append('cielo')
            status['api_status']['cielo'] = 'operational'
            status['wallet_compatible'] = True
        else:
            status['api_status']['cielo'] = 'not_configured'
            status['zeus_compatible'] = False
        
        # Check Birdeye API
        if self.birdeye_api_key:
            status['apis_configured'].append('birdeye')
            status['api_status']['birdeye'] = 'operational'
            status['token_analysis_ready'] = True
        else:
            status['api_status']['birdeye'] = 'not_configured'
        
        # Check Helius API
        if self.helius_api_key:
            status['apis_configured'].append('helius')
            status['api_status']['helius'] = 'operational'
        else:
            status['api_status']['helius'] = 'not_configured'
        
        # Check RPC
        status['apis_configured'].append('rpc')
        status['api_status']['rpc'] = 'operational'
        
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
    
    def test_cielo_api_connection(self, test_wallet: str = "DhDiCRqc4BAojxUDzBonf7KAujejtpUryxDsuqPqGKA9") -> Dict[str, Any]:
        """
        Test Cielo API connection with a known wallet address.
        Used for debugging and API key validation.
        """
        try:
            logger.info(f"🧪 Testing Cielo API connection with wallet: {test_wallet[:8]}...")
            
            result = self.get_wallet_trading_stats(test_wallet)
            
            if result.get('success'):
                logger.info("✅ Cielo API connection test successful!")
                
                # Log the structure for debugging
                data = result.get('data', {})
                if isinstance(data, dict):
                    logger.info(f"Response contains {len(data)} fields: {list(data.keys())}")
                    
                    # Log sample values (first few fields)
                    sample_fields = list(data.keys())[:5]
                    for field in sample_fields:
                        value = data.get(field)
                        value_str = str(value)[:100] if value is not None else "None"
                        logger.info(f"  {field}: {type(value).__name__} = {value_str}")
                
                return {
                    'connection_test': 'success',
                    'api_working': True,
                    'response_fields': list(data.keys()) if isinstance(data, dict) else [],
                    'auth_method': result.get('auth_method_used', 'unknown')
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
        if self.birdeye_api_key:
            apis.append("Birdeye✅")
        if self.helius_api_key:
            apis.append("Helius✅")
        apis.append("RPC✅")
        
        return f"ZeusAPIManager({', '.join(apis)})"