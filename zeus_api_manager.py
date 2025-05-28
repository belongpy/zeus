"""
Zeus API Manager - Cielo Finance with Authentication
Fixes HTTP 403 "API key is required" error
"""

import logging
import time
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("zeus.api_manager")

class ZeusAPIManager:
    """Zeus API manager with proper Cielo Finance authentication."""
    
    def __init__(self, birdeye_api_key: str = "", cielo_api_key: str = "", 
                 helius_api_key: str = "", rpc_url: str = "https://api.mainnet-beta.solana.com"):
        """Initialize with proper authentication."""
        
        # Store API keys
        self.birdeye_api_key = birdeye_api_key.strip() if birdeye_api_key else ""
        self.cielo_api_key = cielo_api_key.strip() if cielo_api_key else ""
        self.helius_api_key = helius_api_key.strip() if helius_api_key else ""
        self.rpc_url = rpc_url
        
        # API endpoints
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
    
    def _get_cielo_headers(self) -> Dict[str, str]:
        """Get headers with API key for Cielo Finance API."""
        base_headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        if self.cielo_api_key:
            # Try different common API key header patterns
            # We'll test multiple options since the documentation wasn't clear
            auth_headers = [
                {'Authorization': f'Bearer {self.cielo_api_key}'},
                {'Authorization': f'Api-Key {self.cielo_api_key}'},
                {'X-API-KEY': self.cielo_api_key},
                {'x-api-key': self.cielo_api_key},
                {'API-KEY': self.cielo_api_key},
                {'api-key': self.cielo_api_key},
                {'apikey': self.cielo_api_key},
                {'Authorization': self.cielo_api_key}
            ]
            
            # Return the base headers for the first attempt
            # We'll try different auth methods in the main function
            return {**base_headers, **auth_headers[0]}
        
        return base_headers
    
    def get_wallet_trading_stats(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get wallet trading statistics from Cielo Finance API.
        Tries multiple authentication methods to handle HTTP 403.
        """
        try:
            if not self.cielo_api_key:
                return {
                    'success': False,
                    'error': 'Cielo Finance API key not configured'
                }
            
            self.api_stats['cielo']['calls'] += 1
            
            # Use exact endpoint from documentation
            url = f"{self.cielo_base_url}/{wallet_address}/trading-stats"
            
            logger.info(f"Making Cielo API call: {url}")
            logger.info(f"Using API key: {self.cielo_api_key[:15]}...")
            
            # Try different authentication methods
            auth_methods = [
                {'Authorization': f'Bearer {self.cielo_api_key}'},
                {'X-API-KEY': self.cielo_api_key},
                {'x-api-key': self.cielo_api_key},
                {'API-KEY': self.cielo_api_key},
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
                
                logger.info(f"Trying auth method {i}/{len(auth_methods)}: {auth_method}")
                
                try:
                    response = self.session.get(url, headers=headers, timeout=30)
                    
                    logger.info(f"Response: HTTP {response.status_code}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        self.api_stats['cielo']['success'] += 1
                        
                        logger.info(f"✅ Cielo API success with {auth_method}!")
                        logger.info(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Non-dict response'}")
                        
                        return {
                            'success': True,
                            'data': data,
                            'auth_method_used': auth_method
                        }
                    
                    elif response.status_code == 403:
                        logger.info(f"403 Forbidden with {auth_method} - trying next method")
                        if response.text:
                            logger.debug(f"403 response: {response.text[:200]}")
                        continue
                    
                    elif response.status_code == 401:
                        logger.info(f"401 Unauthorized with {auth_method} - trying next method")
                        continue
                    
                    elif response.status_code == 404:
                        # 404 means wallet not found, but auth worked
                        logger.warning(f"⚠️ Wallet {wallet_address[:8]}... not found in Cielo database")
                        self.api_stats['cielo']['errors'] += 1
                        return {
                            'success': False,
                            'error': f'Wallet not found in Cielo Finance database',
                            'auth_method_used': auth_method
                        }
                    
                    else:
                        logger.info(f"HTTP {response.status_code} with {auth_method} - trying next method")
                        continue
                
                except requests.exceptions.RequestException as e:
                    logger.info(f"Request error with {auth_method}: {str(e)}")
                    continue
            
            # If we get here, all auth methods failed
            error_msg = f"All authentication methods failed. API key may be invalid or endpoint changed."
            logger.error(f"❌ {error_msg}")
            self.api_stats['cielo']['errors'] += 1
            
            return {
                'success': False,
                'error': error_msg,
                'attempted_auth_methods': len(auth_methods)
            }
            
        except Exception as e:
            error_msg = f"Cielo Finance API error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.api_stats['cielo']['errors'] += 1
            return {
                'success': False,
                'error': error_msg
            }
    
    def get_wallet_pnl_tokens(self, wallet_address: str) -> Dict[str, Any]:
        """Get wallet token PnL from Cielo Finance API."""
        try:
            if not self.cielo_api_key:
                return {
                    'success': False,
                    'error': 'Cielo Finance API key not configured'
                }
            
            self.api_stats['cielo']['calls'] += 1
            
            url = f"{self.cielo_base_url}/{wallet_address}/pnl/tokens"
            headers = self._get_cielo_headers()
            
            logger.info(f"Making Cielo PnL API call: {url}")
            
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
                error_msg = f"Cielo PnL API error: HTTP {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text[:200]}"
                
                logger.error(f"❌ {error_msg}")
                self.api_stats['cielo']['errors'] += 1
                return {
                    'success': False,
                    'error': error_msg
                }
            
        except Exception as e:
            error_msg = f"Cielo PnL API error: {str(e)}"
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