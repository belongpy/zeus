"""
Zeus API Manager - Unified API Integration Layer
Handles all external API calls for Zeus wallet analysis

APIs Supported:
- Cielo Finance (wallet trading stats) - REQUIRED
- Birdeye (token price data) - RECOMMENDED  
- Helius (enhanced transaction parsing) - OPTIONAL
- Solana RPC (direct blockchain access) - FALLBACK
"""

import logging
import time
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("zeus.api_manager")

class ZeusAPIManager:
    """Unified API manager for Zeus wallet analysis."""
    
    def __init__(self, birdeye_api_key: str = "", cielo_api_key: str = "", 
                 helius_api_key: str = "", rpc_url: str = "https://api.mainnet-beta.solana.com"):
        """
        Initialize Zeus API manager.
        
        Args:
            birdeye_api_key: Birdeye API key (recommended)
            cielo_api_key: Cielo Finance API key (required)
            helius_api_key: Helius API key (optional)
            rpc_url: Solana RPC endpoint URL
        """
        self.birdeye_api_key = birdeye_api_key
        self.cielo_api_key = cielo_api_key
        self.helius_api_key = helius_api_key
        self.rpc_url = rpc_url
        
        # For now, we'll simulate the API classes since we don't have the actual SDKs
        self.birdeye_api = None
        self.cielo_api = None
        self.helius_api = None
        
        # API performance tracking
        self.api_stats = {
            'birdeye': {'calls': 0, 'success': 0, 'errors': 0},
            'helius': {'calls': 0, 'success': 0, 'errors': 0},
            'rpc': {'calls': 0, 'success': 0, 'errors': 0},
            'cielo': {'calls': 0, 'success': 0, 'errors': 0}
        }
        
        # Initialize APIs
        self._initialize_apis()
    
    def _initialize_apis(self):
        """Initialize all APIs with error handling."""
        
        # For development, we'll use mock implementations
        # In production, you would import the actual API classes
        
        if self.cielo_api_key:
            logger.info("✅ Cielo Finance API key configured")
        else:
            logger.warning("❌ Cielo Finance API key not provided")
        
        if self.birdeye_api_key:
            logger.info("✅ Birdeye API key configured")
        else:
            logger.info("ℹ️ Birdeye API key not provided - limited token analysis")
        
        if self.helius_api_key:
            logger.info("✅ Helius API key configured")
        else:
            logger.info("ℹ️ Helius API key not provided - basic transaction parsing")
    
    def get_wallet_trading_stats(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get wallet trading statistics from Cielo Finance.
        
        Args:
            wallet_address: Wallet address to analyze
            
        Returns:
            Dict with trading statistics
        """
        try:
            if not self.cielo_api_key:
                return {
                    'success': False,
                    'error': 'Cielo Finance API not configured'
                }
            
            self.api_stats['cielo']['calls'] += 1
            
            # Mock implementation - replace with actual Cielo API call
            # In production: result = self.cielo_api.get_wallet_trading_stats(wallet_address)
            
            # For now, return mock data
            mock_data = {
                'wallet_address': wallet_address,
                'total_trades': 25,
                'total_volume_sol': 150.5,
                'win_rate': 0.68,
                'avg_hold_time_hours': 24.5,
                'largest_win_percent': 450.0,
                'largest_loss_percent': -75.0
            }
            
            self.api_stats['cielo']['success'] += 1
            
            return {
                'success': True,
                'data': mock_data
            }
            
        except Exception as e:
            logger.error(f"Error getting wallet trading stats: {str(e)}")
            self.api_stats['cielo']['errors'] += 1
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_enhanced_transactions(self, wallet_address: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get enhanced parsed transactions using Helius API.
        
        Args:
            wallet_address: Wallet address
            limit: Maximum number of transactions
            
        Returns:
            Dict with enhanced transaction data
        """
        try:
            if not self.helius_api_key:
                return {
                    'success': False,
                    'error': 'Helius API not configured'
                }
            
            self.api_stats['helius']['calls'] += 1
            
            # Mock implementation - replace with actual Helius API call
            mock_transactions = []
            for i in range(min(limit, 10)):  # Mock 10 transactions
                mock_transactions.append({
                    'signature': f'mock_signature_{i}',
                    'timestamp': int(time.time()) - (i * 3600),  # 1 hour apart
                    'type': 'swap',
                    'token_mint': f'mock_token_{i}',
                    'sol_amount': 5.0 + i,
                    'success': True
                })
            
            self.api_stats['helius']['success'] += 1
            
            return {
                'success': True,
                'data': mock_transactions
            }
            
        except Exception as e:
            logger.error(f"Error getting enhanced transactions: {str(e)}")
            self.api_stats['helius']['errors'] += 1
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_token_price_history(self, token_address: str, start_time: int, 
                              end_time: int, resolution: str = "1h") -> Dict[str, Any]:
        """
        Get token price history for ROI calculations.
        
        Args:
            token_address: Token contract address
            start_time: Start timestamp (seconds)
            end_time: End timestamp (seconds)
            resolution: Time resolution
            
        Returns:
            Dict with price history data
        """
        try:
            if not self.birdeye_api_key:
                return {
                    'success': False,
                    'error': 'Birdeye API not configured'
                }
            
            self.api_stats['birdeye']['calls'] += 1
            
            # Mock implementation - replace with actual Birdeye API call
            mock_prices = []
            current_time = start_time
            base_price = 0.001  # Mock base price
            
            while current_time <= end_time:
                # Simulate price movement
                price_change = (hash(str(current_time)) % 200 - 100) / 1000  # -0.1 to +0.1
                price = base_price * (1 + price_change)
                
                mock_prices.append({
                    'timestamp': current_time,
                    'value': price,
                    'volume': 1000 + (hash(str(current_time)) % 5000)
                })
                
                current_time += 3600  # 1 hour intervals
                base_price = price  # Use previous price as base
            
            self.api_stats['birdeye']['success'] += 1
            
            return {
                'success': True,
                'data': {
                    'items': mock_prices,
                    'token_address': token_address
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting token price history: {str(e)}")
            self.api_stats['birdeye']['errors'] += 1
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_token_info(self, token_address: str) -> Dict[str, Any]:
        """
        Get token information (market cap, etc.).
        
        Args:
            token_address: Token contract address
            
        Returns:
            Dict with token information
        """
        try:
            if not self.birdeye_api_key:
                return {
                    'success': False,
                    'error': 'Birdeye API not configured'
                }
            
            self.api_stats['birdeye']['calls'] += 1
            
            # Mock implementation
            mock_info = {
                'address': token_address,
                'symbol': 'MOCK',
                'name': 'Mock Token',
                'decimals': 6,
                'supply': 1000000,
                'market_cap': 50000,
                'price': 0.05,
                'volume_24h': 25000
            }
            
            self.api_stats['birdeye']['success'] += 1
            
            return {
                'success': True,
                'data': mock_info
            }
            
        except Exception as e:
            logger.error(f"Error getting token info: {str(e)}")
            self.api_stats['birdeye']['errors'] += 1
            return {
                'success': False,
                'error': str(e)
            }
    
    def make_rpc_call(self, method: str, params: List[Any]) -> Dict[str, Any]:
        """
        Make direct RPC call to Solana node.
        
        Args:
            method: RPC method name
            params: Method parameters
            
        Returns:
            Dict with RPC response
        """
        try:
            self.api_stats['rpc']['calls'] += 1
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params
            }
            
            response = requests.post(
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
    
    def health_check(self) -> bool:
        """
        Check if core APIs are accessible.
        
        Returns:
            bool: True if core APIs are working
        """
        try:
            # Check RPC
            rpc_result = self.make_rpc_call("getHealth", [])
            if not rpc_result.get('success'):
                logger.warning("RPC health check failed")
                return False
            
            # For mock implementation, always return True if we have at least one API key
            if self.cielo_api_key or self.birdeye_api_key or self.helius_api_key:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return False
    
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
        try:
            rpc_health = self.make_rpc_call("getHealth", [])
            if rpc_health.get('success'):
                status['api_status']['rpc'] = 'operational'
            else:
                status['api_status']['rpc'] = 'limited'
        except:
            status['api_status']['rpc'] = 'error'
        
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