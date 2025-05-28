"""
Zeus Analyzer - Core Wallet Analysis Engine
30-Day Analysis with Smart Token Sampling and Binary Decisions

Features:
- 30-day analysis window (vs Phoenix's 7-day)
- Minimum 6 unique token trades requirement
- Smart sampling: 5 initial → 10 if inconclusive
- Binary decision system (Follow Wallet/Follow Sells)
- New scoring system implementation
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("zeus.analyzer")

class ZeusAnalyzer:
    """Core wallet analysis engine with binary decision system."""
    
    def __init__(self, api_manager: Any, config: Dict[str, Any]):
        """
        Initialize Zeus analyzer.
        
        Args:
            api_manager: API manager for external calls
            config: Zeus configuration
        """
        self.api_manager = api_manager
        self.config = config
        
        # Analysis settings
        self.analysis_config = config.get('analysis', {})
        self.days_to_analyze = self.analysis_config.get('days_to_analyze', 30)
        self.min_unique_tokens = self.analysis_config.get('min_unique_tokens', 6)
        self.initial_token_sample = self.analysis_config.get('initial_token_sample', 5)
        self.max_token_sample = self.analysis_config.get('max_token_sample', 10)
        self.composite_score_threshold = self.analysis_config.get('composite_score_threshold', 65.0)
        self.exit_quality_threshold = self.analysis_config.get('exit_quality_threshold', 70.0)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Rate limiting for API calls
        self._last_api_call = 0
        self._api_call_lock = threading.Lock()
        
        logger.info(f"Zeus Analyzer initialized with {self.days_to_analyze}-day analysis window")
    
    def analyze_single_wallet(self, wallet_address: str) -> Dict[str, Any]:
        """
        Analyze a single wallet with binary decision system.
        
        Args:
            wallet_address: Wallet address to analyze
            
        Returns:
            Dict containing analysis results and binary decisions
        """
        logger.info(f"🔍 Starting Zeus analysis for {wallet_address[:8]}...{wallet_address[-4:]}")
        
        try:
            # Step 1: Get wallet trading data from Cielo Finance
            wallet_data = self._get_wallet_trading_data(wallet_address)
            
            if not wallet_data.get('success'):
                return {
                    'success': False,
                    'wallet_address': wallet_address,
                    'error': f"Failed to get wallet data: {wallet_data.get('error', 'Unknown error')}",
                    'error_type': 'DATA_FETCH_ERROR'
                }
            
            # Step 2: Get recent token swaps with 30-day window
            token_swaps = self._get_recent_token_swaps(wallet_address)
            
            if not token_swaps:
                return {
                    'success': False,
                    'wallet_address': wallet_address,
                    'error': 'No token swaps found in 30-day period',
                    'error_type': 'NO_SWAPS'
                }
            
            # Step 3: Check minimum token requirement
            unique_tokens = self._count_unique_tokens(token_swaps)
            
            if unique_tokens < self.min_unique_tokens:
                return {
                    'success': False,
                    'wallet_address': wallet_address,
                    'error': f'Insufficient unique tokens: {unique_tokens} < {self.min_unique_tokens}',
                    'error_type': 'INSUFFICIENT_VOLUME',
                    'unique_tokens_found': unique_tokens
                }
            
            # Step 4: Smart token sampling
            analysis_result = self._smart_token_analysis(wallet_address, token_swaps)
            
            if not analysis_result.get('success'):
                return analysis_result
            
            # Step 5: Calculate scores and binary decisions
            from zeus_scorer import ZeusScorer
            scorer = ZeusScorer(self.config)
            
            scoring_result = scorer.calculate_composite_score(analysis_result['token_analysis'])
            binary_decisions = self._make_binary_decisions(scoring_result, analysis_result)
            strategy_recommendation = self._generate_strategy_recommendation(
                binary_decisions, scoring_result, analysis_result
            )
            
            # Final result
            return {
                'success': True,
                'wallet_address': wallet_address,
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_days': self.days_to_analyze,
                'unique_tokens_traded': unique_tokens,
                'tokens_analyzed': analysis_result.get('tokens_analyzed', 0),
                'composite_score': scoring_result.get('composite_score', 0),
                'scoring_breakdown': scoring_result.get('component_scores', {}),
                'binary_decisions': binary_decisions,
                'strategy_recommendation': strategy_recommendation,
                'token_analysis': analysis_result.get('token_analysis', []),
                'wallet_data': wallet_data.get('data', {}),
                'conclusive_analysis': analysis_result.get('conclusive', True)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing wallet {wallet_address}: {str(e)}")
            return {
                'success': False,
                'wallet_address': wallet_address,
                'error': f'Analysis error: {str(e)}',
                'error_type': 'ANALYSIS_ERROR'
            }
    
    def analyze_wallets_batch(self, wallet_addresses: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple wallets in batch.
        
        Args:
            wallet_addresses: List of wallet addresses
            
        Returns:
            Dict containing batch analysis results
        """
        logger.info(f"🚀 Starting batch analysis of {len(wallet_addresses)} wallets")
        
        analyses = []
        failed_analyses = []
        
        for i, wallet_address in enumerate(wallet_addresses, 1):
            logger.info(f"Analyzing wallet {i}/{len(wallet_addresses)}: {wallet_address[:8]}...{wallet_address[-4:]}")
            
            try:
                result = self.analyze_single_wallet(wallet_address)
                
                if result.get('success'):
                    analyses.append(result)
                    score = result.get('composite_score', 0)
                    follow_wallet = result.get('binary_decisions', {}).get('follow_wallet', False)
                    logger.info(f"  ✅ Score: {score:.1f}/100, Follow: {'YES' if follow_wallet else 'NO'}")
                else:
                    failed_analyses.append(result)
                    logger.warning(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
                
                # Small delay between analyses
                if i < len(wallet_addresses):
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error analyzing wallet {wallet_address}: {str(e)}")
                failed_analyses.append({
                    'success': False,
                    'wallet_address': wallet_address,
                    'error': str(e),
                    'error_type': 'BATCH_ERROR'
                })
        
        return {
            'success': True,
            'batch_timestamp': datetime.now().isoformat(),
            'total_requested': len(wallet_addresses),
            'total_analyzed': len(wallet_addresses),
            'successful_analyses': len(analyses),
            'failed_analyses': len(failed_analyses),
            'analyses': analyses,
            'failed': failed_analyses
        }
    
    def _get_wallet_trading_data(self, wallet_address: str) -> Dict[str, Any]:
        """Get wallet trading data from Cielo Finance API."""
        try:
            if not self.api_manager.cielo_api:
                return {
                    'success': False,
                    'error': 'Cielo Finance API not configured'
                }
            
            # Get trading stats
            trading_stats = self.api_manager.get_wallet_trading_stats(wallet_address)
            
            if not trading_stats.get('success'):
                return {
                    'success': False,
                    'error': f"Cielo API error: {trading_stats.get('error', 'Unknown error')}"
                }
            
            return {
                'success': True,
                'data': trading_stats.get('data', {})
            }
            
        except Exception as e:
            logger.error(f"Error getting wallet data: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_recent_token_swaps(self, wallet_address: str) -> List[Dict[str, Any]]:
        """
        Get recent token swaps from the wallet within 30-day window.
        Uses RPC + enhanced transaction parsing.
        """
        try:
            # Calculate 30-day cutoff
            cutoff_date = datetime.now() - timedelta(days=self.days_to_analyze)
            cutoff_timestamp = int(cutoff_date.timestamp())
            
            # Get enhanced transactions if Helius is available
            if self.api_manager.helius_api:
                tx_result = self.api_manager.get_enhanced_transactions(wallet_address, limit=200)
                
                if tx_result.get('success'):
                    # Parse transactions for token swaps
                    swaps = self._parse_enhanced_transactions(tx_result.get('data', []), cutoff_timestamp)
                    if swaps:
                        logger.info(f"Found {len(swaps)} swaps via Helius enhanced transactions")
                        return swaps
            
            # Fallback to RPC-based transaction parsing
            logger.info("Using RPC fallback for transaction parsing")
            return self._get_swaps_via_rpc(wallet_address, cutoff_timestamp)
            
        except Exception as e:
            logger.error(f"Error getting recent swaps: {str(e)}")
            return []
    
    def _parse_enhanced_transactions(self, transactions: List[Dict[str, Any]], 
                                   cutoff_timestamp: int) -> List[Dict[str, Any]]:
        """Parse enhanced transactions from Helius into swap data."""
        swaps = []
        
        for tx in transactions:
            try:
                # Check transaction timestamp
                tx_timestamp = tx.get('timestamp', 0)
                if tx_timestamp < cutoff_timestamp:
                    continue
                
                # Look for swap events
                events = tx.get('events', {})
                if 'swap' in events:
                    swap_event = events['swap']
                    swap_data = self._extract_swap_from_event(swap_event, tx_timestamp)
                    if swap_data:
                        swaps.append(swap_data)
                
            except Exception as e:
                logger.debug(f"Error parsing transaction: {str(e)}")
                continue
        
        return swaps
    
    def _extract_swap_from_event(self, swap_event: Dict[str, Any], timestamp: int) -> Optional[Dict[str, Any]]:
        """Extract swap data from Helius swap event."""
        try:
            # Extract token information
            token_inputs = swap_event.get('tokenInputs', [])
            token_outputs = swap_event.get('tokenOutputs', [])
            
            # Find non-SOL token
            token_mint = None
            token_amount = 0
            sol_amount = 0
            swap_type = 'unknown'
            
            sol_mint = "So11111111111111111111111111111111111111112"
            
            # Check inputs and outputs
            for token_input in token_inputs:
                mint = token_input.get('mint', '')
                if mint == sol_mint:
                    sol_amount = float(token_input.get('rawTokenAmount', {}).get('tokenAmount', 0)) / 1e9
                    swap_type = 'buy'
                else:
                    token_mint = mint
                    token_amount = float(token_input.get('rawTokenAmount', {}).get('tokenAmount', 0))
            
            for token_output in token_outputs:
                mint = token_output.get('mint', '')
                if mint == sol_mint:
                    sol_amount = float(token_output.get('rawTokenAmount', {}).get('tokenAmount', 0)) / 1e9
                    swap_type = 'sell'
                else:
                    token_mint = mint
                    token_amount = float(token_output.get('rawTokenAmount', {}).get('tokenAmount', 0))
            
            if not token_mint or sol_amount <= 0:
                return None
            
            return {
                'token_mint': token_mint,
                'timestamp': timestamp,
                'type': swap_type,
                'token_amount': token_amount,
                'sol_amount': sol_amount,
                'signature': swap_event.get('signature', ''),
                'source': 'helius_enhanced'
            }
            
        except Exception as e:
            logger.debug(f"Error extracting swap from event: {str(e)}")
            return None
    
    def _get_swaps_via_rpc(self, wallet_address: str, cutoff_timestamp: int) -> List[Dict[str, Any]]:
        """Get swaps via direct RPC calls (fallback method)."""
        try:
            # This is a simplified implementation
            # In a full implementation, you'd parse RPC transaction data
            # For now, return empty list and rely on Cielo Finance data
            logger.warning("RPC-based swap parsing not fully implemented - relying on Cielo data")
            return []
            
        except Exception as e:
            logger.error(f"Error getting swaps via RPC: {str(e)}")
            return []
    
    def _count_unique_tokens(self, swaps: List[Dict[str, Any]]) -> int:
        """Count unique tokens from swap data."""
        unique_tokens = set()
        for swap in swaps:
            token_mint = swap.get('token_mint')
            if token_mint:
                unique_tokens.add(token_mint)
        return len(unique_tokens)
    
    def _smart_token_analysis(self, wallet_address: str, token_swaps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Smart token sampling: analyze 5 initial, expand to 10 if inconclusive.
        
        Returns analysis result with conclusive flag.
        """
        try:
            # Group swaps by token
            token_groups = defaultdict(list)
            for swap in token_swaps:
                token_mint = swap.get('token_mint')
                if token_mint:
                    token_groups[token_mint].append(swap)
            
            # Sort tokens by most recent activity
            sorted_tokens = sorted(
                token_groups.items(),
                key=lambda x: max(swap.get('timestamp', 0) for swap in x[1]),
                reverse=True
            )
            
            # Phase 1: Analyze initial 5 tokens
            initial_tokens = sorted_tokens[:self.initial_token_sample]
            initial_analysis = self._analyze_token_group(initial_tokens)
            
            # Check if analysis is conclusive
            is_conclusive = self._is_analysis_conclusive(initial_analysis)
            
            if is_conclusive or len(sorted_tokens) <= self.initial_token_sample:
                logger.info(f"Analysis conclusive with {len(initial_tokens)} tokens")
                return {
                    'success': True,
                    'token_analysis': initial_analysis,
                    'tokens_analyzed': len(initial_tokens),
                    'conclusive': True,
                    'analysis_phase': 'initial'
                }
            
            # Phase 2: Expand to 10 tokens for better analysis
            logger.info("Initial analysis inconclusive, expanding to 10 tokens")
            extended_tokens = sorted_tokens[:self.max_token_sample]
            extended_analysis = self._analyze_token_group(extended_tokens)
            
            return {
                'success': True,
                'token_analysis': extended_analysis,
                'tokens_analyzed': len(extended_tokens),
                'conclusive': False,  # Required expansion
                'analysis_phase': 'extended'
            }
            
        except Exception as e:
            logger.error(f"Error in smart token analysis: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_token_group(self, token_groups: List[Tuple[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        """Analyze a group of tokens and their swaps."""
        analyzed_tokens = []
        
        for token_mint, swaps in token_groups:
            try:
                # Analyze this token's trading pattern
                token_analysis = self._analyze_single_token(token_mint, swaps)
                if token_analysis:
                    analyzed_tokens.append(token_analysis)
                    
            except Exception as e:
                logger.debug(f"Error analyzing token {token_mint}: {str(e)}")
                continue
        
        return analyzed_tokens
    
    def _analyze_single_token(self, token_mint: str, swaps: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze trading pattern for a single token."""
        try:
            # Sort swaps by timestamp
            sorted_swaps = sorted(swaps, key=lambda x: x.get('timestamp', 0))
            
            # Basic metrics
            total_swaps = len(sorted_swaps)
            buy_swaps = [s for s in sorted_swaps if s.get('type') == 'buy']
            sell_swaps = [s for s in sorted_swaps if s.get('type') == 'sell']
            
            # Calculate basic performance
            total_sol_in = sum(s.get('sol_amount', 0) for s in buy_swaps)
            total_sol_out = sum(s.get('sol_amount', 0) for s in sell_swaps)
            
            # Estimate ROI (simplified)
            roi_percent = 0
            if total_sol_in > 0:
                if total_sol_out > 0:
                    # Completed trade
                    roi_percent = ((total_sol_out / total_sol_in) - 1) * 100
                else:
                    # Open position - estimate current value
                    roi_percent = 0  # Neutral for open positions
            
            # Calculate hold time
            if len(sorted_swaps) >= 2:
                first_timestamp = sorted_swaps[0].get('timestamp', 0)
                last_timestamp = sorted_swaps[-1].get('timestamp', 0)
                hold_time_hours = (last_timestamp - first_timestamp) / 3600
            else:
                hold_time_hours = 0
            
            # Determine trade status
            if sell_swaps:
                trade_status = 'completed'
            else:
                trade_status = 'open'
            
            # Get token price data for better analysis
            token_price_data = self._get_token_price_data(token_mint, sorted_swaps[0].get('timestamp'))
            
            return {
                'token_mint': token_mint,
                'total_swaps': total_swaps,
                'buy_count': len(buy_swaps),
                'sell_count': len(sell_swaps),
                'total_sol_in': total_sol_in,
                'total_sol_out': total_sol_out,
                'roi_percent': roi_percent,
                'hold_time_hours': hold_time_hours,
                'trade_status': trade_status,
                'first_timestamp': sorted_swaps[0].get('timestamp'),
                'last_timestamp': sorted_swaps[-1].get('timestamp'),
                'price_data': token_price_data,
                'swaps': sorted_swaps
            }
            
        except Exception as e:
            logger.debug(f"Error analyzing token {token_mint}: {str(e)}")
            return None
    
    def _get_token_price_data(self, token_mint: str, start_timestamp: int) -> Dict[str, Any]:
        """Get token price data for ROI calculation."""
        try:
            # Use API manager to get token price history
            start_time = datetime.fromtimestamp(start_timestamp)
            end_time = datetime.now()
            
            price_history = self.api_manager.get_token_price_history(
                token_mint,
                int(start_time.timestamp()),
                int(end_time.timestamp()),
                "1h"
            )
            
            if price_history.get('success'):
                items = price_history.get('data', {}).get('items', [])
                if items:
                    initial_price = items[0].get('value', 0)
                    current_price = items[-1].get('value', 0)
                    max_price = max(item.get('value', 0) for item in items)
                    
                    return {
                        'initial_price': initial_price,
                        'current_price': current_price,
                        'max_price': max_price,
                        'price_available': True
                    }
            
            return {'price_available': False}
            
        except Exception as e:
            logger.debug(f"Error getting price data for {token_mint}: {str(e)}")
            return {'price_available': False}
    
    def _is_analysis_conclusive(self, token_analysis: List[Dict[str, Any]]) -> bool:
        """
        Check if analysis is conclusive enough for binary decision.
        
        Criteria for conclusive analysis:
        - Clear win/loss pattern
        - Sufficient trade volume
        - Consistent behavior pattern
        """
        if len(token_analysis) < 3:
            return False
        
        # Check for clear patterns
        completed_trades = [t for t in token_analysis if t.get('trade_status') == 'completed']
        
        if len(completed_trades) < 2:
            return False  # Need more completed trades
        
        # Check ROI distribution
        rois = [t.get('roi_percent', 0) for t in completed_trades]
        
        # Clear win pattern
        if sum(1 for roi in rois if roi > 50) >= len(rois) * 0.6:
            return True
        
        # Clear loss pattern
        if sum(1 for roi in rois if roi < -25) >= len(rois) * 0.6:
            return True
        
        # Mixed results - need more data
        return False
    
    def _make_binary_decisions(self, scoring_result: Dict[str, Any], 
                             analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Make binary decisions based on scoring and analysis."""
        try:
            composite_score = scoring_result.get('composite_score', 0)
            token_analysis = analysis_result.get('token_analysis', [])
            
            # Decision 1: Follow Wallet
            follow_wallet = self._decide_follow_wallet(composite_score, scoring_result, token_analysis)
            
            # Decision 2: Follow Sells (only if following wallet)
            follow_sells = False
            if follow_wallet:
                follow_sells = self._decide_follow_sells(scoring_result, token_analysis)
            
            return {
                'follow_wallet': follow_wallet,
                'follow_sells': follow_sells,
                'composite_score': composite_score,
                'decision_reasoning': self._get_decision_reasoning(
                    follow_wallet, follow_sells, composite_score, scoring_result
                )
            }
            
        except Exception as e:
            logger.error(f"Error making binary decisions: {str(e)}")
            return {
                'follow_wallet': False,
                'follow_sells': False,
                'composite_score': 0,
                'decision_reasoning': f"Error in decision making: {str(e)}"
            }
    
    def _decide_follow_wallet(self, composite_score: float, scoring_result: Dict[str, Any], 
                            token_analysis: List[Dict[str, Any]]) -> bool:
        """
        Decide whether to follow wallet based on composite score and volume.
        
        Criteria:
        - Composite score >= 65
        - Minimum token volume met (checked earlier)
        - No bot behavior detected
        """
        # Score threshold
        if composite_score < self.composite_score_threshold:
            return False
        
        # Check for bot behavior (same-block trades)
        bot_behavior_score = scoring_result.get('component_scores', {}).get('bot_behavior_penalty', 0)
        if bot_behavior_score < -10:  # Significant bot behavior penalty
            return False
        
        # Check trading discipline
        discipline_score = scoring_result.get('component_scores', {}).get('trading_discipline', 0)
        if discipline_score < 5:  # Very poor discipline
            return False
        
        return True
    
    def _decide_follow_sells(self, scoring_result: Dict[str, Any], 
                           token_analysis: List[Dict[str, Any]]) -> bool:
        """
        Decide whether to follow sells based on exit quality.
        
        Criteria:
        - Exit quality >= 70%
        - Low dump rate (<25%)
        - Consistent profit capture (>60%)
        """
        try:
            # Calculate exit quality metrics
            completed_trades = [t for t in token_analysis if t.get('trade_status') == 'completed']
            
            if len(completed_trades) < 2:
                return False  # Not enough exit data
            
            # Check profitable exits
            profitable_exits = [t for t in completed_trades if t.get('roi_percent', 0) > 0]
            profit_rate = len(profitable_exits) / len(completed_trades) if completed_trades else 0
            
            if profit_rate < 0.6:  # Less than 60% profitable exits
                return False
            
            # Check for dump behavior (very quick sells)
            quick_sells = [t for t in completed_trades if t.get('hold_time_hours', 0) < 0.1]  # <6 minutes
            dump_rate = len(quick_sells) / len(completed_trades) if completed_trades else 0
            
            if dump_rate > 0.25:  # More than 25% dump trades
                return False
            
            # Check average exit quality
            avg_roi = sum(t.get('roi_percent', 0) for t in profitable_exits) / len(profitable_exits) if profitable_exits else 0
            
            if avg_roi < 30:  # Less than 30% average profit on wins
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error deciding follow sells: {str(e)}")
            return False
    
    def _get_decision_reasoning(self, follow_wallet: bool, follow_sells: bool, 
                              composite_score: float, scoring_result: Dict[str, Any]) -> str:
        """Generate reasoning for binary decisions."""
        reasoning_parts = []
        
        # Follow wallet reasoning
        if follow_wallet:
            reasoning_parts.append(f"Follow Wallet: YES (Score: {composite_score:.1f} ≥ {self.composite_score_threshold})")
        else:
            if composite_score < self.composite_score_threshold:
                reasoning_parts.append(f"Follow Wallet: NO (Score: {composite_score:.1f} < {self.composite_score_threshold})")
            else:
                reasoning_parts.append("Follow Wallet: NO (Failed other criteria)")
        
        # Follow sells reasoning
        if follow_wallet:
            if follow_sells:
                reasoning_parts.append("Follow Sells: YES (Good exit discipline)")
            else:
                reasoning_parts.append("Follow Sells: NO (Poor exit quality or dump behavior)")
        else:
            reasoning_parts.append("Follow Sells: NO (Not following wallet)")
        
        return " | ".join(reasoning_parts)
    
    def _generate_strategy_recommendation(self, binary_decisions: Dict[str, Any], 
                                        scoring_result: Dict[str, Any],
                                        analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate TP/SL strategy recommendation based on binary decisions."""
        try:
            follow_wallet = binary_decisions.get('follow_wallet', False)
            follow_sells = binary_decisions.get('follow_sells', False)
            composite_score = binary_decisions.get('composite_score', 0)
            
            if not follow_wallet:
                return {
                    'copy_entries': False,
                    'copy_exits': False,
                    'tp1_percent': 0,
                    'tp2_percent': 0,
                    'tp3_percent': 0,
                    'stop_loss_percent': -35,
                    'position_size_sol': '0',
                    'reasoning': 'Do not follow - insufficient score or volume'
                }
            
            # Determine trader pattern
            token_analysis = analysis_result.get('token_analysis', [])
            trader_pattern = self._identify_trader_pattern(token_analysis)
            
            if follow_sells:
                # Mirror their strategy with safety buffer
                avg_exit_roi = self._calculate_average_exit_roi(token_analysis)
                
                return {
                    'copy_entries': True,
                    'copy_exits': True,
                    'tp1_percent': max(50, int(avg_exit_roi * 0.8)),  # 80% of their average
                    'tp2_percent': max(100, int(avg_exit_roi * 1.5)),  # 150% of their average
                    'tp3_percent': max(200, int(avg_exit_roi * 3.0)),  # 300% of their average
                    'stop_loss_percent': -35,
                    'position_size_sol': self._recommend_position_size(token_analysis),
                    'reasoning': f'Mirror strategy - they have good exit discipline (Pattern: {trader_pattern})'
                }
            else:
                # Custom strategy based on pattern
                if trader_pattern == 'gem_hunter':
                    return {
                        'copy_entries': True,
                        'copy_exits': False,
                        'tp1_percent': 100,
                        'tp2_percent': 300,
                        'tp3_percent': 800,
                        'stop_loss_percent': -40,
                        'position_size_sol': self._recommend_position_size(token_analysis),
                        'reasoning': 'Gem hunter - they find good tokens but exit too early'
                    }
                elif trader_pattern == 'consistent_scalper':
                    return {
                        'copy_entries': True,
                        'copy_exits': False,
                        'tp1_percent': 50,
                        'tp2_percent': 100,
                        'tp3_percent': 200,
                        'stop_loss_percent': -25,
                        'position_size_sol': self._recommend_position_size(token_analysis),
                        'reasoning': 'Consistent scalper - steady but exits early'
                    }
                elif trader_pattern == 'volatile_trader':
                    return {
                        'copy_entries': True,
                        'copy_exits': False,
                        'tp1_percent': 60,
                        'tp2_percent': 150,
                        'tp3_percent': 400,
                        'stop_loss_percent': -30,
                        'position_size_sol': self._recommend_position_size(token_analysis),
                        'reasoning': 'Volatile trader - account for volatility'
                    }
                else:
                    # Mixed strategy
                    return {
                        'copy_entries': True,
                        'copy_exits': False,
                        'tp1_percent': 75,
                        'tp2_percent': 200,
                        'tp3_percent': 500,
                        'stop_loss_percent': -35,
                        'position_size_sol': self._recommend_position_size(token_analysis),
                        'reasoning': 'Mixed pattern - balanced approach'
                    }
            
        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            return {
                'copy_entries': False,
                'copy_exits': False,
                'tp1_percent': 0,
                'tp2_percent': 0,
                'tp3_percent': 0,
                'stop_loss_percent': -35,
                'position_size_sol': '0',
                'reasoning': f'Error generating strategy: {str(e)}'
            }
    
    def _identify_trader_pattern(self, token_analysis: List[Dict[str, Any]]) -> str:
        """Identify trader pattern from token analysis."""
        if not token_analysis:
            return 'unknown'
        
        completed_trades = [t for t in token_analysis if t.get('trade_status') == 'completed']
        
        if len(completed_trades) < 2:
            return 'insufficient_data'
        
        # Calculate pattern metrics
        rois = [t.get('roi_percent', 0) for t in completed_trades]
        hold_times = [t.get('hold_time_hours', 0) for t in completed_trades]
        
        avg_roi = sum(rois) / len(rois)
        avg_hold_time = sum(hold_times) / len(hold_times)
        roi_std = (sum((roi - avg_roi) ** 2 for roi in rois) / len(rois)) ** 0.5
        
        # High variance, high upside = gem hunter
        if roi_std > 100 and max(rois) > 200:
            return 'gem_hunter'
        
        # Low variance, consistent profits = scalper
        if roi_std < 50 and avg_roi > 20:
            return 'consistent_scalper'
        
        # High variance = volatile trader
        if roi_std > 80:
            return 'volatile_trader'
        
        return 'mixed_strategy'
    
    def _calculate_average_exit_roi(self, token_analysis: List[Dict[str, Any]]) -> float:
        """Calculate average exit ROI for profitable trades."""
        completed_trades = [t for t in token_analysis if t.get('trade_status') == 'completed']
        profitable_trades = [t for t in completed_trades if t.get('roi_percent', 0) > 0]
        
        if not profitable_trades:
            return 50  # Default
        
        return sum(t.get('roi_percent', 0) for t in profitable_trades) / len(profitable_trades)
    
    def _recommend_position_size(self, token_analysis: List[Dict[str, Any]]) -> str:
        """Recommend position size based on their typical bet size."""
        if not token_analysis:
            return '1-5'
        
        # Calculate their average bet size
        bet_sizes = []
        for token in token_analysis:
            total_sol_in = token.get('total_sol_in', 0)
            if total_sol_in > 0:
                bet_sizes.append(total_sol_in)
        
        if not bet_sizes:
            return '1-5'
        
        avg_bet_size = sum(bet_sizes) / len(bet_sizes)
        
        if avg_bet_size < 1:
            return '0.5-2'
        elif avg_bet_size < 5:
            return '1-5'
        elif avg_bet_size < 10:
            return '2-10'
        else:
            return '5-15'  # Cap recommendation
    
    def __del__(self):
        """Cleanup thread pool."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)