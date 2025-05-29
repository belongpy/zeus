"""
Zeus Scorer - Work with Exact Data for Scoring
FINAL FIX:
- Scoring system works with token analysis data (for scoring logic)
- Export system extracts exact Cielo field values (for CSV output)
- Keep scoring thresholds and logic unchanged
- Focus on making scoring reliable with available data
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("zeus.scorer")

class ZeusScorer:
    """Implements Zeus scoring system that works with token analysis data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize scorer with configuration."""
        self.config = config
        self.analysis_config = config.get('analysis', {})
        
        # Scoring weights
        self.component_weights = {
            'risk_adjusted_performance': 0.30,
            'distribution_quality': 0.25,
            'trading_discipline': 0.20,
            'market_impact_awareness': 0.15,
            'consistency_reliability': 0.10
        }
        
        logger.info("Zeus Scorer initialized for token analysis data processing")
    
    def calculate_composite_score(self, token_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate composite score from token analysis data."""
        try:
            if not token_analysis:
                return self._get_zero_score("No token analysis data")
            
            logger.info(f"📊 Calculating composite score for {len(token_analysis)} tokens")
            
            # Extract metrics from token analysis data
            metrics = self._extract_token_analysis_metrics(token_analysis)
            
            if not metrics:
                return self._get_zero_score("Failed to extract metrics from token analysis")
            
            logger.info(f"Extracted token analysis metrics: {len(metrics)} fields")
            
            # Check volume qualifier first
            volume_qualifier = self._calculate_volume_qualifier(metrics)
            if volume_qualifier['disqualified']:
                return self._get_zero_score(volume_qualifier['reason'])
            
            logger.info(f"Volume qualifier: {volume_qualifier['tier']} (×{volume_qualifier['multiplier']})")
            
            # Calculate component scores
            component_scores = {}
            
            # 1. Risk-Adjusted Performance (30%)
            component_scores['risk_adjusted_performance'] = self._calculate_risk_adjusted_score(metrics)
            
            # 2. Distribution Quality (25%)
            component_scores['distribution_quality'] = self._calculate_distribution_score(metrics)
            
            # 3. Trading Discipline (20%)
            component_scores['trading_discipline'] = self._calculate_discipline_score(metrics)
            
            # 4. Market Impact Awareness (15%)
            component_scores['market_impact_awareness'] = self._calculate_market_impact_score(metrics)
            
            # 5. Consistency & Reliability (10%)
            component_scores['consistency_reliability'] = self._calculate_consistency_score(metrics)
            
            # Log component scores
            logger.info("Component Scores (0-1 scale):")
            for component, score in component_scores.items():
                logger.info(f"  {component}: {score:.3f}")
            
            # Apply volume qualifier multiplier
            volume_multiplier = volume_qualifier['multiplier']
            
            # Calculate weighted composite score
            composite_score = 0
            for component, weight in self.component_weights.items():
                component_score = component_scores.get(component, 0)
                weighted_score = component_score * weight * 100  # Convert to 0-100 scale
                composite_score += weighted_score
            
            # Apply volume qualifier
            composite_score *= volume_multiplier
            
            # Cap at 100
            composite_score = min(100, max(0, composite_score))
            
            logger.info(f"Final composite score: {composite_score:.1f}/100")
            
            return {
                'composite_score': round(composite_score, 1),
                'component_scores': {
                    'risk_adjusted_score': round(component_scores['risk_adjusted_performance'] * 30, 1),
                    'distribution_score': round(component_scores['distribution_quality'] * 25, 1),
                    'discipline_score': round(component_scores['trading_discipline'] * 20, 1),
                    'market_impact_score': round(component_scores['market_impact_awareness'] * 15, 1),
                    'consistency_score': round(component_scores['consistency_reliability'] * 10, 1)
                },
                'volume_qualifier': volume_qualifier,
                'metrics_used': metrics,
                'total_tokens_analyzed': len(token_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {str(e)}")
            return self._get_zero_score(f"Calculation error: {str(e)}")
    
    def _extract_token_analysis_metrics(self, token_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metrics from token analysis data for scoring."""
        try:
            logger.info(f"Extracting metrics from {len(token_analysis)} token analyses")
            
            # Filter completed trades for most metrics
            completed_trades = [t for t in token_analysis if t.get('trade_status') == 'completed']
            all_trades = token_analysis
            
            logger.info(f"Found {len(completed_trades)} completed trades out of {len(all_trades)} total")
            
            # Basic counts
            total_tokens = len(all_trades)
            completed_count = len(completed_trades)
            
            if completed_count == 0:
                logger.warning("No completed trades found - using limited metrics")
                return {
                    'total_tokens': total_tokens,
                    'completed_trades': 0,
                    'avg_roi': 0,
                    'roi_std': 0,
                    'win_rate': 0,
                    'active_days': 1,
                    'avg_bet_size_sol': 1,
                    'same_block_rate': 0,
                    'rois': []
                }
            
            # ROI metrics from completed trades
            rois = []
            for trade in completed_trades:
                roi = trade.get('roi_percent', 0)
                if roi is not None and isinstance(roi, (int, float)):
                    rois.append(float(roi))
            
            if not rois:
                logger.warning("No valid ROI data found")
                rois = [0]
            
            logger.info(f"Extracted {len(rois)} ROI values: min={min(rois):.1f}%, max={max(rois):.1f}%")
            
            # Calculate ROI statistics
            avg_roi = float(np.mean(rois))
            roi_std = float(np.std(rois)) if len(rois) > 1 else 0
            max_roi = float(max(rois))
            min_roi = float(min(rois))
            
            # Hold time metrics
            hold_times = []
            for trade in completed_trades:
                hold_time = trade.get('hold_time_hours', 0)
                if hold_time is not None and isinstance(hold_time, (int, float)) and hold_time > 0:
                    hold_times.append(float(hold_time))
            
            avg_hold_time = float(np.mean(hold_times)) if hold_times else 24.0
            
            # Bet size metrics
            bet_sizes = []
            for trade in all_trades:
                bet_size = trade.get('total_sol_in', 0)
                if bet_size is not None and isinstance(bet_size, (int, float)) and bet_size > 0:
                    bet_sizes.append(float(bet_size))
            
            avg_bet_size = float(np.mean(bet_sizes)) if bet_sizes else 5.0
            
            # Distribution metrics
            moonshots = sum(1 for roi in rois if roi >= 400)  # 5x+
            big_wins = sum(1 for roi in rois if 100 <= roi < 400)  # 2x-5x
            small_wins = sum(1 for roi in rois if 0 < roi < 100)  # Profitable <2x
            small_losses = sum(1 for roi in rois if -50 < roi <= 0)  # Small losses
            heavy_losses = sum(1 for roi in rois if roi <= -50)  # Heavy losses
            
            # Calculate distribution percentages
            total_completed = len(rois)
            moonshot_rate = (moonshots / total_completed * 100) if total_completed > 0 else 0
            big_win_rate = (big_wins / total_completed * 100) if total_completed > 0 else 0
            heavy_loss_rate = (heavy_losses / total_completed * 100) if total_completed > 0 else 0
            
            # Win rate
            wins = sum(1 for roi in rois if roi > 0)
            win_rate = (wins / total_completed * 100) if total_completed > 0 else 0
            
            # Same-block detection (flipper behavior)
            same_block_trades = 0
            for trade in all_trades:
                hold_time = trade.get('hold_time_hours', 24)
                if hold_time < 0.1:  # Less than 6 minutes
                    same_block_trades += 1
            
            same_block_rate = (same_block_trades / total_tokens * 100) if total_tokens > 0 else 0
            
            # Loss cutting behavior
            quick_cut_losses = sum(1 for t in completed_trades 
                                 if t.get('roi_percent', 0) < -10 and t.get('hold_time_hours', 24) < 4)
            slow_cut_losses = sum(1 for t in completed_trades 
                                if t.get('roi_percent', 0) < -10 and t.get('hold_time_hours', 0) > 24 * 7)
            
            # Activity pattern
            active_days = 30  # Default to full analysis period
            timestamps = []
            for trade in all_trades:
                first_ts = trade.get('first_timestamp', 0)
                last_ts = trade.get('last_timestamp', 0)
                if first_ts and last_ts:
                    timestamps.extend([first_ts, last_ts])
            
            if len(timestamps) >= 2:
                earliest = min(timestamps)
                latest = max(timestamps)
                active_days = max(1, (latest - earliest) / 86400)  # Convert to days
            
            logger.info(f"Calculated metrics: win_rate={win_rate:.1f}%, avg_roi={avg_roi:.1f}%, moonshot_rate={moonshot_rate:.1f}%")
            
            return {
                'total_tokens': total_tokens,
                'completed_trades': completed_count,
                'avg_roi': avg_roi,
                'roi_std': roi_std,
                'max_roi': max_roi,
                'min_roi': min_roi,
                'avg_hold_time_hours': avg_hold_time,
                'avg_bet_size_sol': avg_bet_size,
                'moonshot_rate': moonshot_rate,
                'big_win_rate': big_win_rate,
                'heavy_loss_rate': heavy_loss_rate,
                'win_rate': win_rate,
                'same_block_rate': same_block_rate,
                'quick_cut_losses': quick_cut_losses,
                'slow_cut_losses': slow_cut_losses,
                'active_days': active_days,
                'rois': rois
            }
            
        except Exception as e:
            logger.error(f"Error extracting token analysis metrics: {str(e)}")
            return {}
    
    def _calculate_volume_qualifier(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate volume qualifier score and check disqualification."""
        total_tokens = metrics.get('total_tokens', 0)
        
        logger.info(f"Volume qualifier check: {total_tokens} tokens")
        
        if total_tokens >= 6:
            return {
                'tokens': total_tokens,
                'qualifier_points': 100,
                'multiplier': 1.0,
                'disqualified': False,
                'tier': 'baseline'
            }
        elif total_tokens >= 4:
            return {
                'tokens': total_tokens,
                'qualifier_points': 80,
                'multiplier': 0.8,
                'disqualified': False,
                'tier': 'emerging'
            }
        elif total_tokens >= 2:
            return {
                'tokens': total_tokens,
                'qualifier_points': 60,
                'multiplier': 0.6,
                'disqualified': False,
                'tier': 'very_new'
            }
        else:
            return {
                'tokens': total_tokens,
                'qualifier_points': 0,
                'multiplier': 0.0,
                'disqualified': True,
                'tier': 'insufficient',
                'reason': f'Insufficient tokens: {total_tokens} < 2 minimum'
            }
    
    def _calculate_risk_adjusted_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate Risk-Adjusted Performance Score (30% weight)."""
        try:
            # Average ROI component (60% of this score)
            avg_roi = metrics.get('avg_roi', 0)
            if avg_roi > 100:
                roi_score = 1.0
            elif avg_roi >= 50:
                roi_score = 0.8
            elif avg_roi >= 25:
                roi_score = 0.6
            elif avg_roi >= 0:
                roi_score = 0.4
            else:
                roi_score = 0.2  # Negative average ROI
            
            # Standard Deviation component (25% of this score) 
            roi_std = metrics.get('roi_std', 0)
            if roi_std < 50:
                std_score = 1.0
            elif roi_std < 100:
                std_score = 0.9
            elif roi_std < 200:
                std_score = 0.8
            else:
                std_score = 0.7
            
            # Win rate component (15% of this score)
            win_rate = metrics.get('win_rate', 0)
            if win_rate > 70:
                win_score = 1.0
            elif win_rate > 50:
                win_score = 0.8
            elif win_rate > 30:
                win_score = 0.6
            else:
                win_score = 0.4
            
            # Combine components
            risk_adjusted_score = (
                roi_score * 0.60 +
                std_score * 0.25 +
                win_score * 0.15
            )
            
            logger.debug(f"Risk-adjusted: avg_roi={roi_score:.2f}, std={std_score:.2f}, win={win_score:.2f} → {risk_adjusted_score:.3f}")
            
            return min(1.0, max(0.0, risk_adjusted_score))
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted score: {str(e)}")
            return 0.0
    
    def _calculate_distribution_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate Distribution Quality Score (25% weight)."""
        try:
            # Moonshot Rate component (50% of this score)
            moonshot_rate = metrics.get('moonshot_rate', 0)
            if moonshot_rate > 15:
                moonshot_score = 1.0
            elif moonshot_rate >= 10:
                moonshot_score = 0.8
            elif moonshot_rate >= 5:
                moonshot_score = 0.6
            elif moonshot_rate > 0:
                moonshot_score = 0.4
            else:
                moonshot_score = 0.2
            
            # Big Win Distribution component (30% of this score)
            big_win_rate = metrics.get('big_win_rate', 0)
            if big_win_rate > 30:
                big_win_score = 1.0
            elif big_win_rate >= 20:
                big_win_score = 0.8
            elif big_win_rate >= 10:
                big_win_score = 0.6
            elif big_win_rate > 0:
                big_win_score = 0.4
            else:
                big_win_score = 0.2
            
            # Loss Distribution component (20% of this score)
            heavy_loss_rate = metrics.get('heavy_loss_rate', 0)
            if heavy_loss_rate < 10:
                loss_score = 1.0
            elif heavy_loss_rate < 20:
                loss_score = 0.8
            elif heavy_loss_rate < 30:
                loss_score = 0.6
            elif heavy_loss_rate < 50:
                loss_score = 0.4
            else:
                loss_score = 0.2
            
            # Combine components
            distribution_score = (
                moonshot_score * 0.50 +
                big_win_score * 0.30 +
                loss_score * 0.20
            )
            
            logger.debug(f"Distribution: moonshot={moonshot_score:.2f}, big_win={big_win_score:.2f}, loss={loss_score:.2f} → {distribution_score:.3f}")
            
            return min(1.0, max(0.0, distribution_score))
            
        except Exception as e:
            logger.error(f"Error calculating distribution score: {str(e)}")
            return 0.0
    
    def _calculate_discipline_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate Trading Discipline Score (20% weight)."""
        try:
            # Loss Management component (40% of this score)
            quick_cuts = metrics.get('quick_cut_losses', 0)
            slow_cuts = metrics.get('slow_cut_losses', 0)
            total_losses = quick_cuts + slow_cuts
            
            if total_losses == 0:
                loss_mgmt_score = 0.8  # Neutral - no loss data
            else:
                quick_cut_rate = quick_cuts / total_losses
                if quick_cut_rate > 0.8:
                    loss_mgmt_score = 1.0
                elif quick_cut_rate > 0.6:
                    loss_mgmt_score = 0.8
                elif quick_cut_rate > 0.4:
                    loss_mgmt_score = 0.6
                elif quick_cut_rate > 0.2:
                    loss_mgmt_score = 0.4
                else:
                    loss_mgmt_score = 0.2
            
            # Exit Behavior component (35% of this score)
            win_rate = metrics.get('win_rate', 0)
            if win_rate > 60:
                exit_score = 1.0
            elif win_rate > 45:
                exit_score = 0.8
            elif win_rate > 30:
                exit_score = 0.6
            elif win_rate > 15:
                exit_score = 0.4
            else:
                exit_score = 0.2
            
            # Fast Sell Detection component (25% of this score)
            same_block_rate = metrics.get('same_block_rate', 0)
            if same_block_rate < 5:
                fast_sell_score = 1.0
            elif same_block_rate < 10:
                fast_sell_score = 0.8
            elif same_block_rate < 20:
                fast_sell_score = 0.6
            elif same_block_rate < 40:
                fast_sell_score = 0.3
            else:
                fast_sell_score = 0.0  # Heavy penalty for flipper behavior
            
            # Combine components
            discipline_score = (
                loss_mgmt_score * 0.40 +
                exit_score * 0.35 +
                fast_sell_score * 0.25
            )
            
            logger.debug(f"Discipline: loss_mgmt={loss_mgmt_score:.2f}, exit={exit_score:.2f}, fast_sell={fast_sell_score:.2f} → {discipline_score:.3f}")
            
            return min(1.0, max(0.0, discipline_score))
            
        except Exception as e:
            logger.error(f"Error calculating discipline score: {str(e)}")
            return 0.0
    
    def _calculate_market_impact_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate Market Impact Awareness Score (15% weight)."""
        try:
            # Bet Sizing component (60% of this score)
            avg_bet_size = metrics.get('avg_bet_size_sol', 0)
            if 1 <= avg_bet_size <= 10:
                bet_size_score = 1.0  # Optimal range
            elif 0.5 <= avg_bet_size < 1:
                bet_size_score = 0.9  # Good but small
            elif 10 < avg_bet_size <= 20:
                bet_size_score = 0.6  # Large but manageable
            elif avg_bet_size > 20:
                bet_size_score = 0.4  # Too large - whale behavior
            elif avg_bet_size > 0:
                bet_size_score = 0.7  # Very small bets but positive
            else:
                bet_size_score = 0.5  # No bet size data
            
            # Consistency component (40% of this score)
            roi_std = metrics.get('roi_std', 100)  # Use ROI std as consistency measure
            if roi_std < 50:
                consistency_score = 1.0
            elif roi_std < 100:
                consistency_score = 0.8
            elif roi_std < 200:
                consistency_score = 0.6
            else:
                consistency_score = 0.4
            
            # Combine components
            market_impact_score = (
                bet_size_score * 0.60 +
                consistency_score * 0.40
            )
            
            logger.debug(f"Market impact: bet_size={bet_size_score:.2f}, consistency={consistency_score:.2f} → {market_impact_score:.3f}")
            
            return min(1.0, max(0.0, market_impact_score))
            
        except Exception as e:
            logger.error(f"Error calculating market impact score: {str(e)}")
            return 0.0
    
    def _calculate_consistency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate Consistency & Reliability Score (10% weight)."""
        try:
            # Activity Pattern component (70% of this score)
            active_days = metrics.get('active_days', 0)
            if active_days > 25:
                activity_score = 1.0
            elif active_days > 15:
                activity_score = 0.8
            elif active_days > 7:
                activity_score = 0.6
            elif active_days > 3:
                activity_score = 0.4
            elif active_days > 0:
                activity_score = 0.2
            else:
                activity_score = 0.1
            
            # Red Flags component (30% of this score)
            same_block_rate = metrics.get('same_block_rate', 0)
            
            if same_block_rate > 50:
                red_flag_score = 0.0  # Likely bot
            elif same_block_rate > 30:
                red_flag_score = 0.3  # High bot suspicion
            elif same_block_rate > 15:
                red_flag_score = 0.6  # Some bot behavior
            elif same_block_rate > 5:
                red_flag_score = 0.8  # Minimal bot behavior
            else:
                red_flag_score = 1.0  # Clean behavior
            
            # Combine components
            consistency_score = (
                activity_score * 0.70 +
                red_flag_score * 0.30
            )
            
            logger.debug(f"Consistency: activity={activity_score:.2f}, red_flags={red_flag_score:.2f} → {consistency_score:.3f}")
            
            return min(1.0, max(0.0, consistency_score))
            
        except Exception as e:
            logger.error(f"Error calculating consistency score: {str(e)}")
            return 0.0
    
    def _get_zero_score(self, reason: str) -> Dict[str, Any]:
        """Return zero score with reason."""
        return {
            'composite_score': 0.0,
            'component_scores': {
                'risk_adjusted_score': 0.0,
                'distribution_score': 0.0,
                'discipline_score': 0.0,
                'market_impact_score': 0.0,
                'consistency_score': 0.0
            },
            'volume_qualifier': {
                'disqualified': True,
                'reason': reason,
                'tokens': 0,
                'tier': 'insufficient'
            },
            'total_tokens_analyzed': 0
        }
    
    def get_score_explanation(self, scoring_result: Dict[str, Any]) -> str:
        """Generate human-readable explanation of the score."""
        try:
            composite_score = scoring_result.get('composite_score', 0)
            component_scores = scoring_result.get('component_scores', {})
            volume_qualifier = scoring_result.get('volume_qualifier', {})
            
            explanation = []
            
            # Overall score
            if composite_score >= 80:
                explanation.append(f"🟢 EXCELLENT (Score: {composite_score}/100)")
            elif composite_score >= 65:
                explanation.append(f"🟡 GOOD (Score: {composite_score}/100)")
            elif composite_score >= 50:
                explanation.append(f"🟠 AVERAGE (Score: {composite_score}/100)")
            else:
                explanation.append(f"🔴 POOR (Score: {composite_score}/100)")
            
            # Volume qualifier
            if volume_qualifier.get('disqualified'):
                explanation.append(f"❌ DISQUALIFIED: {volume_qualifier.get('reason', 'Unknown')}")
                return " | ".join(explanation)
            
            tokens = volume_qualifier.get('tokens', 0)
            tier = volume_qualifier.get('tier', 'unknown')
            explanation.append(f"📊 Volume: {tokens} tokens ({tier})")
            
            # Component breakdown
            risk_score = component_scores.get('risk_adjusted_score', 0)
            dist_score = component_scores.get('distribution_score', 0)
            disc_score = component_scores.get('discipline_score', 0)
            
            explanation.append(f"🎯 Risk-Adj: {risk_score:.1f}/30")
            explanation.append(f"📈 Distribution: {dist_score:.1f}/25") 
            explanation.append(f"⚖️ Discipline: {disc_score:.1f}/20")
            
            return " | ".join(explanation)
            
        except Exception as e:
            logger.error(f"Error generating score explanation: {str(e)}")
            return f"Score: {scoring_result.get('composite_score', 0)}/100"