"""
Zeus Scorer - Updated for Corrected Exit Analysis
ENHANCEMENTS:
- Enhanced field validation for corrected exit analysis data
- Updated pattern thresholds (5 minutes, 24 hours) 
- Better handling of actual vs final ROI data
- Enhanced error handling and data quality checks
- Maintained core scoring logic for consistency
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("zeus.scorer")

class ZeusScorer:
    """Implements Zeus scoring system with enhanced validation and corrected exit analysis awareness."""
    
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
        
        # Updated thresholds
        self.very_short_threshold_hours = 0.083  # 5 minutes
        self.long_hold_threshold_hours = 24      # 24 hours
        
        logger.info("Zeus Scorer initialized with CORRECTED EXIT ANALYSIS support")
        logger.info(f"  Very short holds: <{self.very_short_threshold_hours * 60:.0f} minutes")
        logger.info(f"  Long holds: >{self.long_hold_threshold_hours} hours")
        logger.info(f"  Exit analysis: Enhanced to handle actual vs final ROI")
    
    def calculate_composite_score(self, token_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate composite score from token analysis data with enhanced validation."""
        try:
            if not token_analysis:
                return self._get_zero_score("No token analysis data")
            
            logger.info(f"📊 Calculating composite score for {len(token_analysis)} tokens with CORRECTED EXIT AWARENESS")
            
            # Validate token analysis data
            validation_result = self._validate_token_analysis_data_enhanced(token_analysis)
            if not validation_result['valid']:
                logger.warning(f"⚠️ Data validation issues: {validation_result['issues']}")
            
            # Extract metrics from token analysis data with corrected exit awareness
            metrics = self._extract_token_analysis_metrics_enhanced(token_analysis)
            
            if not metrics:
                return self._get_zero_score("Failed to extract metrics from token analysis")
            
            # Validate extracted metrics
            metrics_validation = self._validate_extracted_metrics_enhanced(metrics)
            if not metrics_validation['valid']:
                logger.warning(f"⚠️ Metrics validation issues: {metrics_validation['issues']}")
            
            logger.info(f"Extracted token analysis metrics: {len(metrics)} fields")
            
            # Check volume qualifier first
            volume_qualifier = self._calculate_volume_qualifier(metrics)
            if volume_qualifier['disqualified']:
                return self._get_zero_score(volume_qualifier['reason'])
            
            logger.info(f"Volume qualifier: {volume_qualifier['tier']} (×{volume_qualifier['multiplier']})")
            
            # Calculate component scores with corrected exit analysis awareness
            component_scores = {}
            
            # 1. Risk-Adjusted Performance (30%) - Enhanced for actual vs final ROI
            component_scores['risk_adjusted_performance'] = self._calculate_risk_adjusted_score_enhanced(metrics)
            
            # 2. Distribution Quality (25%) - Enhanced for actual exit patterns
            component_scores['distribution_quality'] = self._calculate_distribution_score_enhanced(metrics)
            
            # 3. Trading Discipline (20%) - WITH CORRECTED THRESHOLDS
            component_scores['trading_discipline'] = self._calculate_discipline_score_enhanced(metrics)
            
            # 4. Market Impact Awareness (15%)
            component_scores['market_impact_awareness'] = self._calculate_market_impact_score(metrics)
            
            # 5. Consistency & Reliability (10%) - WITH CORRECTED THRESHOLDS
            component_scores['consistency_reliability'] = self._calculate_consistency_score_enhanced(metrics)
            
            # Log component scores
            logger.info("Component Scores (0-1 scale) with CORRECTED EXIT ANALYSIS:")
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
            
            logger.info(f"Final composite score: {composite_score:.1f}/100 (CORRECTED EXIT ANALYSIS)")
            
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
                'total_tokens_analyzed': len(token_analysis),
                'data_validation': validation_result,
                'metrics_validation': metrics_validation,
                'updated_thresholds': {
                    'very_short_hours': self.very_short_threshold_hours,
                    'long_hold_hours': self.long_hold_threshold_hours
                },
                'corrected_exit_analysis': True
            }
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {str(e)}")
            return self._get_zero_score(f"Calculation error: {str(e)}")
    
    def _validate_token_analysis_data_enhanced(self, token_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced validation of token analysis data quality with corrected exit analysis awareness."""
        validation = {
            'valid': True,
            'issues': [],
            'token_count': len(token_analysis),
            'valid_tokens': 0,
            'invalid_tokens': 0,
            'corrected_tokens': 0  # NEW: Tokens with corrected exit analysis
        }
        
        try:
            required_fields = ['token_mint', 'roi_percent', 'trade_status', 'hold_time_hours']
            
            for i, token in enumerate(token_analysis):
                token_issues = []
                
                # Check required fields
                for field in required_fields:
                    if field not in token:
                        token_issues.append(f"missing_{field}")
                
                # Enhanced validation for corrected exit data
                if 'data_source' in token and 'corrected' in str(token['data_source']).lower():
                    validation['corrected_tokens'] += 1
                
                # Validate data types and ranges with enhanced checks
                if 'roi_percent' in token:
                    roi = token['roi_percent']
                    if not isinstance(roi, (int, float)):
                        token_issues.append(f"invalid_roi_type_{type(roi).__name__}")
                    elif roi < -100 or roi > 10000:
                        token_issues.append(f"invalid_roi_range_{roi}")
                
                if 'hold_time_hours' in token:
                    hold_time = token['hold_time_hours']
                    if not isinstance(hold_time, (int, float)):
                        token_issues.append(f"invalid_hold_time_type_{type(hold_time).__name__}")
                    elif hold_time < 0 or hold_time > 8760:  # Max 1 year
                        token_issues.append(f"invalid_hold_time_range_{hold_time}")
                
                if token_issues:
                    validation['invalid_tokens'] += 1
                    validation['issues'].extend([f"token_{i}_{issue}" for issue in token_issues])
                else:
                    validation['valid_tokens'] += 1
            
            # Overall validation with enhanced criteria
            if validation['valid_tokens'] < len(token_analysis) * 0.7:  # At least 70% valid
                validation['valid'] = False
                validation['issues'].append(f"insufficient_valid_tokens_{validation['valid_tokens']}/{len(token_analysis)}")
            
            # Log corrected data usage
            if validation['corrected_tokens'] > 0:
                logger.info(f"✅ Found {validation['corrected_tokens']} tokens with corrected exit analysis")
            
            return validation
            
        except Exception as e:
            return {
                'valid': False,
                'issues': [f"validation_error_{str(e)}"],
                'token_count': len(token_analysis),
                'valid_tokens': 0,
                'invalid_tokens': len(token_analysis),
                'corrected_tokens': 0
            }
    
    def _validate_extracted_metrics_enhanced(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced validation of extracted metrics with corrected exit analysis awareness."""
        validation = {
            'valid': True,
            'issues': [],
            'metrics_count': len(metrics),
            'corrected_analysis_detected': False
        }
        
        try:
            # Required metrics with expected ranges
            expected_metrics = {
                'total_tokens': (1, 100),
                'avg_roi': (-100, 10000),
                'roi_std': (0, 5000),
                'win_rate': (0, 100),
                'avg_hold_time_hours': (0, 8760),
                'same_block_rate': (0, 100)
            }
            
            for metric, (min_val, max_val) in expected_metrics.items():
                if metric not in metrics:
                    validation['issues'].append(f"missing_{metric}")
                    continue
                
                value = metrics[metric]
                if not isinstance(value, (int, float)):
                    validation['issues'].append(f"invalid_type_{metric}_{type(value).__name__}")
                elif not (min_val <= value <= max_val):
                    validation['issues'].append(f"out_of_range_{metric}_{value}")
            
            # Enhanced validation for corrected exit analysis
            if 'corrected_exit_analysis' in metrics or 'actual_exit_rois' in metrics:
                validation['corrected_analysis_detected'] = True
                logger.info("✅ Corrected exit analysis data detected in metrics")
            
            # Additional enhanced validation
            if metrics.get('completed_trades', 0) > metrics.get('total_tokens', 0):
                validation['issues'].append("completed_trades_exceeds_total")
            
            # Check for realistic flipper behavior
            avg_hold_time = metrics.get('avg_hold_time_hours', 24)
            same_block_rate = metrics.get('same_block_rate', 0)
            if avg_hold_time < 0.1 and same_block_rate > 50:
                logger.info("📊 Flipper behavior detected - very short holds with high same-block rate")
            
            if validation['issues']:
                validation['valid'] = False
            
            return validation
            
        except Exception as e:
            return {
                'valid': False,
                'issues': [f"enhanced_metrics_validation_error_{str(e)}"],
                'metrics_count': len(metrics),
                'corrected_analysis_detected': False
            }
    
    def _extract_token_analysis_metrics_enhanced(self, token_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced extraction of metrics from token analysis data with corrected exit analysis support."""
        try:
            logger.info(f"Extracting ENHANCED metrics from {len(token_analysis)} token analyses")
            
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
                    'rois': [],
                    'corrected_exit_analysis': False
                }
            
            # Enhanced ROI metrics extraction with corrected exit analysis support
            rois = []
            actual_exit_rois = []  # NEW: Track actual exit ROIs vs final token ROIs
            corrected_data_detected = False
            
            for trade in completed_trades:
                roi = trade.get('roi_percent', 0)
                if roi is not None and isinstance(roi, (int, float)):
                    rois.append(float(roi))
                
                # Enhanced: Check for corrected exit analysis data
                if 'actual_exit_roi' in trade or 'corrected_analysis' in str(trade.get('data_source', '')):
                    actual_exit_roi = trade.get('actual_exit_roi', roi)
                    if isinstance(actual_exit_roi, (int, float)):
                        actual_exit_rois.append(float(actual_exit_roi))
                        corrected_data_detected = True
            
            if not rois:
                logger.warning("No valid ROI data found")
                rois = [0]
            
            # Use corrected exit ROIs if available, otherwise use regular ROIs
            analysis_rois = actual_exit_rois if actual_exit_rois else rois
            
            logger.info(f"Extracted {len(rois)} ROI values: min={min(rois):.1f}%, max={max(rois):.1f}%")
            if corrected_data_detected:
                logger.info(f"✅ Using {len(actual_exit_rois)} CORRECTED exit ROI values for enhanced analysis")
                logger.info(f"   Corrected range: min={min(actual_exit_rois):.1f}%, max={max(actual_exit_rois):.1f}%")
            
            # Calculate ROI statistics using the appropriate data set
            try:
                avg_roi = float(np.mean(analysis_rois))
                roi_std = float(np.std(analysis_rois)) if len(analysis_rois) > 1 else 0
                max_roi = float(max(analysis_rois))
                min_roi = float(min(analysis_rois))
            except Exception as stats_error:
                logger.debug(f"Error calculating statistics: {str(stats_error)}")
                avg_roi = sum(analysis_rois) / len(analysis_rois) if analysis_rois else 0
                roi_std = 0
                max_roi = max(analysis_rois) if analysis_rois else 0
                min_roi = min(analysis_rois) if analysis_rois else 0
            
            # Hold time metrics with UPDATED THRESHOLDS
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
            
            # Enhanced distribution metrics using corrected ROI data
            moonshots = sum(1 for roi in analysis_rois if roi >= 400)  # 5x+
            big_wins = sum(1 for roi in analysis_rois if 100 <= roi < 400)  # 2x-5x
            small_wins = sum(1 for roi in analysis_rois if 0 < roi < 100)  # Profitable <2x
            small_losses = sum(1 for roi in analysis_rois if -50 < roi <= 0)  # Small losses
            heavy_losses = sum(1 for roi in analysis_rois if roi <= -50)  # Heavy losses
            
            # Calculate distribution percentages
            total_completed = len(analysis_rois)
            moonshot_rate = (moonshots / total_completed * 100) if total_completed > 0 else 0
            big_win_rate = (big_wins / total_completed * 100) if total_completed > 0 else 0
            heavy_loss_rate = (heavy_losses / total_completed * 100) if total_completed > 0 else 0
            
            # Win rate using corrected data
            wins = sum(1 for roi in analysis_rois if roi > 0)
            win_rate = (wins / total_completed * 100) if total_completed > 0 else 0
            
            # Same-block detection with UPDATED THRESHOLD (flipper behavior)
            same_block_trades = 0
            for trade in all_trades:
                hold_time = trade.get('hold_time_hours', 24)
                if hold_time < self.very_short_threshold_hours:  # Less than 5 minutes
                    same_block_trades += 1
            
            same_block_rate = (same_block_trades / total_tokens * 100) if total_tokens > 0 else 0
            
            # Loss cutting behavior with UPDATED THRESHOLDS
            quick_cut_losses = sum(1 for t in completed_trades 
                                 if t.get('roi_percent', 0) < -10 and t.get('hold_time_hours', 24) < 4)
            slow_cut_losses = sum(1 for t in completed_trades 
                                if t.get('roi_percent', 0) < -10 and t.get('hold_time_hours', 0) > 24 * 7)
            
            # Long hold detection with UPDATED THRESHOLD
            long_holds = sum(1 for t in all_trades 
                           if t.get('hold_time_hours', 0) > self.long_hold_threshold_hours)
            long_hold_rate = (long_holds / total_tokens * 100) if total_tokens > 0 else 0
            
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
            
            logger.info(f"Calculated ENHANCED metrics:")
            logger.info(f"  win_rate={win_rate:.1f}%, avg_roi={avg_roi:.1f}%, moonshot_rate={moonshot_rate:.1f}%")
            logger.info(f"  UPDATED THRESHOLDS: same_block_rate={same_block_rate:.1f}%, long_hold_rate={long_hold_rate:.1f}%")
            if corrected_data_detected:
                logger.info(f"  ✅ CORRECTED EXIT ANALYSIS: Used actual exit ROIs for {len(actual_exit_rois)} trades")
            
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
                'long_hold_rate': long_hold_rate,
                'quick_cut_losses': quick_cut_losses,
                'slow_cut_losses': slow_cut_losses,
                'active_days': active_days,
                'rois': rois,
                'actual_exit_rois': actual_exit_rois,  # NEW: Corrected exit ROIs
                'corrected_exit_analysis': corrected_data_detected,
                'updated_thresholds_applied': True
            }
            
        except Exception as e:
            logger.error(f"Error extracting enhanced token analysis metrics: {str(e)}")
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
    
    def _calculate_risk_adjusted_score_enhanced(self, metrics: Dict[str, Any]) -> float:
        """Enhanced Risk-Adjusted Performance Score (30% weight) with corrected exit analysis awareness."""
        try:
            # Use corrected exit data if available
            corrected_analysis = metrics.get('corrected_exit_analysis', False)
            if corrected_analysis:
                logger.debug("Using CORRECTED exit analysis data for risk-adjusted score")
            
            # Average ROI component (60% of this score) - Enhanced for actual exits
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
            
            # Enhanced bonus for corrected analysis
            if corrected_analysis and avg_roi > 0:
                roi_score = min(1.0, roi_score * 1.05)  # 5% bonus for corrected data
            
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
            
            logger.debug(f"Enhanced risk-adjusted: avg_roi={roi_score:.2f}, std={std_score:.2f}, win={win_score:.2f} → {risk_adjusted_score:.3f}")
            if corrected_analysis:
                logger.debug(f"  ✅ Enhanced with corrected exit analysis")
            
            return min(1.0, max(0.0, risk_adjusted_score))
            
        except Exception as e:
            logger.error(f"Error calculating enhanced risk-adjusted score: {str(e)}")
            return 0.0
    
    def _calculate_distribution_score_enhanced(self, metrics: Dict[str, Any]) -> float:
        """Enhanced Distribution Quality Score (25% weight) with corrected exit analysis awareness."""
        try:
            # Use corrected exit data if available
            corrected_analysis = metrics.get('corrected_exit_analysis', False)
            if corrected_analysis:
                logger.debug("Using CORRECTED exit analysis data for distribution score")
            
            # Moonshot Rate component (50% of this score) - Enhanced for actual exits
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
            
            # Enhanced penalty for flippers claiming moonshots (likely incorrect)
            avg_hold_time = metrics.get('avg_hold_time_hours', 24)
            if avg_hold_time < 0.1 and moonshot_rate > 10:  # Flipper with high moonshots
                moonshot_score *= 0.5  # 50% penalty - likely incorrect data
                logger.debug(f"Applied flipper moonshot penalty: hold_time={avg_hold_time:.3f}h, moonshot_rate={moonshot_rate:.1f}%")
            
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
            
            # Enhanced bonus for corrected analysis
            if corrected_analysis:
                distribution_score = min(1.0, distribution_score * 1.03)  # 3% bonus for corrected data
            
            logger.debug(f"Enhanced distribution: moonshot={moonshot_score:.2f}, big_win={big_win_score:.2f}, loss={loss_score:.2f} → {distribution_score:.3f}")
            if corrected_analysis:
                logger.debug(f"  ✅ Enhanced with corrected exit analysis")
            
            return min(1.0, max(0.0, distribution_score))
            
        except Exception as e:
            logger.error(f"Error calculating enhanced distribution score: {str(e)}")
            return 0.0
    
    def _calculate_discipline_score_enhanced(self, metrics: Dict[str, Any]) -> float:
        """Enhanced Trading Discipline Score (20% weight) with CORRECTED THRESHOLDS."""
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
            
            # Exit Behavior component (35% of this score) - Enhanced with corrected analysis
            win_rate = metrics.get('win_rate', 0)
            corrected_analysis = metrics.get('corrected_exit_analysis', False)
            
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
            
            # Enhanced bonus for corrected exit analysis
            if corrected_analysis and win_rate > 40:
                exit_score = min(1.0, exit_score * 1.1)  # 10% bonus for good corrected performance
            
            # Fast Sell Detection component (25% of this score) - UPDATED THRESHOLD
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
            
            # Enhanced: Adjust for realistic flipper patterns
            avg_hold_time = metrics.get('avg_hold_time_hours', 24)
            if avg_hold_time < 0.1 and same_block_rate > 30:  # Confirmed flipper
                # Don't completely penalize - they might be good at quick profits
                avg_roi = metrics.get('avg_roi', 0)
                if avg_roi > 20:  # Good flipper performance
                    fast_sell_score = max(fast_sell_score, 0.4)  # Give some credit
                    logger.debug(f"Applied good flipper bonus: avg_roi={avg_roi:.1f}%")
            
            # Combine components
            discipline_score = (
                loss_mgmt_score * 0.40 +
                exit_score * 0.35 +
                fast_sell_score * 0.25
            )
            
            logger.debug(f"Enhanced discipline: loss_mgmt={loss_mgmt_score:.2f}, exit={exit_score:.2f}, fast_sell={fast_sell_score:.2f} → {discipline_score:.3f}")
            logger.debug(f"  UPDATED: same_block_rate={same_block_rate:.1f}% (threshold <5min)")
            if corrected_analysis:
                logger.debug(f"  ✅ Enhanced with corrected exit analysis")
            
            return min(1.0, max(0.0, discipline_score))
            
        except Exception as e:
            logger.error(f"Error calculating enhanced discipline score: {str(e)}")
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
    
    def _calculate_consistency_score_enhanced(self, metrics: Dict[str, Any]) -> float:
        """Enhanced Consistency & Reliability Score (10% weight) with CORRECTED THRESHOLDS."""
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
            
            # Red Flags component (30% of this score) - UPDATED THRESHOLDS with Enhanced Analysis
            same_block_rate = metrics.get('same_block_rate', 0)
            avg_hold_time = metrics.get('avg_hold_time_hours', 24)
            corrected_analysis = metrics.get('corrected_exit_analysis', False)
            
            if same_block_rate > 50:
                red_flag_score = 0.0  # Likely bot
            elif same_block_rate > 30:
                # Enhanced: Check if it's profitable flipper behavior
                avg_roi = metrics.get('avg_roi', 0)
                win_rate = metrics.get('win_rate', 0)
                if avg_roi > 15 and win_rate > 40:  # Profitable flipper
                    red_flag_score = 0.5  # Reduced penalty for profitable behavior
                    logger.debug(f"Applied profitable flipper adjustment: ROI={avg_roi:.1f}%, WR={win_rate:.1f}%")
                else:
                    red_flag_score = 0.3  # High bot suspicion
            elif same_block_rate > 15:
                red_flag_score = 0.6  # Some bot behavior
            elif same_block_rate > 5:
                red_flag_score = 0.8  # Minimal bot behavior
            else:
                red_flag_score = 1.0  # Clean behavior
            
            # Enhanced bonus for corrected analysis
            if corrected_analysis:
                red_flag_score = min(1.0, red_flag_score * 1.05)  # 5% bonus for corrected data
            
            # Combine components
            consistency_score = (
                activity_score * 0.70 +
                red_flag_score * 0.30
            )
            
            logger.debug(f"Enhanced consistency: activity={activity_score:.2f}, red_flags={red_flag_score:.2f} → {consistency_score:.3f}")
            logger.debug(f"  UPDATED: same_block_rate={same_block_rate:.1f}% (threshold <5min)")
            if corrected_analysis:
                logger.debug(f"  ✅ Enhanced with corrected exit analysis")
            
            return min(1.0, max(0.0, consistency_score))
            
        except Exception as e:
            logger.error(f"Error calculating enhanced consistency score: {str(e)}")
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
            'total_tokens_analyzed': 0,
            'data_validation': {
                'valid': False,
                'issues': [reason]
            },
            'updated_thresholds': {
                'very_short_hours': self.very_short_threshold_hours,
                'long_hold_hours': self.long_hold_threshold_hours
            },
            'corrected_exit_analysis': False
        }
    
    def get_score_explanation(self, scoring_result: Dict[str, Any]) -> str:
        """Generate human-readable explanation of the score with corrected exit analysis awareness."""
        try:
            composite_score = scoring_result.get('composite_score', 0)
            component_scores = scoring_result.get('component_scores', {})
            volume_qualifier = scoring_result.get('volume_qualifier', {})
            corrected_analysis = scoring_result.get('corrected_exit_analysis', False)
            
            explanation = []
            
            # Overall score with corrected analysis indicator
            if composite_score >= 80:
                explanation.append(f"🟢 EXCELLENT (Score: {composite_score}/100)")
            elif composite_score >= 65:
                explanation.append(f"🟡 GOOD (Score: {composite_score}/100)")
            elif composite_score >= 50:
                explanation.append(f"🟠 AVERAGE (Score: {composite_score}/100)")
            else:
                explanation.append(f"🔴 POOR (Score: {composite_score}/100)")
            
            # Add corrected analysis indicator
            if corrected_analysis:
                explanation.append("✅ CORRECTED")
            
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
            
            # Updated thresholds info
            updated_thresholds = scoring_result.get('updated_thresholds', {})
            if updated_thresholds:
                very_short_min = updated_thresholds.get('very_short_hours', 0.083) * 60
                long_hold_hrs = updated_thresholds.get('long_hold_hours', 24)
                explanation.append(f"⚡ Thresholds: <{very_short_min:.0f}min | >{long_hold_hrs}h")
            
            return " | ".join(explanation)
            
        except Exception as e:
            logger.error(f"Error generating score explanation: {str(e)}")
            return f"Score: {scoring_result.get('composite_score', 0)}/100"