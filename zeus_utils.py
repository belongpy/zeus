"""
Zeus Utilities - Enhanced for Corrected Exit Analysis and Realistic TP/SL
MAJOR ENHANCEMENTS:
- Added utility functions for corrected exit analysis
- Pattern-based TP/SL validation and calculation utilities
- Enhanced trader pattern identification with realistic thresholds
- Utility functions to validate and correct inflated TP/SL recommendations
- Support for actual vs final ROI analysis
"""

import os
import re
import time
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import base58

logger = logging.getLogger("zeus.utils")

class ZeusUtils:
    """Collection of utility functions for Zeus system with CORRECTED exit analysis and realistic TP/SL utilities."""
    
    @staticmethod
    def validate_solana_address(address: str) -> Dict[str, Any]:
        """
        Validate Solana wallet/token address with SAFE type checking.
        
        Args:
            address: Address to validate
            
        Returns:
            Dict with validation result and details
        """
        try:
            if not address or not isinstance(address, str):
                return {
                    'valid': False,
                    'error': 'Address must be a non-empty string'
                }
            
            # Remove whitespace
            address = address.strip()
            
            # Check length (Solana addresses are typically 32-44 characters)
            if not (32 <= len(address) <= 44):
                return {
                    'valid': False,
                    'error': f'Address length {len(address)} invalid (expected 32-44 characters)'
                }
            
            # Check base58 encoding
            try:
                decoded = base58.b58decode(address)
                if len(decoded) != 32:
                    return {
                        'valid': False,
                        'error': 'Decoded address must be 32 bytes'
                    }
            except Exception:
                return {
                    'valid': False,
                    'error': 'Invalid base58 encoding'
                }
            
            # Additional checks for common invalid patterns
            if address == '1' * len(address):
                return {
                    'valid': False,
                    'error': 'Address cannot be all same character'
                }
            
            return {
                'valid': True,
                'address': address,
                'length': len(address),
                'type': 'solana_address'
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}'
            }
    
    @staticmethod
    def identify_corrected_trader_pattern(metrics: Dict[str, Any], use_corrected_thresholds: bool = True) -> str:
        """
        Identify trader pattern based on CORRECTED analysis with realistic thresholds.
        
        Args:
            metrics: Trading metrics from corrected exit analysis
            use_corrected_thresholds: Use corrected thresholds (5min/24hr)
            
        Returns:
            Identified pattern with corrected analysis
        """
        try:
            if not isinstance(metrics, dict):
                return 'analysis_error'
                
            # Extract metrics with SAFE validation
            avg_hold_time_hours = ZeusUtils._safe_float(metrics.get('avg_hold_time_hours', 24), 24)
            avg_roi = ZeusUtils._safe_float(metrics.get('avg_roi', 0), 0)
            moonshot_rate = ZeusUtils._safe_float(metrics.get('moonshot_rate', 0), 0)
            win_rate = ZeusUtils._safe_float(metrics.get('win_rate', 50), 50)
            total_trades = ZeusUtils._safe_int(metrics.get('total_trades', 0), 0)
            corrected_analysis = metrics.get('corrected_exit_analysis', False)
            
            # CORRECTED thresholds
            if use_corrected_thresholds:
                very_short_threshold = 0.083  # 5 minutes
                long_hold_threshold = 24      # 24 hours
            else:
                very_short_threshold = 0.2    # 12 minutes (old)
                long_hold_threshold = 48      # 48 hours (old)
            
            # Enhanced pattern identification with corrected exit analysis awareness
            if avg_hold_time_hours < very_short_threshold:
                if corrected_analysis and avg_roi > 15:
                    return 'skilled_flipper'  # Good at quick profits
                else:
                    return 'flipper'
            elif avg_hold_time_hours < 1:
                if avg_roi > 30:
                    return 'sniper' 
                else:
                    return 'impulsive_trader'
            elif moonshot_rate > 10 and avg_roi > 100:
                if corrected_analysis:
                    return 'verified_gem_hunter'  # Confirmed with actual exits
                else:
                    return 'gem_hunter'
            elif avg_hold_time_hours > long_hold_threshold:
                if avg_roi > 50:
                    return 'position_trader'
                else:
                    return 'bag_holder'
            elif win_rate > 60 and avg_roi > 20:
                return 'consistent_trader'
            else:
                return 'mixed_strategy'
                
        except Exception as e:
            logger.error(f"Error identifying corrected trader pattern: {str(e)}")
            return 'analysis_error'
    
    @staticmethod
    def calculate_realistic_tp_sl(pattern: str, corrected_performance: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculate REALISTIC TP/SL levels based on trader pattern and corrected performance data.
        MAJOR FIX: Provides realistic TP/SL levels that make sense for each pattern.
        
        Args:
            pattern: Identified trader pattern
            corrected_performance: Corrected performance metrics from actual exit analysis
            
        Returns:
            Dict with realistic TP/SL recommendations
        """
        try:
            # Realistic pattern-based defaults (CORRECTED)
            pattern_defaults = {
                'flipper': {
                    'tp1': 25, 'tp2': 50, 'tp3': 80, 'stop_loss': -15,
                    'reasoning': 'Flipper pattern - Quick exits at small gains'
                },
                'skilled_flipper': {
                    'tp1': 35, 'tp2': 65, 'tp3': 100, 'stop_loss': -12,
                    'reasoning': 'Skilled flipper - Slightly higher but still realistic quick profits'
                },
                'sniper': {
                    'tp1': 50, 'tp2': 100, 'tp3': 150, 'stop_loss': -20,
                    'reasoning': 'Sniper pattern - Fast but bigger profits'
                },
                'gem_hunter': {
                    'tp1': 100, 'tp2': 250, 'tp3': 500, 'stop_loss': -40,
                    'reasoning': 'Gem hunter pattern - Patient for larger gains'
                },
                'verified_gem_hunter': {
                    'tp1': 150, 'tp2': 350, 'tp3': 700, 'stop_loss': -35,
                    'reasoning': 'Verified gem hunter - Confirmed large exit capability'
                },
                'position_trader': {
                    'tp1': 80, 'tp2': 180, 'tp3': 350, 'stop_loss': -30,
                    'reasoning': 'Position trader - Long-term hold strategy'
                },
                'consistent_trader': {
                    'tp1': 60, 'tp2': 130, 'tp3': 250, 'stop_loss': -25,
                    'reasoning': 'Consistent trader - Balanced approach'
                }
            }
            
            base_recommendation = pattern_defaults.get(pattern, pattern_defaults['consistent_trader'])
            
            # Enhancement with corrected performance data
            if corrected_performance and isinstance(corrected_performance, dict):
                if corrected_performance.get('based_on_actual_exits'):
                    try:
                        actual_tp1 = ZeusUtils._safe_float(corrected_performance.get('avg_tp1', base_recommendation['tp1']), base_recommendation['tp1'])
                        actual_tp2 = ZeusUtils._safe_float(corrected_performance.get('avg_tp2', base_recommendation['tp2']), base_recommendation['tp2'])
                        actual_sl = ZeusUtils._safe_float(corrected_performance.get('avg_stop_loss', base_recommendation['stop_loss']), base_recommendation['stop_loss'])
                        
                        # Validate that actual data makes sense for the pattern
                        validated_tp1, validated_tp2, validated_sl = ZeusUtils.validate_tp_sl_for_pattern(
                            pattern, actual_tp1, actual_tp2, actual_sl
                        )
                        
                        return {
                            'tp1': validated_tp1,
                            'tp2': validated_tp2,
                            'tp3': max(validated_tp2 + 50, int(validated_tp2 * 1.5)),
                            'stop_loss': validated_sl,
                            'reasoning': f'CORRECTED {pattern} based on actual exits with validation',
                            'based_on_actual_exits': True,
                            'validation_applied': True,
                            'original_performance': corrected_performance
                        }
                    except Exception as perf_error:
                        logger.debug(f"Error processing corrected performance data: {str(perf_error)}")
            
            return {
                **base_recommendation,
                'based_on_actual_exits': False,
                'pattern_based': True,
                'validation_applied': True
            }
            
        except Exception as e:
            logger.error(f"Error calculating realistic TP/SL: {str(e)}")
            return {
                'tp1': 50, 'tp2': 120, 'tp3': 250, 'stop_loss': -30,
                'reasoning': f'Default due to error: {str(e)}',
                'based_on_actual_exits': False
            }
    
    @staticmethod
    def validate_tp_sl_for_pattern(pattern: str, tp1: float, tp2: float, stop_loss: float) -> Tuple[float, float, float]:
        """
        Validate TP/SL levels to ensure they make sense for the trading pattern.
        CRITICAL: Prevents inflated TP recommendations that don't match trading behavior.
        
        Args:
            pattern: Trading pattern
            tp1, tp2: Take profit levels
            stop_loss: Stop loss level
            
        Returns:
            Validated and corrected TP/SL levels
        """
        try:
            # Pattern-specific validation ranges
            if pattern in ['flipper', 'skilled_flipper']:
                # Flippers should have LOW TP levels - they exit quickly
                tp1 = max(15, min(80, tp1))
                tp2 = max(tp1 + 15, min(120, tp2))
                stop_loss = max(-25, min(-8, stop_loss))
                
            elif pattern == 'sniper':
                # Snipers take quick but slightly larger profits
                tp1 = max(30, min(120, tp1))
                tp2 = max(tp1 + 20, min(200, tp2))
                stop_loss = max(-30, min(-12, stop_loss))
                
            elif pattern in ['gem_hunter', 'verified_gem_hunter']:
                # Gem hunters can have higher TPs but still realistic
                tp1 = max(60, min(300, tp1))
                tp2 = max(tp1 + 50, min(600, tp2))
                stop_loss = max(-50, min(-20, stop_loss))
                
            elif pattern == 'position_trader':
                # Position traders hold longer for bigger gains
                tp1 = max(50, min(200, tp1))
                tp2 = max(tp1 + 40, min(400, tp2))
                stop_loss = max(-40, min(-20, stop_loss))
                
            elif pattern == 'consistent_trader':
                # Consistent traders have moderate, balanced levels
                tp1 = max(40, min(150, tp1))
                tp2 = max(tp1 + 30, min(300, tp2))
                stop_loss = max(-35, min(-15, stop_loss))
                
            else:
                # Default/mixed strategy - conservative levels
                tp1 = max(25, min(100, tp1))
                tp2 = max(tp1 + 25, min(200, tp2))
                stop_loss = max(-35, min(-20, stop_loss))
            
            return int(tp1), int(tp2), int(stop_loss)
            
        except Exception as e:
            logger.error(f"Error validating TP/SL for pattern {pattern}: {str(e)}")
            return 50, 120, -30
    
    @staticmethod
    def detect_inflated_tp_sl(pattern: str, tp1: float, tp2: float) -> Dict[str, Any]:
        """
        Detect if TP/SL recommendations are inflated and don't match the trading pattern.
        
        Args:
            pattern: Trading pattern
            tp1, tp2: Take profit levels to validate
            
        Returns:
            Detection result with corrections if needed
        """
        try:
            detection = {
                'inflated': False,
                'severity': 'none',
                'issues': [],
                'corrected_tp1': int(tp1),
                'corrected_tp2': int(tp2),
                'pattern_expected_range': {}
            }
            
            # Pattern-specific inflation detection
            if pattern in ['flipper', 'skilled_flipper']:
                expected_tp1_max = 80
                expected_tp2_max = 120
                detection['pattern_expected_range'] = {'tp1_max': expected_tp1_max, 'tp2_max': expected_tp2_max}
                
                if tp1 > expected_tp1_max:
                    detection['inflated'] = True
                    detection['issues'].append(f'TP1 {tp1}% too high for flipper (max {expected_tp1_max}%)')
                    detection['corrected_tp1'] = min(int(tp1), expected_tp1_max)
                
                if tp2 > expected_tp2_max:
                    detection['inflated'] = True
                    detection['issues'].append(f'TP2 {tp2}% too high for flipper (max {expected_tp2_max}%)')
                    detection['corrected_tp2'] = min(int(tp2), expected_tp2_max)
                
            elif pattern == 'sniper':
                expected_tp1_max = 120
                expected_tp2_max = 200
                detection['pattern_expected_range'] = {'tp1_max': expected_tp1_max, 'tp2_max': expected_tp2_max}
                
                if tp1 > expected_tp1_max or tp2 > expected_tp2_max:
                    detection['inflated'] = True
                    detection['corrected_tp1'] = min(int(tp1), expected_tp1_max)
                    detection['corrected_tp2'] = min(int(tp2), expected_tp2_max)
                    detection['issues'].append(f'TP levels too high for sniper pattern')
                
            elif pattern in ['gem_hunter', 'verified_gem_hunter']:
                expected_tp1_max = 300
                expected_tp2_max = 600
                detection['pattern_expected_range'] = {'tp1_max': expected_tp1_max, 'tp2_max': expected_tp2_max}
                
                if tp1 > expected_tp1_max or tp2 > expected_tp2_max:
                    detection['inflated'] = True
                    detection['corrected_tp1'] = min(int(tp1), expected_tp1_max)
                    detection['corrected_tp2'] = min(int(tp2), expected_tp2_max)
                    detection['issues'].append(f'TP levels unrealistic even for gem hunter')
            
            # Determine severity
            if detection['inflated']:
                if tp1 > 200 and pattern in ['flipper', 'skilled_flipper']:
                    detection['severity'] = 'critical'  # Flipper with 200%+ TP is completely wrong
                elif tp1 > 150 or tp2 > 400:
                    detection['severity'] = 'high'
                else:
                    detection['severity'] = 'moderate'
            
            return detection
            
        except Exception as e:
            logger.error(f"Error detecting inflated TP/SL: {str(e)}")
            return {
                'inflated': False,
                'severity': 'error',
                'issues': [f'Detection error: {str(e)}'],
                'corrected_tp1': int(tp1),
                'corrected_tp2': int(tp2)
            }
    
    @staticmethod
    def infer_actual_exit_behavior(final_roi: float, hold_time_hours: float, num_swaps: int, pattern: str) -> Dict[str, Any]:
        """
        Infer actual exit behavior vs final token performance.
        CORE FUNCTION: Distinguishes between what they got vs what the token did after.
        
        Args:
            final_roi: Final ROI of the token position
            hold_time_hours: How long they held
            num_swaps: Number of buy/sell transactions
            pattern: Trading pattern
            
        Returns:
            Inferred actual exit behavior
        """
        try:
            # Flipper behavior inference
            if pattern in ['flipper', 'skilled_flipper'] or hold_time_hours < 0.1:
                if final_roi > 100:
                    # Token pumped after they sold - they likely got 20-50%
                    actual_exit_roi = min(final_roi, 30 + (final_roi * 0.1))  # Conservative estimate
                    exit_strategy = 'quick_flip_before_pump'
                elif final_roi > 0:
                    # Small gain - they probably got most of it
                    actual_exit_roi = final_roi * 0.8
                    exit_strategy = 'quick_flip_small_gain'
                else:
                    # Loss - they probably cut it quick
                    actual_exit_roi = max(final_roi, -20)
                    exit_strategy = 'quick_loss_cut'
                
                return {
                    'actual_exit_roi': actual_exit_roi,
                    'exit_strategy': exit_strategy,
                    'confidence': 'high',
                    'reasoning': f'Flipper behavior: {hold_time_hours:.3f}h hold suggests early exit'
                }
            
            # Multiple swap analysis
            elif num_swaps > 3:
                if final_roi > 200:
                    # Partial exits during pump
                    actual_exit_roi = final_roi * 0.6  # Got 60% of the action
                    exit_strategy = 'partial_profit_taking'
                elif final_roi > 50:
                    actual_exit_roi = final_roi * 0.8  # Got most of it
                    exit_strategy = 'graduated_exits'
                else:
                    actual_exit_roi = final_roi  # Probably held to the end
                    exit_strategy = 'multiple_small_exits'
                
                return {
                    'actual_exit_roi': actual_exit_roi,
                    'exit_strategy': exit_strategy,
                    'confidence': 'medium',
                    'reasoning': f'Multiple swaps ({num_swaps}) suggest complex exit strategy'
                }
            
            # Long hold analysis
            elif hold_time_hours > 24:
                if final_roi > 500:
                    # True gem hunter - held through the pump
                    actual_exit_roi = final_roi * 0.9  # Got most of the gains
                    exit_strategy = 'diamond_hands'
                elif final_roi > 0:
                    actual_exit_roi = final_roi * 0.85
                    exit_strategy = 'patient_holder'
                else:
                    actual_exit_roi = final_roi  # Bag held
                    exit_strategy = 'bag_holder'
                
                return {
                    'actual_exit_roi': actual_exit_roi,
                    'exit_strategy': exit_strategy,
                    'confidence': 'medium',
                    'reasoning': f'Long hold ({hold_time_hours:.1f}h) suggests patient strategy'
                }
            
            # Default case
            else:
                actual_exit_roi = final_roi * 0.9  # Assume they got most of it
                exit_strategy = 'standard_exit'
                
                return {
                    'actual_exit_roi': actual_exit_roi,
                    'exit_strategy': exit_strategy,
                    'confidence': 'low',
                    'reasoning': 'Standard trade - assume 90% of final performance'
                }
                
        except Exception as e:
            logger.error(f"Error inferring actual exit behavior: {str(e)}")
            return {
                'actual_exit_roi': final_roi * 0.8,
                'exit_strategy': 'unknown',
                'confidence': 'low',
                'reasoning': f'Error in analysis: {str(e)}'
            }
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """
        Format percentage value for display with updated precision.
        
        Args:
            value: Percentage value
            decimals: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        try:
            if isinstance(value, (int, float)):
                if value >= 0:
                    return f"+{value:.{decimals}f}%"
                else:
                    return f"{value:.{decimals}f}%"
            else:
                return "0.0%"
        except:
            return "0.0%"
    
    @staticmethod
    def format_tp_sl_recommendation(tp1: int, tp2: int, tp3: int, stop_loss: int, pattern: str) -> str:
        """
        Format TP/SL recommendation for display with pattern context.
        
        Args:
            tp1, tp2, tp3: Take profit levels
            stop_loss: Stop loss level
            pattern: Trading pattern
            
        Returns:
            Formatted recommendation string
        """
        try:
            # Validate the levels first
            validated_tp1, validated_tp2, validated_sl = ZeusUtils.validate_tp_sl_for_pattern(pattern, tp1, tp2, stop_loss)
            
            recommendation = f"TP: {validated_tp1}%/{validated_tp2}%/{tp3}% | SL: {validated_sl}%"
            
            # Add pattern context
            if pattern in ['flipper', 'skilled_flipper']:
                recommendation += " (Quick exits)"
            elif pattern == 'gem_hunter':
                recommendation += " (Patient for gems)"
            elif pattern == 'position_trader':
                recommendation += " (Long-term holds)"
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error formatting TP/SL recommendation: {str(e)}")
            return f"TP: {tp1}%/{tp2}%/{tp3}% | SL: {stop_loss}%"
    
    @staticmethod
    def calculate_roi_percentage(initial_value: float, final_value: float) -> float:
        """
        Calculate ROI percentage with enhanced precision and SAFE type checking.
        
        Args:
            initial_value: Initial investment value
            final_value: Final value
            
        Returns:
            ROI percentage
        """
        try:
            if not isinstance(initial_value, (int, float)) or not isinstance(final_value, (int, float)):
                return 0.0
            if initial_value <= 0:
                return 0.0
            return round(((final_value / initial_value) - 1) * 100, 2)
        except:
            return 0.0
    
    @staticmethod
    def format_duration(seconds: float, use_corrected_thresholds: bool = True) -> str:
        """
        Format duration in human-readable format with corrected thresholds.
        
        Args:
            seconds: Duration in seconds
            use_corrected_thresholds: Use corrected thresholds (5min/24hr)
            
        Returns:
            Formatted duration string
        """
        try:
            if not isinstance(seconds, (int, float)) or seconds < 0:
                return "0s"
                
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                minutes = seconds / 60
                # Mark very short holds with corrected threshold
                if use_corrected_thresholds and minutes < 5:
                    return f"{minutes:.1f}m ⚡"  # Very short indicator
                return f"{minutes:.1f}m"
            elif seconds < 86400:
                hours = seconds / 3600
                return f"{hours:.1f}h"
            else:
                days = seconds / 86400
                # Mark long holds with corrected threshold
                if use_corrected_thresholds and days >= 1:  # 24+ hours
                    return f"{days:.1f}d 🔒"  # Long hold indicator
                return f"{days:.1f}d"
        except:
            return "0s"
    
    @staticmethod
    def create_corrected_analysis_summary(analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create summary of corrected analysis results with TP/SL validation.
        
        Args:
            analysis_results: List of analysis results
            
        Returns:
            Summary with corrected analysis insights
        """
        try:
            summary = {
                'total_wallets': len(analysis_results),
                'successful_analyses': 0,
                'failed_analyses': 0,
                'pattern_distribution': {},
                'tp_sl_validation': {
                    'inflated_recommendations': 0,
                    'corrected_recommendations': 0,
                    'pattern_mismatches': []
                },
                'corrected_exit_analysis': {
                    'wallets_with_corrected_data': 0,
                    'avg_correction_impact': 0
                }
            }
            
            inflated_count = 0
            corrected_count = 0
            
            for analysis in analysis_results:
                if not isinstance(analysis, dict):
                    continue
                    
                if analysis.get('success'):
                    summary['successful_analyses'] += 1
                    
                    # Pattern distribution
                    pattern = 'unknown'
                    trade_pattern_analysis = analysis.get('trade_pattern_analysis', {})
                    if isinstance(trade_pattern_analysis, dict):
                        pattern = trade_pattern_analysis.get('pattern', 'unknown')
                    
                    summary['pattern_distribution'][pattern] = summary['pattern_distribution'].get(pattern, 0) + 1
                    
                    # TP/SL validation
                    strategy = analysis.get('strategy_recommendation', {})
                    if isinstance(strategy, dict):
                        tp1 = ZeusUtils._safe_int(strategy.get('tp1_percent', 50), 50)
                        tp2 = ZeusUtils._safe_int(strategy.get('tp2_percent', 120), 120)
                        
                        # Check for inflation
                        inflation_check = ZeusUtils.detect_inflated_tp_sl(pattern, tp1, tp2)
                        if inflation_check['inflated']:
                            inflated_count += 1
                            if inflation_check['severity'] == 'critical':
                                summary['tp_sl_validation']['pattern_mismatches'].append({
                                    'wallet': analysis.get('wallet_address', '')[:8],
                                    'pattern': pattern,
                                    'tp1': tp1,
                                    'tp2': tp2,
                                    'issues': inflation_check['issues']
                                })
                    
                    # Check for corrected exit analysis
                    if isinstance(trade_pattern_analysis, dict) and trade_pattern_analysis.get('exit_analysis_corrected'):
                        summary['corrected_exit_analysis']['wallets_with_corrected_data'] += 1
                        corrected_count += 1
                        
                else:
                    summary['failed_analyses'] += 1
            
            summary['tp_sl_validation']['inflated_recommendations'] = inflated_count
            summary['tp_sl_validation']['corrected_recommendations'] = corrected_count
            
            # Calculate correction impact
            if summary['successful_analyses'] > 0:
                summary['corrected_exit_analysis']['correction_rate'] = (corrected_count / summary['successful_analyses']) * 100
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating corrected analysis summary: {str(e)}")
            return {
                'total_wallets': len(analysis_results) if isinstance(analysis_results, list) else 0,
                'error': str(e)
            }
    
    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safe float conversion with SAFE default."""
        try:
            if isinstance(value, (int, float)) and not (isinstance(value, float) and value != value):  # Check for NaN
                return float(value)
            else:
                return float(default)
        except:
            return float(default)
    
    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Safe int conversion with SAFE default."""
        try:
            if isinstance(value, (int, float)) and not (isinstance(value, float) and value != value):  # Check for NaN
                return int(value)
            else:
                return int(default)
        except:
            return int(default)
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safe division with default value and SAFE type checking.
        
        Args:
            numerator: Numerator
            denominator: Denominator
            default: Default value if division by zero
            
        Returns:
            Division result or default
        """
        try:
            if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
                return default
            if denominator == 0:
                return default
            return numerator / denominator
        except:
            return default
    
    @staticmethod
    def validate_corrected_analysis_data(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate corrected analysis data quality and consistency.
        
        Args:
            analysis_data: Analysis data to validate
            
        Returns:
            Validation result
        """
        try:
            validation = {
                'valid': True,
                'issues': [],
                'corrected_features_detected': [],
                'recommendations': []
            }
            
            # Check for corrected exit analysis
            trade_pattern_analysis = analysis_data.get('trade_pattern_analysis', {})
            if isinstance(trade_pattern_analysis, dict):
                if trade_pattern_analysis.get('exit_analysis_corrected'):
                    validation['corrected_features_detected'].append('exit_analysis_corrected')
                
                pattern = trade_pattern_analysis.get('pattern', 'unknown')
                tp_sl_analysis = trade_pattern_analysis.get('tp_sl_analysis', {})
                
                if isinstance(tp_sl_analysis, dict):
                    tp1 = ZeusUtils._safe_float(tp_sl_analysis.get('avg_tp1', 50), 50)
                    tp2 = ZeusUtils._safe_float(tp_sl_analysis.get('avg_tp2', 120), 120)
                    
                    # Validate TP/SL makes sense for pattern
                    inflation_check = ZeusUtils.detect_inflated_tp_sl(pattern, tp1, tp2)
                    if inflation_check['inflated']:
                        validation['valid'] = False
                        validation['issues'].extend(inflation_check['issues'])
                        validation['recommendations'].append(
                            f"Use corrected TP levels: {inflation_check['corrected_tp1']}%/{inflation_check['corrected_tp2']}%"
                        )
            
            # Check strategy recommendation consistency
            strategy = analysis_data.get('strategy_recommendation', {})
            if isinstance(strategy, dict):
                copy_exits = strategy.get('copy_exits', False)
                tp1 = ZeusUtils._safe_int(strategy.get('tp1_percent', 50), 50)
                
                # If copying exits, TP levels should be moderate
                if copy_exits and tp1 > 200:
                    validation['issues'].append('High TP levels with copy_exits strategy')
                    validation['recommendations'].append('Consider lower TP levels when copying exits')
            
            return validation
            
        except Exception as e:
            return {
                'valid': False,
                'issues': [f'Validation error: {str(e)}'],
                'corrected_features_detected': [],
                'recommendations': []
            }

class PerformanceTimer:
    """Context manager for timing operations with enhanced logging and SAFE validation."""
    
    def __init__(self, operation_name: str = "Operation", log_threshold_seconds: float = 1.0):
        self.operation_name = str(operation_name) if operation_name else "Operation"
        self.log_threshold_seconds = float(log_threshold_seconds) if isinstance(log_threshold_seconds, (int, float)) else 1.0
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type:
            logger.error(f"{self.operation_name} failed after {duration:.2f}s")
        else:
            # Only log if duration exceeds threshold
            if duration >= self.log_threshold_seconds:
                logger.info(f"{self.operation_name} completed in {duration:.2f}s")
            else:
                logger.debug(f"{self.operation_name} completed in {duration:.2f}s")
    
    @property
    def duration(self) -> float:
        """Get operation duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

def setup_zeus_logging(level: str = "INFO", log_file: str = "zeus.log", 
                      enable_enhanced_logging: bool = True) -> None:
    """
    Setup Zeus logging configuration with enhanced features and SAFE validation.
    
    Args:
        level: Logging level
        log_file: Log file path
        enable_enhanced_logging: Enable enhanced logging features
    """
    import sys
    
    try:
        # SAFE parameter validation
        if not isinstance(level, str):
            level = "INFO"
        if not isinstance(log_file, str):
            log_file = "zeus.log"
        
        # Create formatter
        if enable_enhanced_logging:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Setup file handler with rotation
        if enable_enhanced_logging:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=3, encoding='utf-8'
            )
        else:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger("zeus")
        root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        root_logger.propagate = False
        
        if enable_enhanced_logging:
            logger.info("Enhanced logging enabled with CORRECTED EXIT ANALYSIS support")
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")