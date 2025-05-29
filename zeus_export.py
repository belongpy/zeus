"""
Zeus Export - Updated for Corrected TP/SL Export
MAJOR UPDATES:
- Export corrected TP/SL values that are realistic and actionable
- Handle corrected exit analysis data properly
- Preserve all existing export functionality
- Enhanced logging for corrected values
"""

import os
import csv
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

logger = logging.getLogger("zeus.export")

def export_zeus_analysis(results: Dict[str, Any], output_file: str) -> bool:
    """Export Zeus analysis results to CSV with CORRECTED TP/SL values."""
    try:
        # SAFE input validation
        if not isinstance(results, dict):
            logger.error("Results must be a dictionary")
            return False
        
        if not isinstance(output_file, str) or not output_file.strip():
            logger.error("Output file must be a valid string")
            return False
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Extract analyses with SAFE validation
        analyses = results.get('analyses', [])
        if not isinstance(analyses, list):
            logger.warning("No valid analyses list found")
            analyses = []
        
        if not analyses:
            logger.warning("No analyses to export")
            return False
        
        # Prepare CSV data with CORRECTED TP/SL values
        csv_data = []
        
        for analysis in analyses:
            if not isinstance(analysis, dict):
                logger.debug("Skipping invalid analysis entry")
                continue
                
            if not analysis.get('success'):
                csv_data.append(_create_failed_row_safe(analysis))
                continue
            
            # Create analysis row with CORRECTED Cielo field values and TP/SL
            csv_data.append(_create_corrected_analysis_row(analysis))
        
        # Sort by composite score (highest first) with SAFE comparison
        csv_data.sort(key=lambda x: _safe_float(x.get('composite_score', 0), 0), reverse=True)
        
        # Write CSV with CORRECTED values
        if csv_data:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = _get_updated_csv_fieldnames()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        logger.info(f"✅ Exported {len(csv_data)} wallet analyses with CORRECTED TP/SL VALUES to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error exporting Zeus analysis: {str(e)}")
        return False

def _get_updated_csv_fieldnames() -> List[str]:
    """Get updated CSV fieldnames."""
    return [
        'wallet_address',
        'composite_score',
        '7_day_winrate',
        'days_since_last_trade',
        'roi_7_day',
        'usd_profit_2_days',
        'usd_profit_7_days', 
        'usd_profit_30_days',
        'copy_wallet',
        'copy_sells',
        'tp_1',
        'tp_2',
        'stop_loss',
        'avg_sol_buy_per_token',
        'avg_buys_per_token',
        'average_holding_time_minutes',
        'unique_tokens_30d',
        'trader_pattern',
        'strategy_reason',
        'decision_reason'
    ]

def _create_corrected_analysis_row(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create CSV row with CORRECTED TP/SL values and SAFE Cielo field extraction."""
    try:
        # SAFE extraction of basic data
        wallet_address = str(analysis.get('wallet_address', ''))
        binary_decisions = analysis.get('binary_decisions', {})
        strategy = analysis.get('strategy_recommendation', {})
        wallet_data = analysis.get('wallet_data', {})
        
        # Ensure nested data is valid
        if not isinstance(binary_decisions, dict):
            binary_decisions = {}
        if not isinstance(strategy, dict):
            strategy = {}
        if not isinstance(wallet_data, dict):
            wallet_data = {}
        
        logger.info(f"📊 CORRECTED CSV EXPORT: {wallet_address[:8]}...")
        
        # Get timestamp data with SAFE extraction (1 decimal precision)
        real_days_since_last = _extract_days_since_last_trade_safe(analysis)
        
        # Extract SAFE Cielo field values using actual field names
        cielo_values = _extract_safe_cielo_fields(wallet_address, wallet_data)
        
        # Get CORRECTED TP/SL recommendations
        corrected_tp_sl = _extract_corrected_tp_sl_recommendations(analysis)
        
        # Create the row with CORRECTED values
        row = {
            'wallet_address': wallet_address,
            'composite_score': round(_safe_float(analysis.get('composite_score', 0), 0), 1),
            '7_day_winrate': cielo_values['winrate_7_day'],
            'days_since_last_trade': real_days_since_last,
            'roi_7_day': cielo_values['roi_7_day'],
            'usd_profit_2_days': cielo_values['usd_profit_2_days'],
            'usd_profit_7_days': cielo_values['usd_profit_7_days'],
            'usd_profit_30_days': cielo_values['usd_profit_30_days'],
            'copy_wallet': 'YES' if binary_decisions.get('follow_wallet', False) else 'NO',
            'copy_sells': 'YES' if binary_decisions.get('follow_sells', False) else 'NO',
            'tp_1': corrected_tp_sl['tp1'],  # CORRECTED VALUES
            'tp_2': corrected_tp_sl['tp2'],  # CORRECTED VALUES
            'stop_loss': corrected_tp_sl['stop_loss'],  # CORRECTED VALUES
            'avg_sol_buy_per_token': cielo_values['avg_sol_buy_per_token'],
            'avg_buys_per_token': cielo_values['avg_buys_per_token'],
            'average_holding_time_minutes': cielo_values['avg_hold_time_minutes'],
            'unique_tokens_30d': cielo_values['unique_tokens_30d'],
            'trader_pattern': _identify_corrected_trader_pattern(analysis),
            'strategy_reason': _generate_corrected_strategy_reasoning(analysis),
            'decision_reason': _generate_decision_reasoning_safe(analysis)
        }
        
        logger.info(f"  CORRECTED TP/SL for {wallet_address[:8]}:")
        logger.info(f"    Pattern: {row['trader_pattern']}")
        logger.info(f"    TP1: {corrected_tp_sl['tp1']}% (CORRECTED)")
        logger.info(f"    TP2: {corrected_tp_sl['tp2']}% (CORRECTED)")
        logger.info(f"    SL: {corrected_tp_sl['stop_loss']}% (CORRECTED)")
        logger.info(f"    Reasoning: {corrected_tp_sl['reasoning']}")
        
        return row
        
    except Exception as e:
        logger.error(f"Error creating corrected analysis row: {str(e)}")
        return _create_error_row_safe(analysis, str(e))

def _extract_days_since_last_trade_safe(analysis: Dict[str, Any]) -> float:
    """Extract days since last trade from Helius timestamp data with SAFE validation (1 decimal precision)."""
    try:
        if not isinstance(analysis, dict):
            return 999.0
        
        last_tx_data = analysis.get('last_transaction_data', {})
        if isinstance(last_tx_data, dict) and last_tx_data.get('success', False):
            days_since = last_tx_data.get('days_since_last_trade', 999)
            if isinstance(days_since, (int, float)) and days_since >= 0:
                return round(float(days_since), 1)  # 1 decimal precision
        return 999.0
    except Exception as e:
        logger.error(f"Error extracting days since last trade: {str(e)}")
        return 999.0

def _extract_safe_cielo_fields(wallet_address: str, wallet_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract SAFE field values from Cielo Trading Stats API response using ACTUAL field names.
    """
    try:
        logger.info(f"📊 SAFE CIELO FIELD EXTRACTION for {wallet_address[:8]}...")
        
        # Initialize with safe defaults
        values = {
            'roi_7_day': 0.0,
            'winrate_7_day': 0.0,
            'avg_hold_time_minutes': 0.0,
            'usd_profit_2_days': 0.0,
            'usd_profit_7_days': 0.0,
            'usd_profit_30_days': 0.0,
            'avg_sol_buy_per_token': 0.0,
            'avg_buys_per_token': 0.0,
            'unique_tokens_30d': 0
        }
        
        # SAFE extraction of Cielo data from nested structure
        cielo_data = None
        if isinstance(wallet_data, dict):
            if 'data' in wallet_data and isinstance(wallet_data['data'], dict):
                cielo_data = wallet_data['data']
            elif wallet_data.get('source') in ['cielo_trading_stats', 'cielo_finance_real', 'cielo_trading_stats_safe']:
                cielo_data = wallet_data.get('data', {})
            else:
                cielo_data = wallet_data
        
        if not isinstance(cielo_data, dict):
            logger.warning(f"No valid Cielo data found for {wallet_address[:8]}")
            return values
        
        # Extract values using actual Cielo field names
        # 1. Winrate
        if 'winrate' in cielo_data:
            try:
                winrate_value = cielo_data['winrate']
                if isinstance(winrate_value, (int, float)):
                    values['winrate_7_day'] = round(float(winrate_value), 2)
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to parse winrate: {e}")
        
        # 2. ROI calculation from PnL and total_buy_amount_usd
        pnl_field = cielo_data.get('pnl')
        buy_amount_field = cielo_data.get('total_buy_amount_usd')
        
        if pnl_field is not None and buy_amount_field is not None:
            try:
                if isinstance(pnl_field, (int, float)) and isinstance(buy_amount_field, (int, float)):
                    pnl = float(pnl_field)
                    total_buy = float(buy_amount_field)
                    if total_buy > 0:
                        roi_percent = (pnl / total_buy) * 100
                        values['roi_7_day'] = round(roi_percent, 2)
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to calculate ROI: {e}")
        
        # 3. Unique tokens from holding_distribution.total_tokens
        holding_dist = cielo_data.get('holding_distribution')
        if isinstance(holding_dist, dict) and 'total_tokens' in holding_dist:
            try:
                tokens_count = holding_dist['total_tokens']
                if isinstance(tokens_count, (int, float)):
                    values['unique_tokens_30d'] = int(tokens_count)
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to parse unique tokens: {e}")
        
        # 4. Hold time conversion from seconds to minutes
        hold_time_sec = cielo_data.get('average_holding_time_sec')
        if hold_time_sec is not None:
            try:
                if isinstance(hold_time_sec, (int, float)):
                    seconds = float(hold_time_sec)
                    minutes = seconds / 60.0
                    values['avg_hold_time_minutes'] = round(minutes, 1)
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to convert hold time: {e}")
        
        # 5. Time-based profits using consecutive_trading_days
        pnl_field = cielo_data.get('pnl')
        trading_days = cielo_data.get('consecutive_trading_days')
        
        if pnl_field is not None and trading_days is not None:
            try:
                if isinstance(pnl_field, (int, float)) and isinstance(trading_days, (int, float)):
                    pnl = float(pnl_field)
                    days = max(1, int(trading_days))
                    daily_profit = pnl / days
                    
                    values['usd_profit_2_days'] = round(daily_profit * 2, 1)
                    values['usd_profit_7_days'] = round(min(pnl, daily_profit * 7), 1)
                    values['usd_profit_30_days'] = round(pnl, 1)
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to calculate time-based profits: {e}")
        
        # 6. Volume metrics conversion
        avg_buy_usd = cielo_data.get('average_buy_amount_usd')
        if avg_buy_usd is not None:
            try:
                if isinstance(avg_buy_usd, (int, float)):
                    avg_usd = float(avg_buy_usd)
                    sol_price_estimate = 100.0  # Rough SOL price estimate
                    values['avg_sol_buy_per_token'] = round(avg_usd / sol_price_estimate, 1)
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to convert volume: {e}")
        
        # 7. Trade frequency calculation
        buy_count = cielo_data.get('buy_count')
        unique_tokens = values['unique_tokens_30d']
        
        if buy_count is not None and unique_tokens > 0:
            try:
                if isinstance(buy_count, (int, float)):
                    buys = int(buy_count)
                    tokens = unique_tokens
                    values['avg_buys_per_token'] = round(buys / tokens, 1)
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to calculate trade frequency: {e}")
        
        return values
        
    except Exception as e:
        logger.error(f"Error extracting safe Cielo fields: {str(e)}")
        return values

def _extract_corrected_tp_sl_recommendations(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract CORRECTED TP/SL recommendations that are realistic and actionable.
    MAJOR FIX: Use corrected exit analysis data, not inflated final ROI values.
    """
    try:
        if not isinstance(analysis, dict):
            return {
                'tp1': 50,
                'tp2': 120,
                'stop_loss': -30,
                'reasoning': 'Default - no analysis data'
            }
        
        # Check for corrected trade pattern analysis
        trade_pattern_analysis = analysis.get('trade_pattern_analysis', {})
        
        if isinstance(trade_pattern_analysis, dict) and trade_pattern_analysis.get('exit_analysis_corrected'):
            # Use CORRECTED exit analysis data
            tp_sl_analysis = trade_pattern_analysis.get('tp_sl_analysis', {})
            pattern = trade_pattern_analysis.get('pattern', 'mixed_strategy')
            
            if isinstance(tp_sl_analysis, dict) and tp_sl_analysis.get('corrected_analysis'):
                # Extract CORRECTED values
                tp1 = _safe_int(tp_sl_analysis.get('avg_tp1', 50), 50)
                tp2 = _safe_int(tp_sl_analysis.get('avg_tp2', 120), 120)
                stop_loss = _safe_int(tp_sl_analysis.get('avg_stop_loss', -30), -30)
                
                # Validate ranges based on pattern
                tp1, tp2, stop_loss = _validate_corrected_tp_sl_for_pattern(pattern, tp1, tp2, stop_loss)
                
                reasoning = f"CORRECTED {pattern} analysis"
                if tp_sl_analysis.get('based_on_actual_exits'):
                    reasoning += " (based on actual exits)"
                
                logger.info(f"✅ Using CORRECTED exit analysis: {pattern} - TP1: {tp1}%, TP2: {tp2}%, SL: {stop_loss}%")
                
                return {
                    'tp1': tp1,
                    'tp2': tp2,
                    'stop_loss': stop_loss,
                    'reasoning': reasoning,
                    'corrected': True
                }
        
        # Fallback to strategy recommendation
        strategy = analysis.get('strategy_recommendation', {})
        if isinstance(strategy, dict):
            tp1 = _safe_int(strategy.get('tp1_percent', 50), 50)
            tp2 = _safe_int(strategy.get('tp2_percent', 120), 120)
            stop_loss = _safe_int(strategy.get('stop_loss_percent', -30), -30)
            
            # Identify pattern for validation
            pattern = _identify_corrected_trader_pattern(analysis)
            tp1, tp2, stop_loss = _validate_corrected_tp_sl_for_pattern(pattern, tp1, tp2, stop_loss)
            
            return {
                'tp1': tp1,
                'tp2': tp2,
                'stop_loss': stop_loss,
                'reasoning': f"Strategy recommendation ({pattern})",
                'corrected': False
            }
        
        # Final fallback - pattern-based defaults
        pattern = _identify_corrected_trader_pattern(analysis)
        defaults = _get_pattern_based_tp_sl_defaults(pattern)
        
        return {
            'tp1': defaults['tp1'],
            'tp2': defaults['tp2'],
            'stop_loss': defaults['stop_loss'],
            'reasoning': f"Pattern-based defaults ({pattern})",
            'corrected': False
        }
        
    except Exception as e:
        logger.error(f"Error extracting corrected TP/SL recommendations: {str(e)}")
        return {
            'tp1': 50,
            'tp2': 120,
            'stop_loss': -30,
            'reasoning': f'Error: {str(e)}'
        }

def _validate_corrected_tp_sl_for_pattern(pattern: str, tp1: int, tp2: int, stop_loss: int) -> tuple:
    """Validate CORRECTED TP/SL levels to ensure they make sense for the trading pattern."""
    try:
        if pattern == 'flipper':
            # Flippers should have LOW TP levels - they exit quickly
            tp1 = max(15, min(60, tp1))
            tp2 = max(tp1 + 10, min(80, tp2))
            stop_loss = max(-25, min(-10, stop_loss))
            
        elif pattern == 'sniper':
            # Snipers take quick profits but slightly higher than flippers
            tp1 = max(25, min(80, tp1))
            tp2 = max(tp1 + 15, min(120, tp2))
            stop_loss = max(-30, min(-15, stop_loss))
            
        elif pattern == 'gem_hunter':
            # Gem hunters can have higher TPs but still realistic
            tp1 = max(50, min(200, tp1))
            tp2 = max(tp1 + 50, min(400, tp2))
            stop_loss = max(-50, min(-25, stop_loss))
            
        elif pattern == 'position_trader':
            # Position traders hold longer for bigger gains
            tp1 = max(60, min(150, tp1))
            tp2 = max(tp1 + 40, min(300, tp2))
            stop_loss = max(-40, min(-20, stop_loss))
            
        elif pattern == 'consistent_trader':
            # Consistent traders have moderate, balanced levels
            tp1 = max(40, min(100, tp1))
            tp2 = max(tp1 + 30, min(200, tp2))
            stop_loss = max(-35, min(-20, stop_loss))
            
        else:
            # Default/mixed strategy - conservative levels
            tp1 = max(30, min(80, tp1))
            tp2 = max(tp1 + 20, min(150, tp2))
            stop_loss = max(-35, min(-20, stop_loss))
        
        return tp1, tp2, stop_loss
        
    except Exception as e:
        logger.error(f"Error validating TP/SL for pattern {pattern}: {str(e)}")
        return 50, 120, -30

def _get_pattern_based_tp_sl_defaults(pattern: str) -> Dict[str, int]:
    """Get realistic pattern-based TP/SL defaults."""
    patterns = {
        'flipper': {'tp1': 25, 'tp2': 45, 'stop_loss': -15},
        'sniper': {'tp1': 40, 'tp2': 70, 'stop_loss': -20},
        'gem_hunter': {'tp1': 100, 'tp2': 250, 'stop_loss': -40},
        'position_trader': {'tp1': 80, 'tp2': 180, 'stop_loss': -30},
        'consistent_trader': {'tp1': 60, 'tp2': 130, 'stop_loss': -25},
        'bag_holder': {'tp1': 40, 'tp2': 100, 'stop_loss': -20},
        'impulsive_trader': {'tp1': 30, 'tp2': 60, 'stop_loss': -25}
    }
    
    return patterns.get(pattern, {'tp1': 50, 'tp2': 120, 'stop_loss': -30})

def _identify_corrected_trader_pattern(analysis: Dict[str, Any]) -> str:
    """Identify trader pattern with CORRECTED analysis priority."""
    try:
        if not isinstance(analysis, dict):
            return 'insufficient_data'
        
        # Check for corrected trade pattern analysis first
        trade_pattern_analysis = analysis.get('trade_pattern_analysis', {})
        if isinstance(trade_pattern_analysis, dict) and trade_pattern_analysis.get('exit_analysis_corrected'):
            pattern = trade_pattern_analysis.get('pattern', 'mixed_strategy')
            if isinstance(pattern, str):
                return pattern
        
        # Fallback to regular trade pattern analysis
        if isinstance(trade_pattern_analysis, dict) and 'pattern' in trade_pattern_analysis:
            pattern = trade_pattern_analysis['pattern']
            if isinstance(pattern, str):
                return pattern
        
        # Final fallback to token analysis
        token_analysis = analysis.get('token_analysis', [])
        
        if not isinstance(token_analysis, list) or not token_analysis:
            return 'insufficient_data'
        
        completed_trades = []
        for t in token_analysis:
            if isinstance(t, dict) and t.get('trade_status') == 'completed':
                completed_trades.append(t)
        
        if len(completed_trades) < 2:
            return 'new_trader'
        
        # Calculate metrics for pattern identification
        rois = []
        hold_times = []
        
        for t in completed_trades:
            roi = _safe_float(t.get('roi_percent', 0), 0)
            hold_time = _safe_float(t.get('hold_time_hours', 0), 0)
            rois.append(roi)
            hold_times.append(hold_time)
        
        if not rois or not hold_times:
            return 'insufficient_data'
        
        avg_roi = sum(rois) / len(rois)
        avg_hold_time = sum(hold_times) / len(hold_times)
        moonshots = sum(1 for roi in rois if roi >= 400)
        
        # Pattern identification with CORRECTED thresholds
        if avg_hold_time < 0.083:  # Less than 5 minutes
            return 'flipper'
        elif avg_hold_time < 1:
            return 'sniper' if avg_roi > 30 else 'impulsive_trader'
        elif moonshots > 0:
            return 'gem_hunter'
        elif avg_hold_time > 24:  # More than 24 hours
            return 'position_trader' if avg_roi > 50 else 'bag_holder'
        elif avg_roi > 20:
            return 'consistent_trader'
        else:
            return 'mixed_strategy'
        
    except Exception as e:
        logger.error(f"Error identifying corrected trader pattern: {str(e)}")
        return 'analysis_error'

def _generate_corrected_strategy_reasoning(analysis: Dict[str, Any]) -> str:
    """Generate strategy reasoning with CORRECTED analysis priority."""
    try:
        if not isinstance(analysis, dict):
            return "Analysis data error"
        
        binary_decisions = analysis.get('binary_decisions', {})
        strategy = analysis.get('strategy_recommendation', {})
        
        if not isinstance(binary_decisions, dict):
            binary_decisions = {}
        if not isinstance(strategy, dict):
            strategy = {}
        
        follow_wallet = binary_decisions.get('follow_wallet', False)
        follow_sells = binary_decisions.get('follow_sells', False)
        
        if not follow_wallet:
            return "Not recommended - insufficient score"
        
        # Get timestamp info
        last_tx_data = analysis.get('last_transaction_data', {})
        days_since_last = 999
        if isinstance(last_tx_data, dict):
            days_since_last = _safe_float(last_tx_data.get('days_since_last_trade', 999), 999)
        
        reasoning_parts = []
        
        # Activity status
        if days_since_last <= 1:
            reasoning_parts.append("Very active")
        elif days_since_last <= 3:
            reasoning_parts.append("Recently active")
        elif days_since_last <= 7:
            reasoning_parts.append("Active")
        else:
            reasoning_parts.append("Less active")
        
        # Strategy decision with CORRECTED TP/SL
        pattern = _identify_corrected_trader_pattern(analysis)
        
        if follow_sells:
            reasoning_parts.append(f"Mirror {pattern} exits")
        else:
            # Get corrected TP levels
            corrected_tp_sl = _extract_corrected_tp_sl_recommendations(analysis)
            tp1 = corrected_tp_sl['tp1']
            tp2 = corrected_tp_sl['tp2']
            reasoning_parts.append(f"CORRECTED {pattern} exits: {tp1}%-{tp2}%")
        
        return " | ".join(reasoning_parts)
        
    except Exception as e:
        logger.error(f"Error generating corrected strategy reasoning: {str(e)}")
        return f"Strategy error: {str(e)}"

def _generate_decision_reasoning_safe(analysis: Dict[str, Any]) -> str:
    """Generate decision reasoning with SAFE validation."""
    try:
        if not isinstance(analysis, dict):
            return "Analysis data error"
        
        binary_decisions = analysis.get('binary_decisions', {})
        composite_score = _safe_float(analysis.get('composite_score', 0), 0)
        
        if not isinstance(binary_decisions, dict):
            binary_decisions = {}
        
        follow_wallet = binary_decisions.get('follow_wallet', False)
        follow_sells = binary_decisions.get('follow_sells', False)
        
        reasoning_parts = []
        
        # Follow wallet decision
        if follow_wallet:
            reasoning_parts.append(f"FOLLOW: Score {composite_score:.1f}/100")
        else:
            reasoning_parts.append(f"DON'T FOLLOW: Score {composite_score:.1f}/100")
        
        # Follow sells decision with CORRECTED context
        if follow_wallet:
            if follow_sells:
                reasoning_parts.append("COPY EXITS: Good exit discipline")
            else:
                pattern = _identify_corrected_trader_pattern(analysis)
                reasoning_parts.append(f"CUSTOM EXITS: Use CORRECTED {pattern} TP/SL")
        
        return " | ".join(reasoning_parts)
        
    except Exception as e:
        logger.error(f"Error generating decision reasoning: {str(e)}")
        return f"Decision error: {str(e)}"

def _create_failed_row_safe(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create CSV row for failed analysis with SAFE validation."""
    wallet_address = str(analysis.get('wallet_address', '')) if isinstance(analysis, dict) else ''
    error_message = str(analysis.get('error', 'Unknown error')) if isinstance(analysis, dict) else 'Unknown error'
    
    return {
        'wallet_address': wallet_address,
        'composite_score': 0,
        '7_day_winrate': 0.0,
        'days_since_last_trade': 999,
        'roi_7_day': 0.0,
        'usd_profit_2_days': 0.0,
        'usd_profit_7_days': 0.0,
        'usd_profit_30_days': 0.0,
        'copy_wallet': 'NO',
        'copy_sells': 'NO',
        'tp_1': 0,
        'tp_2': 0,
        'stop_loss': -35,
        'avg_sol_buy_per_token': 0.0,
        'avg_buys_per_token': 0.0,
        'average_holding_time_minutes': 0.0,
        'unique_tokens_30d': 0,
        'trader_pattern': 'failed_analysis',
        'strategy_reason': f"Analysis failed: {error_message}",
        'decision_reason': f"Cannot analyze: {error_message}"
    }

def _create_error_row_safe(analysis: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
    """Create error row when row creation fails with SAFE validation."""
    wallet_address = str(analysis.get('wallet_address', '')) if isinstance(analysis, dict) else ''
    
    return {
        'wallet_address': wallet_address,
        'composite_score': 0,
        '7_day_winrate': 0.0,
        'days_since_last_trade': 999,
        'roi_7_day': 0.0,
        'usd_profit_2_days': 0.0,
        'usd_profit_7_days': 0.0,
        'usd_profit_30_days': 0.0,
        'copy_wallet': 'NO',
        'copy_sells': 'NO',
        'tp_1': 0,
        'tp_2': 0,
        'stop_loss': -35,
        'avg_sol_buy_per_token': 0.0,
        'avg_buys_per_token': 0.0,
        'average_holding_time_minutes': 0.0,
        'unique_tokens_30d': 0,
        'trader_pattern': 'error',
        'strategy_reason': f'Error: {error_msg}',
        'decision_reason': f'Processing error: {error_msg}'
    }

def export_zeus_summary(results: Dict[str, Any], output_file: str) -> bool:
    """Export Zeus analysis summary to text file with CORRECTED TP/SL info."""
    try:
        # SAFE input validation
        if not isinstance(results, dict):
            logger.error("Results must be a dictionary")
            return False
        
        if not isinstance(output_file, str) or not output_file.strip():
            logger.error("Output file must be a valid string")
            return False
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        analyses = results.get('analyses', [])
        if not isinstance(analyses, list):
            analyses = []
        
        successful_analyses = []
        for a in analyses:
            if isinstance(a, dict) and a.get('success'):
                successful_analyses.append(a)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ZEUS WALLET ANALYSIS SUMMARY - CORRECTED TP/SL VALUES\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("📊 ANALYSIS OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Wallets: {len(successful_analyses)}\n")
            f.write(f"Data Source: SAFE Cielo API field extraction\n")
            f.write(f"Exit Analysis: CORRECTED to infer actual exit points\n")
            f.write(f"TP/SL Values: CORRECTED and realistic for each pattern\n")
            f.write(f"Flipper TP Range: 15-60% (quick exits)\n")
            f.write(f"Gem Hunter TP Range: 50-400% (patient for moonshots)\n")
            f.write(f"Position Trader TP Range: 60-300% (longer holds)\n")
            f.write(f"Validation: Pattern-based TP/SL limits enforced\n\n")
        
        logger.info(f"✅ Exported Zeus summary with CORRECTED TP/SL info to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error exporting Zeus summary: {str(e)}")
        return False

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safe float conversion with SAFE default."""
    try:
        if isinstance(value, (int, float)) and not (isinstance(value, float) and value != value):  # Check for NaN
            return float(value)
        else:
            return float(default)
    except:
        return float(default)

def _safe_int(value: Any, default: int = 0) -> int:
    """Safe int conversion with SAFE default."""
    try:
        if isinstance(value, (int, float)) and not (isinstance(value, float) and value != value):  # Check for NaN
            return int(value)
        else:
            return int(default)
    except:
        return int(default)