"""
Zeus Export - FIXED with Safe Data Processing and Type Validation
CRITICAL FIXES:
- Safe handling of all data types from analysis results
- No more type comparison errors in field extraction
- Defensive programming with proper error handling
- Preserved all existing export functionality
- Enhanced data validation throughout the export pipeline
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
    """Export Zeus analysis results to CSV with SAFE Cielo field extraction."""
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
        
        # Prepare CSV data with SAFE field extraction
        csv_data = []
        
        for analysis in analyses:
            if not isinstance(analysis, dict):
                logger.debug("Skipping invalid analysis entry")
                continue
                
            if not analysis.get('success'):
                csv_data.append(_create_failed_row_safe(analysis))
                continue
            
            # Create analysis row with SAFE Cielo field values
            csv_data.append(_create_safe_cielo_analysis_row(analysis))
        
        # Sort by composite score (highest first) with SAFE comparison
        csv_data.sort(key=lambda x: _safe_float(x.get('composite_score', 0), 0), reverse=True)
        
        # Write CSV with SAFE handling
        if csv_data:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = _get_updated_csv_fieldnames()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        logger.info(f"✅ Exported {len(csv_data)} wallet analyses with SAFE CIELO FIELD VALUES to: {output_file}")
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

def _create_safe_cielo_analysis_row(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create CSV row with SAFE Cielo field extraction using actual field names."""
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
        
        logger.info(f"📊 SAFE CIELO FIELD EXTRACTION: {wallet_address[:8]}...")
        
        # Get timestamp data with SAFE extraction (1 decimal precision)
        real_days_since_last = _extract_days_since_last_trade_safe(analysis)
        
        # Extract SAFE Cielo field values using actual field names
        cielo_values = _extract_safe_cielo_fields(wallet_address, wallet_data)
        
        # Get real TP/SL recommendations from trade analysis
        tp_sl_values = _extract_tp_sl_recommendations_safe(analysis)
        
        # Create the row with SAFE field values
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
            'tp_1': tp_sl_values['tp1'],
            'tp_2': tp_sl_values['tp2'],
            'stop_loss': tp_sl_values['stop_loss'],
            'avg_sol_buy_per_token': cielo_values['avg_sol_buy_per_token'],
            'avg_buys_per_token': cielo_values['avg_buys_per_token'],
            'average_holding_time_minutes': cielo_values['avg_hold_time_minutes'],
            'unique_tokens_30d': cielo_values['unique_tokens_30d'],
            'trader_pattern': _identify_trader_pattern_safe(analysis),
            'strategy_reason': _generate_strategy_reasoning_safe(analysis),
            'decision_reason': _generate_decision_reasoning_safe(analysis)
        }
        
        logger.info(f"  SAFE CIELO VALUES for {wallet_address[:8]}:")
        logger.info(f"    roi_7_day: {cielo_values['roi_7_day']}%")
        logger.info(f"    winrate_7_day: {cielo_values['winrate_7_day']}%")
        logger.info(f"    unique_tokens_30d: {cielo_values['unique_tokens_30d']}")
        logger.info(f"    usd_profit_7_days: ${cielo_values['usd_profit_7_days']}")
        logger.info(f"    avg_hold_time_minutes: {cielo_values['avg_hold_time_minutes']}")
        
        return row
        
    except Exception as e:
        logger.error(f"Error creating safe Cielo analysis row: {str(e)}")
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
    CRITICAL FIX: No more type comparison errors!
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
        
        logger.info(f"Available Cielo API fields: {list(cielo_data.keys())}")
        
        # 1. SAFE EXTRACT WINRATE - Field name: 'winrate' (CONFIRMED from debug)
        if 'winrate' in cielo_data:
            try:
                winrate_value = cielo_data['winrate']
                if isinstance(winrate_value, (int, float)):
                    values['winrate_7_day'] = round(float(winrate_value), 2)
                    logger.info(f"✅ FOUND WINRATE: {winrate_value}%")
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to parse winrate: {e}")
        
        # 2. SAFE CALCULATE ROI from PnL and total_buy_amount_usd (CONFIRMED from debug)
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
                        logger.info(f"✅ CALCULATED ROI: {pnl} / {total_buy} * 100 = {roi_percent:.2f}%")
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to calculate ROI: {e}")
        
        # 3. SAFE EXTRACT UNIQUE TOKENS from holding_distribution.total_tokens (CONFIRMED from debug)
        holding_dist = cielo_data.get('holding_distribution')
        if isinstance(holding_dist, dict) and 'total_tokens' in holding_dist:
            try:
                tokens_count = holding_dist['total_tokens']
                if isinstance(tokens_count, (int, float)):
                    values['unique_tokens_30d'] = int(tokens_count)
                    logger.info(f"✅ FOUND UNIQUE TOKENS: {tokens_count}")
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to parse unique tokens: {e}")
        
        # 4. SAFE CONVERT HOLD TIME from average_holding_time_sec to minutes (CONFIRMED from debug)
        hold_time_sec = cielo_data.get('average_holding_time_sec')
        if hold_time_sec is not None:
            try:
                if isinstance(hold_time_sec, (int, float)):
                    seconds = float(hold_time_sec)
                    minutes = seconds / 60.0
                    values['avg_hold_time_minutes'] = round(minutes, 1)
                    logger.info(f"✅ CONVERTED HOLD TIME: {seconds}s = {minutes:.1f}min")
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to convert hold time: {e}")
        
        # 5. SAFE CALCULATE TIME-BASED PROFITS using consecutive_trading_days (CONFIRMED from debug)
        pnl_field = cielo_data.get('pnl')
        trading_days = cielo_data.get('consecutive_trading_days')
        
        if pnl_field is not None and trading_days is not None:
            try:
                if isinstance(pnl_field, (int, float)) and isinstance(trading_days, (int, float)):
                    pnl = float(pnl_field)
                    days = max(1, int(trading_days))
                    daily_profit = pnl / days
                    
                    values['usd_profit_2_days'] = round(daily_profit * 2, 1)
                    values['usd_profit_7_days'] = round(min(pnl, daily_profit * 7), 1)  # Cap at total PnL
                    values['usd_profit_30_days'] = round(pnl, 1)
                    
                    logger.info(f"✅ CALCULATED TIME-BASED PROFITS:")
                    logger.info(f"   Daily profit: ${daily_profit:.1f} (PnL: ${pnl} / {days} days)")
                    logger.info(f"   2-day: ${values['usd_profit_2_days']}")
                    logger.info(f"   7-day: ${values['usd_profit_7_days']}")
                    logger.info(f"   30-day: ${values['usd_profit_30_days']}")
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to calculate time-based profits: {e}")
        
        # 6. SAFE CONVERT VOLUME METRICS (USD to SOL) using average_buy_amount_usd (CONFIRMED from debug)
        avg_buy_usd = cielo_data.get('average_buy_amount_usd')
        if avg_buy_usd is not None:
            try:
                if isinstance(avg_buy_usd, (int, float)):
                    avg_usd = float(avg_buy_usd)
                    sol_price_estimate = 100.0  # Rough SOL price estimate
                    values['avg_sol_buy_per_token'] = round(avg_usd / sol_price_estimate, 1)
                    logger.info(f"✅ CONVERTED VOLUME: ${avg_usd} / ${sol_price_estimate} = {values['avg_sol_buy_per_token']} SOL")
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to convert volume: {e}")
        
        # 7. SAFE CALCULATE TRADE FREQUENCY using buy_count and unique_tokens (CONFIRMED from debug)
        buy_count = cielo_data.get('buy_count')
        unique_tokens = values['unique_tokens_30d']
        
        if buy_count is not None and unique_tokens > 0:
            try:
                if isinstance(buy_count, (int, float)):
                    buys = int(buy_count)
                    tokens = unique_tokens
                    values['avg_buys_per_token'] = round(buys / tokens, 1)
                    logger.info(f"✅ CALCULATED TRADE FREQUENCY: {buys} buys / {tokens} tokens = {values['avg_buys_per_token']}")
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to calculate trade frequency: {e}")
        
        logger.info(f"FINAL SAFE CIELO VALUES for {wallet_address[:8]}:")
        for key, value in values.items():
            logger.info(f"  {key}: {value}")
        
        return values
        
    except Exception as e:
        logger.error(f"Error extracting safe Cielo fields: {str(e)}")
        return values

def _extract_tp_sl_recommendations_safe(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Extract TP/SL recommendations from trade analysis data with SAFE validation."""
    try:
        if not isinstance(analysis, dict):
            return {
                'tp1': 75,
                'tp2': 200,
                'stop_loss': -35
            }
        
        strategy = analysis.get('strategy_recommendation', {})
        if not isinstance(strategy, dict):
            strategy = {}
        
        # Get values with safe defaults and validation
        tp1 = _safe_int(strategy.get('tp1_percent', 75), 75)
        tp2 = _safe_int(strategy.get('tp2_percent', 200), 200)
        stop_loss = _safe_int(strategy.get('stop_loss_percent', -35), -35)
        
        # SAFE range validation
        tp1 = max(10, min(500, tp1))
        tp2 = max(tp1 + 20, min(1000, tp2))
        stop_loss = max(-75, min(-10, stop_loss))
        
        return {
            'tp1': tp1,
            'tp2': tp2,
            'stop_loss': stop_loss
        }
        
    except Exception as e:
        logger.error(f"Error extracting TP/SL recommendations: {str(e)}")
        return {
            'tp1': 75,
            'tp2': 200,
            'stop_loss': -35
        }

def _identify_trader_pattern_safe(analysis: Dict[str, Any]) -> str:
    """Identify trader pattern based on analysis data with SAFE validation."""
    try:
        if not isinstance(analysis, dict):
            return 'insufficient_data'
        
        # Check if we have trade pattern analysis
        trade_pattern_analysis = analysis.get('trade_pattern_analysis', {})
        if isinstance(trade_pattern_analysis, dict) and 'pattern' in trade_pattern_analysis:
            pattern = trade_pattern_analysis['pattern']
            if isinstance(pattern, str):
                return pattern
        
        # Fallback to token analysis with SAFE validation
        token_analysis = analysis.get('token_analysis', [])
        
        if not isinstance(token_analysis, list) or not token_analysis:
            return 'insufficient_data'
        
        completed_trades = []
        for t in token_analysis:
            if isinstance(t, dict) and t.get('trade_status') == 'completed':
                completed_trades.append(t)
        
        if len(completed_trades) < 2:
            return 'new_trader'
        
        # Calculate metrics with SAFE extraction
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
        
        # Pattern identification with updated thresholds and SAFE comparisons
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
        logger.error(f"Error identifying trader pattern: {str(e)}")
        return 'analysis_error'

def _generate_strategy_reasoning_safe(analysis: Dict[str, Any]) -> str:
    """Generate strategy reasoning with SAFE validation."""
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
        
        # Get timestamp info with SAFE extraction
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
        
        # Strategy decision
        if follow_sells:
            reasoning_parts.append("Mirror their exits")
        else:
            tp1 = _safe_int(strategy.get('tp1_percent', 0), 0)
            tp2 = _safe_int(strategy.get('tp2_percent', 0), 0)
            reasoning_parts.append(f"Custom exits: {tp1}%-{tp2}%")
        
        return " | ".join(reasoning_parts)
        
    except Exception as e:
        logger.error(f"Error generating strategy reasoning: {str(e)}")
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
        
        # Follow sells decision
        if follow_wallet:
            if follow_sells:
                reasoning_parts.append("COPY EXITS: Good exit discipline")
            else:
                reasoning_parts.append("CUSTOM EXITS: Poor exit quality")
        
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
    """Export Zeus analysis summary to text file with SAFE validation."""
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
            f.write("ZEUS WALLET ANALYSIS SUMMARY - SAFE CIELO FIELD VALUES\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("📊 ANALYSIS OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Wallets: {len(successful_analyses)}\n")
            f.write(f"Data Source: SAFE Cielo API field extraction\n")
            f.write(f"Field Mappings: Updated to match actual Cielo response\n")
            f.write(f"ROI Calculation: PnL / total_buy_amount_usd * 100\n")
            f.write(f"Time Profits: Based on consecutive_trading_days\n")
            f.write(f"Hold Time: Converted from seconds to minutes\n")
            f.write(f"Token Count: Extracted from holding_distribution.total_tokens\n")
            f.write(f"Validation: SAFE type checking throughout\n\n")
        
        logger.info(f"✅ Exported Zeus summary to: {output_file}")
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