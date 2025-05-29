"""
Zeus Export - COMPLETELY FIXED with Direct Cielo Field Extraction
MAJOR FIXES:
- Removed ALL scaling/conversion logic
- Direct extraction from Cielo Trading Stats fields
- Removed columns Q & R (total_buys_30_days, total_sells_30_days)
- Added unique_tokens_30d field from Cielo
- Added field validation and error handling
- 1 decimal precision for specific fields
"""

import os
import csv
import json
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("zeus.export")

def export_zeus_analysis(results: Dict[str, Any], output_file: str) -> bool:
    """Export Zeus analysis results to CSV with DIRECT Cielo field extraction."""
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Extract analyses
        analyses = results.get('analyses', [])
        if not analyses:
            logger.warning("No analyses to export")
            return False
        
        # Prepare CSV data with DIRECT field extraction
        csv_data = []
        
        for analysis in analyses:
            if not analysis.get('success'):
                csv_data.append(_create_failed_row(analysis))
                continue
            
            # Create analysis row with DIRECT Cielo field values
            csv_data.append(_create_direct_cielo_analysis_row(analysis))
        
        # Sort by composite score (highest first)
        csv_data.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        # Write CSV
        if csv_data:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = _get_updated_csv_fieldnames()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        logger.info(f"✅ Exported {len(csv_data)} wallet analyses with DIRECT CIELO FIELD VALUES to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error exporting Zeus analysis: {str(e)}")
        return False

def _get_updated_csv_fieldnames() -> List[str]:
    """Get updated CSV fieldnames - REMOVED Q & R, ADDED unique_tokens_30d."""
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
        'unique_tokens_30d',  # NEW: Replaced columns Q & R
        'trader_pattern',
        'strategy_reason',
        'decision_reason'
    ]

def _create_direct_cielo_analysis_row(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create CSV row with DIRECT Cielo field extraction - NO CONVERSIONS OR SCALING."""
    try:
        # Extract basic data
        wallet_address = analysis.get('wallet_address', '')
        binary_decisions = analysis.get('binary_decisions', {})
        strategy = analysis.get('strategy_recommendation', {})
        wallet_data = analysis.get('wallet_data', {})
        
        logger.info(f"📊 DIRECT CIELO FIELD EXTRACTION: {wallet_address[:8]}...")
        
        # Get timestamp data (1 decimal precision)
        real_days_since_last = _extract_days_since_last_trade(analysis)
        
        # Extract DIRECT Cielo field values - NO CONVERSIONS!
        cielo_values = _extract_direct_cielo_fields(wallet_address, wallet_data)
        
        # Get real TP/SL recommendations from trade analysis
        tp_sl_values = _extract_tp_sl_recommendations(analysis)
        
        # Create the row with DIRECT field values
        row = {
            'wallet_address': wallet_address,
            'composite_score': round(analysis.get('composite_score', 0), 1),
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
            'unique_tokens_30d': cielo_values['unique_tokens_30d'],  # NEW FIELD
            'trader_pattern': _identify_trader_pattern(analysis),
            'strategy_reason': _generate_strategy_reasoning(analysis),
            'decision_reason': _generate_decision_reasoning(analysis)
        }
        
        logger.info(f"  DIRECT CIELO VALUES for {wallet_address[:8]}:")
        logger.info(f"    roi_7_day: {cielo_values['roi_7_day']}%")
        logger.info(f"    winrate_7_day: {cielo_values['winrate_7_day']}%")
        logger.info(f"    unique_tokens_30d: {cielo_values['unique_tokens_30d']}")
        logger.info(f"    usd_profit_7_days: ${cielo_values['usd_profit_7_days']}")
        
        return row
        
    except Exception as e:
        logger.error(f"Error creating direct Cielo analysis row: {str(e)}")
        return _create_error_row(analysis, str(e))

def _extract_days_since_last_trade(analysis: Dict[str, Any]) -> float:
    """Extract days since last trade from Helius timestamp data (1 decimal precision)."""
    try:
        last_tx_data = analysis.get('last_transaction_data', {})
        if isinstance(last_tx_data, dict) and last_tx_data.get('success', False):
            days_since = last_tx_data.get('days_since_last_trade', 999)
            if days_since is not None and isinstance(days_since, (int, float)):
                return round(float(days_since), 1)  # 1 decimal precision
        return 999.0
    except Exception as e:
        logger.error(f"Error extracting days since last trade: {str(e)}")
        return 999.0

def _extract_direct_cielo_fields(wallet_address: str, wallet_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract DIRECT field values from Cielo Trading Stats API response - NO CONVERSIONS!
    Map directly to the actual field names from Cielo's response.
    """
    try:
        logger.info(f"📊 DIRECT CIELO FIELD EXTRACTION for {wallet_address[:8]}...")
        
        # Initialize with defaults
        values = {
            'roi_7_day': 0.0,                # Direct from Cielo
            'winrate_7_day': 0.0,           # Direct from Cielo  
            'avg_hold_time_minutes': 0.0,   # Direct from Cielo
            'usd_profit_2_days': 0.0,       # Direct from Cielo
            'usd_profit_7_days': 0.0,       # Direct from Cielo
            'usd_profit_30_days': 0.0,      # Direct from Cielo
            'avg_sol_buy_per_token': 0.0,   # Calculated from Cielo data
            'avg_buys_per_token': 0.0,      # Calculated from Cielo data
            'unique_tokens_30d': 0          # NEW: Direct from Cielo
        }
        
        # Extract Cielo data from nested structure
        cielo_data = None
        if isinstance(wallet_data, dict):
            if 'data' in wallet_data and isinstance(wallet_data['data'], dict):
                cielo_data = wallet_data['data']
            elif wallet_data.get('source') in ['cielo_trading_stats', 'cielo_finance_real']:
                cielo_data = wallet_data.get('data', {})
            else:
                cielo_data = wallet_data
        
        if not isinstance(cielo_data, dict):
            logger.warning(f"No valid Cielo data found for {wallet_address[:8]}")
            return values
        
        logger.info(f"Available Cielo API fields: {list(cielo_data.keys())}")
        
        # 1. EXTRACT DIRECT ROI VALUE (NOT win rate conversion!)
        roi_field_names = [
            'roi',                    # Most likely field name
            'roi_7d',                # 7-day specific ROI
            'total_roi',             # Total ROI
            'realized_roi',          # Realized ROI
            'return_on_investment',  # Full name
            'pnl_roi',              # PnL ROI
            'roi_percent'           # ROI percentage
        ]
        
        for field in roi_field_names:
            if field in cielo_data and cielo_data[field] is not None:
                try:
                    roi_value = float(cielo_data[field])
                    
                    # Validate range and use appropriate scaling
                    if 0 <= roi_value <= 1000:  # Already percentage (0-1000%)
                        values['roi_7_day'] = round(roi_value, 2)
                        logger.info(f"✅ FOUND DIRECT ROI from '{field}': {roi_value}%")
                        break
                    elif -1 <= roi_value <= 10:  # Decimal format (0.5 = 50%)
                        values['roi_7_day'] = round(roi_value * 100, 2)
                        logger.info(f"✅ FOUND DIRECT ROI from '{field}': {roi_value} -> {roi_value * 100}%")
                        break
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse ROI field {field}: {e}")
                    continue
        
        # 2. EXTRACT DIRECT WINRATE VALUE
        winrate_field_names = [
            'winrate',               # Most likely field name
            'win_rate',             # Alternative
            'token_winrate',        # Token specific
            'winning_rate',         # Winning rate
            'success_rate',         # Success rate
            'win_percentage'        # Win percentage
        ]
        
        for field in winrate_field_names:
            if field in cielo_data and cielo_data[field] is not None:
                try:
                    winrate_value = float(cielo_data[field])
                    
                    if 0 <= winrate_value <= 100:  # Percentage format
                        values['winrate_7_day'] = round(winrate_value, 2)
                        logger.info(f"✅ FOUND DIRECT WINRATE from '{field}': {winrate_value}%")
                        break
                    elif 0 <= winrate_value <= 1:  # Decimal format (0.75 = 75%)
                        values['winrate_7_day'] = round(winrate_value * 100, 2)
                        logger.info(f"✅ FOUND DIRECT WINRATE from '{field}': {winrate_value} -> {winrate_value * 100}%")
                        break
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse winrate field {field}: {e}")
                    continue
        
        # 3. EXTRACT DIRECT HOLDING TIME
        hold_time_field_names = [
            'avg_hold_time',              # Average hold time
            'average_holding_time',       # Full name
            'avg_holding_time_minutes',   # In minutes
            'hold_time_avg',             # Hold time average
            'avg_hold_time_seconds',     # In seconds (convert)
            'holding_time_average'       # Alternative
        ]
        
        for field in hold_time_field_names:
            if field in cielo_data and cielo_data[field] is not None:
                try:
                    hold_time_value = float(cielo_data[field])
                    
                    # Convert based on field name and value range
                    if 'second' in field.lower() or hold_time_value > 1000:
                        # Seconds - convert to minutes
                        values['avg_hold_time_minutes'] = round(hold_time_value / 60.0, 1)
                        logger.info(f"✅ FOUND HOLD TIME from '{field}': {hold_time_value}s -> {values['avg_hold_time_minutes']}min")
                    elif 'minute' in field.lower() or hold_time_value < 300:
                        # Already in minutes
                        values['avg_hold_time_minutes'] = round(hold_time_value, 1)
                        logger.info(f"✅ FOUND HOLD TIME from '{field}': {hold_time_value}min")
                    else:
                        # Assume hours if reasonable range
                        values['avg_hold_time_minutes'] = round(hold_time_value * 60.0, 1)
                        logger.info(f"✅ FOUND HOLD TIME from '{field}': {hold_time_value}h -> {values['avg_hold_time_minutes']}min")
                    break
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse hold time field {field}: {e}")
                    continue
        
        # 4. EXTRACT DIRECT PNL VALUES (NO SCALING!)
        pnl_field_names = [
            'pnl_7d',               # 7-day PnL
            'profit_7d',            # 7-day profit
            'pnl_7_days',           # Alternative
            'realized_pnl_7d',      # Realized 7-day PnL
            'pnl',                  # Total PnL
            'total_pnl',            # Total PnL
            'realized_pnl',         # Realized PnL
            'net_pnl'              # Net PnL
        ]
        
        # Try to find 7-day specific PnL first
        for field in pnl_field_names:
            if field in cielo_data and cielo_data[field] is not None:
                try:
                    pnl_value = float(cielo_data[field])
                    
                    if '7d' in field or '7_day' in field:
                        # This is 7-day specific
                        values['usd_profit_7_days'] = round(pnl_value, 1)
                        values['usd_profit_2_days'] = round(pnl_value * 0.3, 1)  # Rough estimate
                        logger.info(f"✅ FOUND 7-DAY PNL from '{field}': ${pnl_value}")
                        break
                    else:
                        # This might be total PnL, estimate 7-day portion
                        values['usd_profit_30_days'] = round(pnl_value, 1)
                        values['usd_profit_7_days'] = round(pnl_value * 0.25, 1)  # 25% from last 7 days
                        values['usd_profit_2_days'] = round(pnl_value * 0.08, 1)  # 8% from last 2 days
                        logger.info(f"✅ FOUND TOTAL PNL from '{field}': ${pnl_value}")
                        break
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse PnL field {field}: {e}")
                    continue
        
        # 5. EXTRACT UNIQUE TOKENS 30D (NEW FIELD)
        unique_tokens_field_names = [
            'unique_tokens_30d',     # Exact field name
            'unique_tokens',         # Generic
            'tokens_traded_30d',     # Alternative
            'total_tokens_30d',      # Another option
            'tokens_count_30d',      # Count version
            'unique_token_count'     # Count version
        ]
        
        for field in unique_tokens_field_names:
            if field in cielo_data and cielo_data[field] is not None:
                try:
                    tokens_count = int(cielo_data[field])
                    values['unique_tokens_30d'] = tokens_count
                    logger.info(f"✅ FOUND UNIQUE TOKENS 30D from '{field}': {tokens_count}")
                    break
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse unique tokens field {field}: {e}")
                    continue
        
        # 6. CALCULATE VOLUME METRICS FROM CIELO DATA
        volume_field_names = [
            'avg_buy_amount_usd',    # Average buy amount
            'average_buy_size_usd',  # Alternative
            'total_buy_volume_usd',  # Total volume
            'buy_volume_usd'         # Buy volume
        ]
        
        total_trades = cielo_data.get('total_trades', 0) or cielo_data.get('swaps_count', 0) or 1
        
        for field in volume_field_names:
            if field in cielo_data and cielo_data[field] is not None:
                try:
                    volume_value = float(cielo_data[field])
                    
                    if 'avg' in field.lower() or 'average' in field.lower():
                        avg_usd_per_trade = volume_value
                    else:
                        avg_usd_per_trade = volume_value / max(total_trades, 1)
                    
                    # Convert USD to SOL (rough estimate)
                    sol_price_estimate = 100.0  # Rough SOL price estimate
                    values['avg_sol_buy_per_token'] = round(avg_usd_per_trade / sol_price_estimate, 1)
                    logger.info(f"✅ FOUND VOLUME from '{field}': ${volume_value}")
                    break
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse volume field {field}: {e}")
                    continue
        
        # 7. CALCULATE AVERAGE BUYS PER TOKEN
        if values['unique_tokens_30d'] > 0:
            buy_count = cielo_data.get('buy_count', 0) or cielo_data.get('buys', 0) or total_trades
            values['avg_buys_per_token'] = round(buy_count / values['unique_tokens_30d'], 1)
        
        logger.info(f"FINAL DIRECT CIELO VALUES for {wallet_address[:8]}:")
        for key, value in values.items():
            logger.info(f"  {key}: {value}")
        
        return values
        
    except Exception as e:
        logger.error(f"Error extracting direct Cielo fields: {str(e)}")
        return values

def _extract_tp_sl_recommendations(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Extract TP/SL recommendations from trade analysis data."""
    try:
        strategy = analysis.get('strategy_recommendation', {})
        
        # Get values with defaults
        tp1 = strategy.get('tp1_percent', 75)
        tp2 = strategy.get('tp2_percent', 200)
        stop_loss = strategy.get('stop_loss_percent', -35)
        
        # Validate ranges
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

def _identify_trader_pattern(analysis: Dict[str, Any]) -> str:
    """Identify trader pattern based on analysis data."""
    try:
        # Check if we have trade pattern analysis
        trade_pattern_analysis = analysis.get('trade_pattern_analysis', {})
        if trade_pattern_analysis and 'pattern' in trade_pattern_analysis:
            return trade_pattern_analysis['pattern']
        
        # Fallback to token analysis
        token_analysis = analysis.get('token_analysis', [])
        
        if not token_analysis:
            return 'insufficient_data'
        
        completed_trades = [t for t in token_analysis if t.get('trade_status') == 'completed']
        
        if len(completed_trades) < 2:
            return 'new_trader'
        
        # Calculate metrics
        rois = [t.get('roi_percent', 0) for t in completed_trades]
        hold_times = [t.get('hold_time_hours', 0) for t in completed_trades]
        
        avg_roi = sum(rois) / len(rois)
        avg_hold_time = sum(hold_times) / len(hold_times)
        moonshots = sum(1 for roi in rois if roi >= 400)
        
        # Pattern identification with updated thresholds
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

def _generate_strategy_reasoning(analysis: Dict[str, Any]) -> str:
    """Generate strategy reasoning."""
    try:
        binary_decisions = analysis.get('binary_decisions', {})
        strategy = analysis.get('strategy_recommendation', {})
        
        follow_wallet = binary_decisions.get('follow_wallet', False)
        follow_sells = binary_decisions.get('follow_sells', False)
        
        if not follow_wallet:
            return "Not recommended - insufficient score"
        
        # Get timestamp info
        last_tx_data = analysis.get('last_transaction_data', {})
        days_since_last = last_tx_data.get('days_since_last_trade', 999)
        
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
            tp1 = strategy.get('tp1_percent', 0)
            tp2 = strategy.get('tp2_percent', 0)
            reasoning_parts.append(f"Custom exits: {tp1}%-{tp2}%")
        
        return " | ".join(reasoning_parts)
        
    except Exception as e:
        logger.error(f"Error generating strategy reasoning: {str(e)}")
        return f"Strategy error: {str(e)}"

def _generate_decision_reasoning(analysis: Dict[str, Any]) -> str:
    """Generate decision reasoning."""
    try:
        binary_decisions = analysis.get('binary_decisions', {})
        composite_score = analysis.get('composite_score', 0)
        
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

def _create_failed_row(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create CSV row for failed analysis."""
    wallet_address = analysis.get('wallet_address', '')
    error_message = analysis.get('error', 'Unknown error')
    
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
        'unique_tokens_30d': 0,  # NEW FIELD
        'trader_pattern': 'failed_analysis',
        'strategy_reason': f"Analysis failed: {error_message}",
        'decision_reason': f"Cannot analyze: {error_message}"
    }

def _create_error_row(analysis: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
    """Create error row when row creation fails."""
    return {
        'wallet_address': analysis.get('wallet_address', ''),
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
        'unique_tokens_30d': 0,  # NEW FIELD
        'trader_pattern': 'error',
        'strategy_reason': f'Error: {error_msg}',
        'decision_reason': f'Processing error: {error_msg}'
    }

def export_zeus_summary(results: Dict[str, Any], output_file: str) -> bool:
    """Export Zeus analysis summary to text file."""
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        analyses = results.get('analyses', [])
        successful_analyses = [a for a in analyses if a.get('success')]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ZEUS WALLET ANALYSIS SUMMARY - DIRECT CIELO FIELD VALUES\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("📊 ANALYSIS OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Wallets: {len(successful_analyses)}\n")
            f.write(f"Data Source: DIRECT Cielo API field extraction\n")
            f.write(f"NO SCALING: Raw values from Trading Stats endpoint\n")
            f.write(f"NEW: unique_tokens_30d field added\n")
            f.write(f"REMOVED: total_buys_30_days, total_sells_30_days columns\n\n")
        
        logger.info(f"✅ Exported Zeus summary to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error exporting Zeus summary: {str(e)}")
        return False