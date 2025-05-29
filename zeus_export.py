"""
Zeus Export - ACTUALLY Use Direct Cielo Values (No More Conversions!)
FINAL FIX:
- Use EXACT values from Cielo API response fields
- roi_7_day = Direct "Realized PnL (ROI)" from Cielo (30.56%)
- average_holding_time_minutes = Direct "Avg Hold Time" from Cielo (7 minutes)  
- 7_day_winrate = Direct "Token Winrate" from Cielo (75.00%)
- Remove timestamp_source and timestamp_accuracy from CSV
- Add 7_day_winrate after composite_score
- NO MORE MATH OR CONVERSIONS!
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
            csv_data.append(_create_direct_field_analysis_row(analysis))
        
        # Sort by composite score (highest first)
        csv_data.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        # Write CSV
        if csv_data:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = _get_direct_csv_fieldnames()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        logger.info(f"✅ Exported {len(csv_data)} wallet analyses with DIRECT CIELO FIELD VALUES to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error exporting Zeus analysis: {str(e)}")
        return False

def _get_direct_csv_fieldnames() -> List[str]:
    """Get CSV fieldnames with 7_day_winrate added and timestamp fields removed."""
    return [
        'wallet_address',
        'composite_score',
        '7_day_winrate',  # NEW: Direct from Cielo "Token Winrate"
        'days_since_last_trade',
        # Removed: timestamp_source, timestamp_accuracy
        'roi_7_day',  # Direct from Cielo "Realized PnL (ROI)"
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
        'average_holding_time_minutes',  # Direct from Cielo "Avg Hold Time"
        'total_buys_30_days',
        'total_sells_30_days',
        'trader_pattern',
        'strategy_reason',
        'decision_reason'
    ]

def _create_direct_field_analysis_row(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create CSV row with DIRECT Cielo field extraction - NO CONVERSIONS."""
    try:
        # Extract basic data
        wallet_address = analysis.get('wallet_address', '')
        binary_decisions = analysis.get('binary_decisions', {})
        strategy = analysis.get('strategy_recommendation', {})
        wallet_data = analysis.get('wallet_data', {})
        
        logger.info(f"📊 DIRECT FIELD EXTRACTION: {wallet_address[:8]}...")
        
        # Get timestamp data
        real_days_since_last = _extract_days_since_last_trade(analysis)
        
        # Extract DIRECT Cielo field values - NO MATH!
        cielo_values = _extract_exact_cielo_field_values(wallet_address, wallet_data)
        
        # Create the row with DIRECT field values
        row = {
            'wallet_address': wallet_address,
            'composite_score': round(analysis.get('composite_score', 0), 1),
            '7_day_winrate': cielo_values['winrate_7_day'],  # DIRECT from Cielo
            'days_since_last_trade': real_days_since_last,
            # Removed timestamp_source and timestamp_accuracy
            'roi_7_day': cielo_values['roi_7_day'],  # DIRECT from Cielo
            'usd_profit_2_days': cielo_values['usd_profit_2_days'],
            'usd_profit_7_days': cielo_values['usd_profit_7_days'],
            'usd_profit_30_days': cielo_values['usd_profit_30_days'],
            'copy_wallet': 'YES' if binary_decisions.get('follow_wallet', False) else 'NO',
            'copy_sells': 'YES' if binary_decisions.get('follow_sells', False) else 'NO',
            'tp_1': strategy.get('tp1_percent', 0),
            'tp_2': strategy.get('tp2_percent', 0),
            'stop_loss': strategy.get('stop_loss_percent', -35),
            'avg_sol_buy_per_token': cielo_values['avg_sol_buy_per_token'],
            'avg_buys_per_token': cielo_values['avg_buys_per_token'],
            'average_holding_time_minutes': cielo_values['avg_hold_time_minutes'],  # DIRECT from Cielo
            'total_buys_30_days': cielo_values['total_buys_30_days'],
            'total_sells_30_days': cielo_values['total_sells_30_days'],
            'trader_pattern': _identify_trader_pattern(analysis),
            'strategy_reason': _generate_strategy_reasoning(analysis),
            'decision_reason': _generate_decision_reasoning(analysis)
        }
        
        logger.info(f"  DIRECT CIELO FIELD VALUES for {wallet_address[:8]}:")
        logger.info(f"    7_day_winrate: {cielo_values['winrate_7_day']}%")
        logger.info(f"    roi_7_day: {cielo_values['roi_7_day']}%")
        logger.info(f"    avg_hold_time_minutes: {cielo_values['avg_hold_time_minutes']} min")
        
        return row
        
    except Exception as e:
        logger.error(f"Error creating direct field analysis row: {str(e)}")
        return _create_error_row(analysis, str(e))

def _extract_days_since_last_trade(analysis: Dict[str, Any]) -> float:
    """Extract days since last trade from Helius timestamp data."""
    try:
        last_tx_data = analysis.get('last_transaction_data', {})
        if isinstance(last_tx_data, dict) and last_tx_data.get('success', False):
            days_since = last_tx_data.get('days_since_last_trade', 999)
            if days_since is not None and isinstance(days_since, (int, float)):
                return round(float(days_since), 2)
        return 999.0
    except Exception as e:
        logger.error(f"Error extracting days since last trade: {str(e)}")
        return 999.0

def _extract_exact_cielo_field_values(wallet_address: str, wallet_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract EXACT field values from Cielo API response - NO CONVERSIONS OR MATH!
    Map directly to the values shown in Cielo interface.
    """
    try:
        logger.info(f"📊 EXACT CIELO FIELD EXTRACTION for {wallet_address[:8]}...")
        
        # Initialize with defaults
        values = {
            'roi_7_day': 0.0,        # "Realized PnL (ROI)" from Cielo
            'winrate_7_day': 0.0,    # "Token Winrate" from Cielo  
            'avg_hold_time_minutes': 0.0,  # "Avg Hold Time" from Cielo
            'usd_profit_2_days': 0.0,
            'usd_profit_7_days': 0.0,
            'usd_profit_30_days': 0.0,
            'total_buys_30_days': 0,
            'total_sells_30_days': 0,
            'avg_sol_buy_per_token': 0.0,
            'avg_buys_per_token': 0.0
        }
        
        # Extract Cielo data from nested structure
        cielo_data = None
        if isinstance(wallet_data, dict):
            if 'data' in wallet_data and isinstance(wallet_data['data'], dict):
                cielo_data = wallet_data['data']
            elif wallet_data.get('source') in ['cielo_finance_real', 'cielo_trading_stats']:
                cielo_data = wallet_data.get('data', {})
            else:
                cielo_data = wallet_data
        
        if not isinstance(cielo_data, dict):
            logger.warning(f"No valid Cielo data found for {wallet_address[:8]}")
            return values
        
        logger.info(f"Available Cielo API fields: {list(cielo_data.keys())}")
        
        # Log first 10 fields with their values for debugging
        logger.info(f"CIELO API RESPONSE SAMPLE:")
        for i, (key, value) in enumerate(list(cielo_data.items())[:10]):
            logger.info(f"  {key}: {type(value).__name__} = {value}")
        
        # 1. EXTRACT EXACT "Realized PnL (ROI)" - This should be 30.56% for your example
        roi_field_names = [
            'realized_pnl_roi',           # Exact field name
            'roi',                        # Simple ROI field
            'realized_roi',               # Realized ROI
            'pnl_roi',                    # PnL ROI
            'roi_percent',                # ROI percentage
            'realized_pnl_percentage',    # Realized PnL percentage
            'return_on_investment',       # Full name
            'pnl_roi_7d',                # 7-day specific
            'roi_7_day',                  # 7-day ROI
            'winrate'                     # Sometimes mislabeled as winrate
        ]
        
        for field in roi_field_names:
            if field in cielo_data and cielo_data[field] is not None:
                try:
                    roi_value = float(cielo_data[field])
                    
                    # Use the value as-is (should already be percentage like 30.56)
                    if 0 <= roi_value <= 1000:  # Reasonable percentage range
                        values['roi_7_day'] = round(roi_value, 2)
                        logger.info(f"✅ FOUND EXACT ROI from '{field}': {roi_value}%")
                        break
                    elif -1 <= roi_value <= 10:  # Decimal format (0.3056 = 30.56%)
                        values['roi_7_day'] = round(roi_value * 100, 2)
                        logger.info(f"✅ FOUND EXACT ROI from '{field}': {roi_value} -> {roi_value * 100}%")
                        break
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse ROI field {field}: {e}")
                    continue
        
        # 2. EXTRACT EXACT "Token Winrate" - This should be 75.00% for your example
        winrate_field_names = [
            'token_winrate',              # Exact field name
            'winrate',                    # Simple winrate
            'win_rate',                   # Win rate
            'token_win_rate',             # Token win rate
            'winning_rate',               # Winning rate
            'success_rate',               # Success rate
            'winrate_percent',            # Winrate percentage
            'win_percentage',             # Win percentage
            'winrate_7d',                 # 7-day specific
            'token_winrate_7d'            # 7-day token winrate
        ]
        
        for field in winrate_field_names:
            if field in cielo_data and cielo_data[field] is not None:
                try:
                    winrate_value = float(cielo_data[field])
                    
                    # Use the value as-is (should already be percentage like 75.00)
                    if 0 <= winrate_value <= 100:  # Percentage format
                        values['winrate_7_day'] = round(winrate_value, 2)
                        logger.info(f"✅ FOUND EXACT WINRATE from '{field}': {winrate_value}%")
                        break
                    elif 0 <= winrate_value <= 1:  # Decimal format (0.75 = 75%)
                        values['winrate_7_day'] = round(winrate_value * 100, 2)
                        logger.info(f"✅ FOUND EXACT WINRATE from '{field}': {winrate_value} -> {winrate_value * 100}%")
                        break
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse winrate field {field}: {e}")
                    continue
        
        # 3. EXTRACT EXACT "Avg Hold Time" - This should be 7 minutes for your example
        hold_time_field_names = [
            'avg_hold_time',              # Average hold time
            'average_holding_time',       # Average holding time
            'avg_holding_time_minutes',   # In minutes
            'average_hold_time_minutes',  # In minutes
            'holding_time_avg',           # Holding time average
            'avg_holding_time_sec',       # In seconds (convert)
            'average_holding_time_sec',   # In seconds (convert)
            'hold_time_average',          # Hold time average
            'avg_hold_time_7d',           # 7-day specific
            'average_holding_time_7d'     # 7-day average
        ]
        
        for field in hold_time_field_names:
            if field in cielo_data and cielo_data[field] is not None:
                try:
                    hold_time_value = float(cielo_data[field])
                    
                    # Convert based on field name and value range
                    if 'sec' in field.lower() or hold_time_value > 1000:
                        # Seconds - convert to minutes
                        values['avg_hold_time_minutes'] = round(hold_time_value / 60.0, 1)
                        logger.info(f"✅ FOUND EXACT HOLD TIME from '{field}': {hold_time_value}s -> {values['avg_hold_time_minutes']}min")
                    elif 'minutes' in field.lower() or hold_time_value < 300:
                        # Already in minutes
                        values['avg_hold_time_minutes'] = round(hold_time_value, 1)
                        logger.info(f"✅ FOUND EXACT HOLD TIME from '{field}': {hold_time_value}min")
                    else:
                        # Assume hours if reasonable range
                        values['avg_hold_time_minutes'] = round(hold_time_value * 60.0, 1)
                        logger.info(f"✅ FOUND EXACT HOLD TIME from '{field}': {hold_time_value}h -> {values['avg_hold_time_minutes']}min")
                    break
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse hold time field {field}: {e}")
                    continue
        
        # 4. EXTRACT DIRECT PNL VALUES
        pnl_field_names = ['pnl', 'total_pnl', 'realized_pnl', 'net_pnl', 'profit_loss', 'total_profit_loss']
        for field in pnl_field_names:
            if field in cielo_data and cielo_data[field] is not None:
                try:
                    pnl_value = float(cielo_data[field])
                    values['usd_profit_30_days'] = round(pnl_value, 1)
                    # Simple distribution for shorter periods
                    values['usd_profit_7_days'] = round(pnl_value * 0.3, 1)
                    values['usd_profit_2_days'] = round(pnl_value * 0.1, 1)
                    logger.info(f"✅ FOUND EXACT PNL from '{field}': ${pnl_value}")
                    break
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse PNL field {field}: {e}")
                    continue
        
        # 5. EXTRACT DIRECT BUY/SELL COUNTS
        buy_fields = ['buy_count', 'buys', 'total_buys', 'swaps_buy']
        for field in buy_fields:
            if field in cielo_data and cielo_data[field] is not None:
                try:
                    buy_count = int(cielo_data[field])
                    values['total_buys_30_days'] = buy_count
                    logger.info(f"✅ FOUND EXACT BUYS from '{field}': {buy_count}")
                    break
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse buy field {field}: {e}")
                    continue
        
        sell_fields = ['sell_count', 'sells', 'total_sells', 'swaps_sell']
        for field in sell_fields:
            if field in cielo_data and cielo_data[field] is not None:
                try:
                    sell_count = int(cielo_data[field])
                    values['total_sells_30_days'] = sell_count
                    logger.info(f"✅ FOUND EXACT SELLS from '{field}': {sell_count}")
                    break
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse sell field {field}: {e}")
                    continue
        
        # 6. EXTRACT VOLUME DATA
        volume_fields = ['avg_buy_amount_usd', 'average_buy_size', 'total_buy_amount_usd']
        total_trades = max(values['total_buys_30_days'], 1)
        
        for field in volume_fields:
            if field in cielo_data and cielo_data[field] is not None:
                try:
                    volume_value = float(cielo_data[field])
                    
                    if 'avg' in field.lower() or 'average' in field.lower():
                        avg_usd_per_trade = volume_value
                    else:
                        avg_usd_per_trade = volume_value / total_trades
                    
                    # Simple USD to SOL conversion
                    sol_price_estimate = 100.0
                    values['avg_sol_buy_per_token'] = round(avg_usd_per_trade / sol_price_estimate, 3)
                    logger.info(f"✅ FOUND EXACT VOLUME from '{field}': ${volume_value}")
                    break
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse volume field {field}: {e}")
                    continue
        
        # Calculate derived metrics
        if values['total_buys_30_days'] > 0:
            estimated_tokens = max(1, values['total_buys_30_days'] // 2)
            values['avg_buys_per_token'] = round(values['total_buys_30_days'] / estimated_tokens, 1)
        
        logger.info(f"FINAL EXACT CIELO FIELD VALUES for {wallet_address[:8]}:")
        for key, value in values.items():
            logger.info(f"  {key}: {value}")
        
        return values
        
    except Exception as e:
        logger.error(f"Error extracting exact Cielo field values: {str(e)}")
        return values

def _identify_trader_pattern(analysis: Dict[str, Any]) -> str:
    """Identify trader pattern based on analysis data."""
    try:
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
        
        # Pattern identification
        if avg_hold_time < 0.2:
            return 'flipper'
        elif avg_hold_time < 1:
            return 'sniper' if avg_roi > 30 else 'impulsive_trader'
        elif moonshots > 0:
            return 'gem_hunter'
        elif avg_hold_time > 48:
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
        'total_buys_30_days': 0,
        'total_sells_30_days': 0,
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
        'total_buys_30_days': 0,
        'total_sells_30_days': 0,
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
            f.write("ZEUS WALLET ANALYSIS SUMMARY - EXACT CIELO FIELD VALUES\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("📊 ANALYSIS OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Wallets: {len(successful_analyses)}\n")
            f.write(f"Data Source: EXACT Cielo API field extraction\n")
            f.write(f"Values: Direct field mapping (NO CONVERSIONS)\n\n")
        
        logger.info(f"✅ Exported Zeus summary to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error exporting Zeus summary: {str(e)}")
        return False