"""
Zeus Export - COMPLETELY FIXED with Correct Cielo Field Mappings
MAJOR FIXES:
- Updated field mappings to use actual Cielo API field names
- ROI calculation from PnL and buy amounts
- Time-based profit estimation using consecutive trading days
- Hold time conversion from seconds to minutes
- Token count extraction from holding_distribution
- USD to SOL conversion for volume metrics
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
    """Export Zeus analysis results to CSV with CORRECT Cielo field extraction."""
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
        
        # Prepare CSV data with CORRECT field extraction
        csv_data = []
        
        for analysis in analyses:
            if not analysis.get('success'):
                csv_data.append(_create_failed_row(analysis))
                continue
            
            # Create analysis row with CORRECT Cielo field values
            csv_data.append(_create_correct_cielo_analysis_row(analysis))
        
        # Sort by composite score (highest first)
        csv_data.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        # Write CSV
        if csv_data:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = _get_updated_csv_fieldnames()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        logger.info(f"✅ Exported {len(csv_data)} wallet analyses with CORRECT CIELO FIELD VALUES to: {output_file}")
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

def _create_correct_cielo_analysis_row(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create CSV row with CORRECT Cielo field extraction using actual field names."""
    try:
        # Extract basic data
        wallet_address = analysis.get('wallet_address', '')
        binary_decisions = analysis.get('binary_decisions', {})
        strategy = analysis.get('strategy_recommendation', {})
        wallet_data = analysis.get('wallet_data', {})
        
        logger.info(f"📊 CORRECT CIELO FIELD EXTRACTION: {wallet_address[:8]}...")
        
        # Get timestamp data (1 decimal precision)
        real_days_since_last = _extract_days_since_last_trade(analysis)
        
        # Extract CORRECT Cielo field values using actual field names
        cielo_values = _extract_correct_cielo_fields(wallet_address, wallet_data)
        
        # Get real TP/SL recommendations from trade analysis
        tp_sl_values = _extract_tp_sl_recommendations(analysis)
        
        # Create the row with CORRECT field values
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
            'unique_tokens_30d': cielo_values['unique_tokens_30d'],
            'trader_pattern': _identify_trader_pattern(analysis),
            'strategy_reason': _generate_strategy_reasoning(analysis),
            'decision_reason': _generate_decision_reasoning(analysis)
        }
        
        logger.info(f"  CORRECT CIELO VALUES for {wallet_address[:8]}:")
        logger.info(f"    roi_7_day: {cielo_values['roi_7_day']}%")
        logger.info(f"    winrate_7_day: {cielo_values['winrate_7_day']}%")
        logger.info(f"    unique_tokens_30d: {cielo_values['unique_tokens_30d']}")
        logger.info(f"    usd_profit_7_days: ${cielo_values['usd_profit_7_days']}")
        logger.info(f"    avg_hold_time_minutes: {cielo_values['avg_hold_time_minutes']}")
        
        return row
        
    except Exception as e:
        logger.error(f"Error creating correct Cielo analysis row: {str(e)}")
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

def _extract_correct_cielo_fields(wallet_address: str, wallet_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract CORRECT field values from Cielo Trading Stats API response using ACTUAL field names.
    Based on debug results showing the real Cielo API structure.
    """
    try:
        logger.info(f"📊 CORRECT CIELO FIELD EXTRACTION for {wallet_address[:8]}...")
        
        # Initialize with defaults
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
        
        # 1. EXTRACT WINRATE - Field name: 'winrate' (CONFIRMED from debug)
        if 'winrate' in cielo_data and cielo_data['winrate'] is not None:
            try:
                winrate_value = float(cielo_data['winrate'])
                values['winrate_7_day'] = round(winrate_value, 2)
                logger.info(f"✅ FOUND WINRATE: {winrate_value}%")
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to parse winrate: {e}")
        
        # 2. CALCULATE ROI from PnL and total_buy_amount_usd (CONFIRMED from debug)
        if 'pnl' in cielo_data and 'total_buy_amount_usd' in cielo_data:
            try:
                pnl = float(cielo_data['pnl'])
                total_buy = float(cielo_data['total_buy_amount_usd'])
                if total_buy > 0:
                    roi_percent = (pnl / total_buy) * 100
                    values['roi_7_day'] = round(roi_percent, 2)
                    logger.info(f"✅ CALCULATED ROI: {pnl} / {total_buy} * 100 = {roi_percent:.2f}%")
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to calculate ROI: {e}")
        
        # 3. EXTRACT UNIQUE TOKENS from holding_distribution.total_tokens (CONFIRMED from debug)
        if 'holding_distribution' in cielo_data:
            try:
                hold_dist = cielo_data['holding_distribution']
                if isinstance(hold_dist, dict) and 'total_tokens' in hold_dist:
                    tokens_count = int(hold_dist['total_tokens'])
                    values['unique_tokens_30d'] = tokens_count
                    logger.info(f"✅ FOUND UNIQUE TOKENS: {tokens_count}")
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to parse unique tokens: {e}")
        
        # 4. CONVERT HOLD TIME from average_holding_time_sec to minutes (CONFIRMED from debug)
        if 'average_holding_time_sec' in cielo_data:
            try:
                seconds = float(cielo_data['average_holding_time_sec'])
                minutes = seconds / 60.0
                values['avg_hold_time_minutes'] = round(minutes, 1)
                logger.info(f"✅ CONVERTED HOLD TIME: {seconds}s = {minutes:.1f}min")
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to convert hold time: {e}")
        
        # 5. CALCULATE TIME-BASED PROFITS using consecutive_trading_days (CONFIRMED from debug)
        if 'pnl' in cielo_data and 'consecutive_trading_days' in cielo_data:
            try:
                pnl = float(cielo_data['pnl'])
                days = max(1, int(cielo_data['consecutive_trading_days']))
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
        
        # 6. CONVERT VOLUME METRICS (USD to SOL) using average_buy_amount_usd (CONFIRMED from debug)
        if 'average_buy_amount_usd' in cielo_data:
            try:
                avg_usd = float(cielo_data['average_buy_amount_usd'])
                sol_price_estimate = 100.0  # Rough SOL price estimate
                values['avg_sol_buy_per_token'] = round(avg_usd / sol_price_estimate, 1)
                logger.info(f"✅ CONVERTED VOLUME: ${avg_usd} / ${sol_price_estimate} = {values['avg_sol_buy_per_token']} SOL")
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to convert volume: {e}")
        
        # 7. CALCULATE TRADE FREQUENCY using buy_count and unique_tokens (CONFIRMED from debug)
        if 'buy_count' in cielo_data and values['unique_tokens_30d'] > 0:
            try:
                buys = int(cielo_data['buy_count'])
                tokens = values['unique_tokens_30d']
                values['avg_buys_per_token'] = round(buys / tokens, 1)
                logger.info(f"✅ CALCULATED TRADE FREQUENCY: {buys} buys / {tokens} tokens = {values['avg_buys_per_token']}")
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to calculate trade frequency: {e}")
        
        logger.info(f"FINAL CORRECT CIELO VALUES for {wallet_address[:8]}:")
        for key, value in values.items():
            logger.info(f"  {key}: {value}")
        
        return values
        
    except Exception as e:
        logger.error(f"Error extracting correct Cielo fields: {str(e)}")
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
        'unique_tokens_30d': 0,
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
        'unique_tokens_30d': 0,
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
            f.write("ZEUS WALLET ANALYSIS SUMMARY - CORRECT CIELO FIELD VALUES\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("📊 ANALYSIS OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Wallets: {len(successful_analyses)}\n")
            f.write(f"Data Source: CORRECT Cielo API field extraction\n")
            f.write(f"Field Mappings: Updated to match actual Cielo response\n")
            f.write(f"ROI Calculation: PnL / total_buy_amount_usd * 100\n")
            f.write(f"Time Profits: Based on consecutive_trading_days\n")
            f.write(f"Hold Time: Converted from seconds to minutes\n")
            f.write(f"Token Count: Extracted from holding_distribution.total_tokens\n\n")
        
        logger.info(f"✅ Exported Zeus summary to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error exporting Zeus summary: {str(e)}")
        return False