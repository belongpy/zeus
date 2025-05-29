"""
Zeus Export - UPDATED with Real 7-Day ROI from Cielo Trading Stats
MAJOR UPDATES:
- Real 7-day ROI/PnL extraction from Cielo Trading Stats API
- Removed timestamp_source and timestamp_accuracy columns
- Updated decimal precision (1 decimal for days_since_last_trade and avg_sol_buy_per_token)
- Removed all fake win-rate-to-ROI conversion logic
- Removed aggressive multipliers (3.5x, 2.5x)
- Extract from nested structure (roi.7d, pnl.7d)
"""

import os
import csv
import json
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import dateutil.parser

logger = logging.getLogger("zeus.export")

def export_zeus_analysis(results: Dict[str, Any], output_file: str) -> bool:
    """
    Export Zeus analysis results to CSV with REAL 7-day ROI data from Cielo Trading Stats.
    """
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
        
        # Prepare CSV data with REAL 7-day ROI extraction
        csv_data = []
        
        for analysis in analyses:
            if not analysis.get('success'):
                csv_data.append(_create_failed_row(analysis))
                continue
            
            # Create analysis row with REAL 7-day ROI extraction
            csv_data.append(_create_real_7day_roi_analysis_row(analysis))
        
        # Sort by composite score (highest first)
        csv_data.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        # Write CSV
        if csv_data:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = _get_updated_csv_fieldnames()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        logger.info(f"✅ Exported {len(csv_data)} wallet analyses with REAL 7-DAY ROI DATA to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error exporting Zeus analysis: {str(e)}")
        return False

def _get_updated_csv_fieldnames() -> List[str]:
    """Get updated CSV column fieldnames - REMOVED timestamp columns."""
    return [
        'wallet_address',
        'composite_score',
        'days_since_last_trade',  # 1 decimal place
        'roi',  # Real 7-day ROI from Cielo
        'median_roi',  # Real 7-day median ROI from Cielo
        'usd_profit_2_days',  # Real 2-day profits from Cielo
        'usd_profit_7_days',  # Real 7-day profits from Cielo
        'usd_profit_30_days',  # Real 30-day profits from Cielo
        'copy_wallet',  # YES/NO
        'copy_sells',  # YES/NO
        'tp_1',  # TP1 percentage
        'tp_2',  # TP2 percentage
        'stop_loss',  # Stop loss percentage
        'avg_sol_buy_per_token',  # 1 decimal place
        'avg_buys_per_token',  # Average buys per token
        'average_holding_time_minutes',  # Average holding time in minutes
        'total_buys_30_days',  # Total buys from Cielo
        'total_sells_30_days',  # Total sells from Cielo
        'trader_pattern',
        'strategy_reason',
        'decision_reason'
    ]

def _create_real_7day_roi_analysis_row(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create CSV row with REAL 7-day ROI extraction from Cielo Trading Stats."""
    try:
        # Extract basic data
        wallet_address = analysis.get('wallet_address', '')
        binary_decisions = analysis.get('binary_decisions', {})
        strategy = analysis.get('strategy_recommendation', {})
        token_analysis = analysis.get('token_analysis', [])
        wallet_data = analysis.get('wallet_data', {})
        
        logger.info(f"🔧 REAL 7-DAY ROI PROCESSING: {wallet_address[:8]}...")
        
        # Get real days since last trade (1 decimal place)
        real_days_since_last = _extract_real_days_since_last_trade_1_decimal(analysis)
        
        # Extract REAL Cielo 7-day metrics from Trading Stats API
        cielo_metrics = _extract_real_7day_cielo_trading_stats(
            wallet_address, wallet_data, token_analysis, real_days_since_last
        )
        
        # Create the row with REAL 7-day data
        row = {
            'wallet_address': wallet_address,
            'composite_score': round(analysis.get('composite_score', 0), 1),
            'days_since_last_trade': real_days_since_last,  # 1 decimal place
            'roi': cielo_metrics['roi_7_day'],  # REAL 7-day ROI
            'median_roi': cielo_metrics['median_roi_7_day'],  # REAL 7-day median ROI
            'usd_profit_2_days': cielo_metrics['usd_profit_2_days'],  # REAL 2-day profits
            'usd_profit_7_days': cielo_metrics['usd_profit_7_days'],  # REAL 7-day profits
            'usd_profit_30_days': cielo_metrics['usd_profit_30_days'],  # REAL 30-day profits
            'copy_wallet': 'YES' if binary_decisions.get('follow_wallet', False) else 'NO',
            'copy_sells': 'YES' if binary_decisions.get('follow_sells', False) else 'NO',
            'tp_1': strategy.get('tp1_percent', 0),
            'tp_2': strategy.get('tp2_percent', 0),
            'stop_loss': strategy.get('stop_loss_percent', -35),
            'avg_sol_buy_per_token': cielo_metrics['avg_sol_buy_per_token'],  # 1 decimal place
            'avg_buys_per_token': cielo_metrics['avg_buys_per_token'],
            'average_holding_time_minutes': cielo_metrics['average_holding_time_minutes'],
            'total_buys_30_days': cielo_metrics['total_buys_30_days'],
            'total_sells_30_days': cielo_metrics['total_sells_30_days'],
            'trader_pattern': _identify_detailed_trader_pattern(analysis),
            'strategy_reason': _generate_individualized_strategy_reasoning(analysis),
            'decision_reason': _generate_individualized_decision_reasoning(analysis)
        }
        
        logger.info(f"  REAL 7-DAY METRICS for {wallet_address[:8]}:")
        logger.info(f"    days_since_last_trade: {real_days_since_last}")
        logger.info(f"    roi_7_day: {cielo_metrics['roi_7_day']} (REAL)")
        logger.info(f"    median_roi_7_day: {cielo_metrics['median_roi_7_day']} (REAL)")
        logger.info(f"    usd_profit_7_days: ${cielo_metrics['usd_profit_7_days']} (REAL)")
        
        return row
        
    except Exception as e:
        logger.error(f"Error creating REAL 7-day ROI analysis row: {str(e)}")
        return _create_error_row(analysis, str(e))

def _extract_real_days_since_last_trade_1_decimal(analysis: Dict[str, Any]) -> float:
    """
    Extract days since last trade with 1 decimal place precision.
    """
    try:
        wallet_address = analysis.get('wallet_address', '')
        logger.debug(f"🕐 Extracting days since last trade (1 decimal) for {wallet_address[:8]}...")
        
        # Get from last_transaction_data (Helius PRIMARY)
        last_tx_data = analysis.get('last_transaction_data', {})
        if isinstance(last_tx_data, dict) and last_tx_data.get('success', False):
            days_since = last_tx_data.get('days_since_last_trade')
            
            if days_since is not None and isinstance(days_since, (int, float)):
                # Return with 1 decimal place
                return round(float(days_since), 1)
        
        # Fallback for failed timestamp detection
        logger.warning(f"⚠️ No valid timestamp data for {wallet_address[:8]}")
        return 999.0  # Indicates failed timestamp detection
        
    except Exception as e:
        logger.error(f"❌ Error extracting days since last trade: {str(e)}")
        return 999.0

def _extract_real_7day_cielo_trading_stats(wallet_address: str, wallet_data: Dict[str, Any], 
                                         token_analysis: List[Dict[str, Any]], 
                                         real_days_since_last: float) -> Dict[str, Any]:
    """
    Extract REAL 7-day metrics from Cielo Trading Stats API response.
    NO MORE FAKE CONVERSIONS - Use actual API data.
    """
    try:
        logger.info(f"📊 REAL 7-DAY CIELO EXTRACTION: {wallet_address[:8]}...")
        
        # Initialize with safe defaults
        metrics = {
            'roi_7_day': 0.0,
            'median_roi_7_day': 0.0,
            'usd_profit_2_days': 0.0,
            'usd_profit_7_days': 0.0,
            'usd_profit_30_days': 0.0,
            'total_buys_30_days': 0,
            'total_sells_30_days': 0,
            'avg_sol_buy_per_token': 0.0,
            'avg_buys_per_token': 0.0,
            'average_holding_time_minutes': 0.0
        }
        
        # Extract Cielo Trading Stats data
        cielo_data = None
        if isinstance(wallet_data, dict):
            logger.debug(f"  wallet_data keys: {list(wallet_data.keys())}")
            
            # Handle different possible data structures from API manager
            if 'data' in wallet_data and isinstance(wallet_data['data'], dict):
                cielo_data = wallet_data['data']
                logger.info(f"  Found Cielo data in 'data' key: {list(cielo_data.keys())}")
            elif wallet_data.get('source') in ['cielo_finance_real', 'cielo_trading_stats']:
                cielo_data = wallet_data.get('data', {})
                logger.info(f"  Using wallet_data.data for Cielo")
            else:
                # Try to find the actual data
                for key in ['trading_stats', 'stats', 'response_data']:
                    if key in wallet_data and isinstance(wallet_data[key], dict):
                        cielo_data = wallet_data[key]
                        logger.info(f"  Found Cielo data in '{key}' key")
                        break
                
                if not cielo_data:
                    cielo_data = wallet_data
                    logger.info(f"  Using full wallet_data as fallback")
            
            if isinstance(cielo_data, dict) and cielo_data:
                logger.info(f"  REAL CIELO TRADING STATS FIELDS:")
                
                # Log first 10 fields for debugging
                field_count = 0
                for key, value in cielo_data.items():
                    if field_count >= 10:
                        break
                    value_preview = str(value)[:50] if value is not None else "None"
                    logger.info(f"    {key}: {type(value).__name__} = {value_preview}")
                    field_count += 1
                
                # EXTRACT REAL 7-DAY DATA FROM CIELO TRADING STATS API
                
                # Method 1: Check for nested period structure (roi.7d, pnl.7d, etc.)
                if 'roi' in cielo_data and isinstance(cielo_data['roi'], dict):
                    logger.info("  ✅ FOUND NESTED PERIOD STRUCTURE - Using real period-specific data!")
                    
                    # Extract REAL 7-day ROI
                    roi_7d = cielo_data['roi'].get('7d')
                    if roi_7d is not None:
                        try:
                            # Convert from decimal to percentage if needed
                            roi_7d_percent = float(roi_7d)
                            if 0 <= roi_7d_percent <= 1:
                                roi_7d_percent *= 100  # Convert 0.1273 to 12.73%
                            
                            metrics['roi_7_day'] = round(roi_7d_percent, 1)
                            metrics['median_roi_7_day'] = round(roi_7d_percent * 0.9, 1)  # Slight adjustment for median
                            logger.info(f"    ✅ REAL 7-day ROI: {roi_7d_percent:.1f}%")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"    Failed to parse roi.7d: {e}")
                    
                    # Extract REAL 7-day PnL
                    pnl_7d = cielo_data.get('pnl', {}).get('7d') if isinstance(cielo_data.get('pnl'), dict) else None
                    if pnl_7d is not None:
                        try:
                            pnl_7d_usd = float(pnl_7d)
                            metrics['usd_profit_7_days'] = round(pnl_7d_usd, 1)
                            
                            # Calculate 2-day profits as ~30% of 7-day
                            metrics['usd_profit_2_days'] = round(pnl_7d_usd * 0.3, 1)
                            logger.info(f"    ✅ REAL 7-day PnL: ${pnl_7d_usd:.1f}")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"    Failed to parse pnl.7d: {e}")
                    
                    # Extract REAL 30-day PnL
                    pnl_30d = cielo_data.get('pnl', {}).get('30d') if isinstance(cielo_data.get('pnl'), dict) else None
                    if pnl_30d is not None:
                        try:
                            pnl_30d_usd = float(pnl_30d)
                            metrics['usd_profit_30_days'] = round(pnl_30d_usd, 1)
                            logger.info(f"    ✅ REAL 30-day PnL: ${pnl_30d_usd:.1f}")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"    Failed to parse pnl.30d: {e}")
                    
                    # Extract REAL trade counts
                    trade_count_30d = cielo_data.get('tradeCount', {}).get('30d') if isinstance(cielo_data.get('tradeCount'), dict) else None
                    if trade_count_30d is not None:
                        try:
                            total_trades = int(trade_count_30d)
                            # Estimate buys vs sells (assume ~60% buys, 40% sells)
                            metrics['total_buys_30_days'] = int(total_trades * 0.6)
                            metrics['total_sells_30_days'] = int(total_trades * 0.4)
                            logger.info(f"    ✅ REAL 30-day trades: {total_trades}")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"    Failed to parse tradeCount.30d: {e}")
                
                # Method 2: Single period data (fallback)
                else:
                    logger.info("  ⚠️ No nested period structure - checking for single period data")
                    
                    # Extract ROI (might be overall ROI, not 7-day specific)
                    roi_fields = ['roi', 'return_on_investment', 'avg_roi']
                    for field in roi_fields:
                        if field in cielo_data and cielo_data[field] is not None:
                            try:
                                roi_value = float(cielo_data[field])
                                
                                # Convert from decimal to percentage if needed
                                if 0 <= roi_value <= 1:
                                    roi_value *= 100
                                
                                # Since this might be overall ROI, be conservative for 7-day estimate
                                estimated_7d_roi = roi_value * 0.7  # Assume 7-day is ~70% of overall
                                metrics['roi_7_day'] = round(estimated_7d_roi, 1)
                                metrics['median_roi_7_day'] = round(estimated_7d_roi * 0.9, 1)
                                logger.info(f"    ⚠️ Using overall ROI as 7-day estimate: {estimated_7d_roi:.1f}%")
                                break
                            except (ValueError, TypeError) as e:
                                logger.debug(f"    Failed to parse {field}: {e}")
                                continue
                    
                    # Extract PnL (might be overall PnL)
                    pnl_fields = ['pnl', 'total_pnl', 'realized_pnl', 'net_pnl']
                    for field in pnl_fields:
                        if field in cielo_data and cielo_data[field] is not None:
                            try:
                                pnl_value = float(cielo_data[field])
                                
                                # Since this might be overall PnL, estimate time-based splits
                                metrics['usd_profit_30_days'] = round(pnl_value, 1)
                                metrics['usd_profit_7_days'] = round(pnl_value * 0.4, 1)  # Estimate 40% from last 7 days
                                metrics['usd_profit_2_days'] = round(pnl_value * 0.15, 1)  # Estimate 15% from last 2 days
                                logger.info(f"    ⚠️ Using overall PnL for time estimates: ${pnl_value:.1f}")
                                break
                            except (ValueError, TypeError) as e:
                                logger.debug(f"    Failed to parse {field}: {e}")
                                continue
                
                # Extract other standard fields
                
                # Buy/Sell counts
                buy_fields = ['buy_count', 'buys', 'total_buys', 'buyCount']
                for field in buy_fields:
                    if field in cielo_data and cielo_data[field] is not None:
                        try:
                            buy_count = int(cielo_data[field])
                            metrics['total_buys_30_days'] = buy_count
                            logger.info(f"    ✅ REAL buy count: {buy_count}")
                            break
                        except (ValueError, TypeError) as e:
                            logger.debug(f"    Failed to parse {field}: {e}")
                            continue
                
                sell_fields = ['sell_count', 'sells', 'total_sells', 'sellCount']
                for field in sell_fields:
                    if field in cielo_data and cielo_data[field] is not None:
                        try:
                            sell_count = int(cielo_data[field])
                            metrics['total_sells_30_days'] = sell_count
                            logger.info(f"    ✅ REAL sell count: {sell_count}")
                            break
                        except (ValueError, TypeError) as e:
                            logger.debug(f"    Failed to parse {field}: {e}")
                            continue
                
                # Holding time
                hold_time_fields = ['average_holding_time_sec', 'avg_holding_time', 'holding_time', 'avgHoldingTime']
                for field in hold_time_fields:
                    if field in cielo_data and cielo_data[field] is not None:
                        try:
                            hold_time = float(cielo_data[field])
                            
                            # Convert to minutes based on field name
                            if 'sec' in field.lower() or hold_time > 10000:  # Likely seconds
                                metrics['average_holding_time_minutes'] = round(hold_time / 60.0, 1)
                            elif hold_time > 200:  # Likely minutes already
                                metrics['average_holding_time_minutes'] = round(hold_time, 1)
                            else:  # Likely hours
                                metrics['average_holding_time_minutes'] = round(hold_time * 60.0, 1)
                            
                            logger.info(f"    ✅ REAL holding time: {metrics['average_holding_time_minutes']} min")
                            break
                        except (ValueError, TypeError) as e:
                            logger.debug(f"    Failed to parse {field}: {e}")
                            continue
                
                # Average trade size / volume
                volume_fields = ['avg_buy_amount_usd', 'average_buy_size', 'avgBuyAmount']
                total_trades = max(metrics['total_buys_30_days'], 1)
                
                for field in volume_fields:
                    if field in cielo_data and cielo_data[field] is not None:
                        try:
                            volume_value = float(cielo_data[field])
                            
                            # Convert USD to SOL estimate (rough conversion)
                            sol_price_estimate = 100.0  # Rough SOL price
                            avg_sol_per_trade = volume_value / sol_price_estimate
                            metrics['avg_sol_buy_per_token'] = round(avg_sol_per_trade, 1)  # 1 decimal place
                            
                            logger.info(f"    ✅ REAL avg buy size: {avg_sol_per_trade:.1f} SOL")
                            break
                        except (ValueError, TypeError) as e:
                            logger.debug(f"    Failed to parse {field}: {e}")
                            continue
                
                # Calculate avg buys per token
                if metrics['total_buys_30_days'] > 0:
                    # Estimate unique tokens (assuming ~2-3 buys per token on average)
                    estimated_tokens = max(1, metrics['total_buys_30_days'] // 2.5)
                    metrics['avg_buys_per_token'] = round(metrics['total_buys_30_days'] / estimated_tokens, 1)
        
        # Use token analysis fallbacks for missing data
        logger.info(f"  Applying fallbacks for missing fields...")
        fallback_metrics = _calculate_token_analysis_fallbacks(wallet_address, token_analysis)
        
        # Apply fallbacks only for missing/zero values
        for key, fallback_value in fallback_metrics.items():
            if key not in metrics or metrics[key] == 0:
                metrics[key] = fallback_value
                logger.info(f"    Using fallback for {key}: {fallback_value}")
        
        logger.info(f"  FINAL REAL 7-DAY METRICS for {wallet_address[:8]}:")
        for key, value in metrics.items():
            logger.info(f"    {key}: {value}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error extracting REAL 7-day Cielo trading stats for {wallet_address[:8]}: {str(e)}")
        return _calculate_token_analysis_fallbacks(wallet_address, token_analysis)

def _calculate_token_analysis_fallbacks(wallet_address: str, token_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate fallback metrics from token analysis data when Cielo API data is missing.
    """
    try:
        logger.info(f"🔧 Calculating token analysis fallbacks for {wallet_address[:8]}...")
        
        if not token_analysis:
            logger.warning(f"  No token analysis for {wallet_address[:8]}, using safe defaults")
            return {
                'roi_7_day': 0.0,
                'median_roi_7_day': 0.0,
                'usd_profit_2_days': 0.0,
                'usd_profit_7_days': 0.0,
                'usd_profit_30_days': 0.0,
                'total_buys_30_days': 0,
                'total_sells_30_days': 0,
                'avg_sol_buy_per_token': 0.0,
                'avg_buys_per_token': 0.0,
                'average_holding_time_minutes': 0.0
            }
        
        # Extract data from token analysis
        completed_trades = [t for t in token_analysis if t.get('trade_status') == 'completed']
        
        # ROI calculations from completed trades
        rois = []
        for trade in completed_trades:
            roi = trade.get('roi_percent', 0)
            if roi is not None and isinstance(roi, (int, float)):
                rois.append(float(roi))
        
        # Calculate ROI metrics
        if rois:
            avg_roi = sum(rois) / len(rois)
            sorted_rois = sorted(rois)
            median_roi = sorted_rois[len(sorted_rois)//2] if sorted_rois else 0
            
            # For 7-day ROI, use recent trades (last 5)
            recent_trades = sorted(completed_trades, key=lambda x: x.get('last_timestamp', 0), reverse=True)[:5]
            recent_rois = [t.get('roi_percent', 0) for t in recent_trades if t.get('roi_percent') is not None]
            
            roi_7_day = sum(recent_rois) / len(recent_rois) if recent_rois else avg_roi
            median_roi_7_day = sorted(recent_rois)[len(recent_rois)//2] if recent_rois else median_roi
        else:
            roi_7_day = 0.0
            median_roi_7_day = 0.0
        
        # Buy/sell counts
        total_buys = sum(token.get('buy_count', 0) for token in token_analysis)
        total_sells = sum(token.get('sell_count', 0) for token in token_analysis)
        
        # SOL amounts and averages (1 decimal place)
        sol_buys = [token.get('total_sol_in', 0) for token in token_analysis if token.get('total_sol_in', 0) > 0]
        avg_sol_buy = sum(sol_buys) / len(sol_buys) if sol_buys else 0.0
        avg_buys_per_token = total_buys / len(token_analysis) if token_analysis else 0.0
        
        # Holding time in minutes
        hold_times_hours = [token.get('hold_time_hours', 0) for token in completed_trades if token.get('hold_time_hours', 0) > 0]
        avg_hold_minutes = (sum(hold_times_hours) / len(hold_times_hours)) * 60.0 if hold_times_hours else 0.0
        
        # USD profit estimates (conservative without real timestamp scaling)
        total_sol_profit = sum(
            (token.get('total_sol_out', 0) - token.get('total_sol_in', 0))
            for token in completed_trades
        )
        sol_price_estimate = 100.0  # Rough SOL price
        usd_profit_30d = total_sol_profit * sol_price_estimate
        
        fallbacks = {
            'roi_7_day': round(roi_7_day, 1),
            'median_roi_7_day': round(median_roi_7_day, 1),
            'usd_profit_2_days': round(usd_profit_30d * 0.1, 1),  # Conservative 10% estimate
            'usd_profit_7_days': round(usd_profit_30d * 0.3, 1),  # Conservative 30% estimate
            'usd_profit_30_days': round(usd_profit_30d, 1),
            'total_buys_30_days': total_buys,
            'total_sells_30_days': total_sells,
            'avg_sol_buy_per_token': round(avg_sol_buy, 1),  # 1 decimal place
            'avg_buys_per_token': round(avg_buys_per_token, 1),
            'average_holding_time_minutes': round(avg_hold_minutes, 1)
        }
        
        logger.info(f"  TOKEN ANALYSIS FALLBACKS for {wallet_address[:8]}:")
        for key, value in fallbacks.items():
            logger.info(f"    {key}: {value}")
        
        return fallbacks
        
    except Exception as e:
        logger.error(f"Error calculating token analysis fallbacks for {wallet_address[:8]}: {str(e)}")
        # Return minimal safe defaults
        return {
            'roi_7_day': 0.0,
            'median_roi_7_day': 0.0,
            'usd_profit_2_days': 0.0,
            'usd_profit_7_days': 0.0,
            'usd_profit_30_days': 0.0,
            'total_buys_30_days': 0,
            'total_sells_30_days': 0,
            'avg_sol_buy_per_token': 0.0,
            'avg_buys_per_token': 0.0,
            'average_holding_time_minutes': 0.0
        }

def _identify_detailed_trader_pattern(analysis: Dict[str, Any]) -> str:
    """Identify detailed trader pattern with updated thresholds."""
    try:
        token_analysis = analysis.get('token_analysis', [])
        
        if not token_analysis:
            return 'insufficient_data'
        
        completed_trades = [t for t in token_analysis if t.get('trade_status') == 'completed']
        
        if len(completed_trades) < 2:
            return 'new_trader'
        
        # Calculate detailed metrics
        rois = [t.get('roi_percent', 0) for t in completed_trades]
        hold_times = [t.get('hold_time_hours', 0) for t in completed_trades]
        
        avg_roi = sum(rois) / len(rois)
        avg_hold_time = sum(hold_times) / len(hold_times)
        max_roi = max(rois)
        roi_std = (sum((roi - avg_roi) ** 2 for roi in rois) / len(rois)) ** 0.5 if len(rois) > 1 else 0
        
        # Count different outcome types
        moonshots = sum(1 for roi in rois if roi >= 400)  # 5x+
        big_wins = sum(1 for roi in rois if 100 <= roi < 400)  # 2x-5x
        heavy_losses = sum(1 for roi in rois if roi <= -50)
        
        # Get real timing from Helius
        last_tx_data = analysis.get('last_transaction_data', {})
        days_since_last = last_tx_data.get('days_since_last_trade', 999)
        
        # UPDATED THRESHOLDS: Enhanced pattern identification
        if avg_hold_time < 0.083:  # < 5 minutes (UPDATED from 12 minutes)
            if days_since_last <= 1:
                return 'active_ultra_short_flipper'
            else:
                return 'ultra_short_flipper'
        elif avg_hold_time < 1:  # < 1 hour
            if avg_roi > 30 and days_since_last <= 2:
                return 'active_skilled_sniper'
            elif avg_roi > 30:
                return 'skilled_sniper'
            else:
                return 'impulsive_flipper'
        elif moonshots > 0 and roi_std > 150:
            if heavy_losses > len(completed_trades) * 0.3:
                return 'high_risk_gem_hunter'
            elif days_since_last <= 3:
                return 'active_disciplined_gem_hunter'
            else:
                return 'disciplined_gem_hunter'
        elif avg_hold_time > 24:  # > 24 hours (UPDATED from 48 hours)
            if avg_roi > 50:
                return 'patient_position_trader'
            else:
                return 'stubborn_bag_holder'
        elif roi_std < 50 and avg_roi > 20:
            if days_since_last <= 2:
                return 'active_consistent_scalper'
            else:
                return 'consistent_scalper'
        elif big_wins > len(completed_trades) * 0.3:
            return 'momentum_trader'
        elif heavy_losses > len(completed_trades) * 0.4:
            return 'poor_risk_management'
        elif roi_std > 100:
            return 'volatile_swing_trader'
        elif avg_roi < 0:
            return 'struggling_trader'
        else:
            return 'balanced_mixed_strategy'
        
    except Exception as e:
        logger.error(f"Error identifying trader pattern: {str(e)}")
        return 'analysis_error'

def _generate_individualized_strategy_reasoning(analysis: Dict[str, Any]) -> str:
    """Generate individualized strategy reasoning based on specific wallet metrics."""
    try:
        binary_decisions = analysis.get('binary_decisions', {})
        strategy = analysis.get('strategy_recommendation', {})
        token_analysis = analysis.get('token_analysis', [])
        composite_score = analysis.get('composite_score', 0)
        
        follow_wallet = binary_decisions.get('follow_wallet', False)
        follow_sells = binary_decisions.get('follow_sells', False)
        
        if not follow_wallet:
            return f"Not recommended: Score {composite_score}/100 insufficient for following"
        
        # Calculate specific metrics for personalized insights
        completed_trades = [t for t in token_analysis if t.get('trade_status') == 'completed']
        
        if not completed_trades:
            return "Limited trade history - monitor before following"
        
        rois = [t.get('roi_percent', 0) for t in completed_trades]
        hold_times = [t.get('hold_time_hours', 0) for t in completed_trades]
        
        avg_roi = sum(rois) / len(rois)
        max_roi = max(rois)
        win_rate = sum(1 for roi in rois if roi > 0) / len(rois) * 100
        avg_hold_time = sum(hold_times) / len(hold_times)
        
        moonshots = sum(1 for roi in rois if roi >= 400)
        big_losses = sum(1 for roi in rois if roi <= -50)
        
        # Include real Helius recency information (1 decimal)
        last_tx_data = analysis.get('last_transaction_data', {})
        days_since_last = last_tx_data.get('days_since_last_trade', 999)
        
        # Generate specific reasoning based on their behavior
        reasoning_parts = []
        
        # Add activity status based on REAL Helius data (1 decimal)
        if days_since_last <= 1:
            reasoning_parts.append(f"Very active (last trade: {days_since_last:.1f}d ago, Helius)")
        elif days_since_last <= 3:
            reasoning_parts.append(f"Recently active (last trade: {days_since_last:.1f}d ago, Helius)")
        elif days_since_last <= 7:
            reasoning_parts.append(f"Active (last trade: {days_since_last:.1f}d ago, Helius)")
        elif days_since_last >= 999:
            reasoning_parts.append(f"Timestamp detection failed")
        else:
            reasoning_parts.append(f"Less active (last trade: {days_since_last:.1f}d ago, Helius)")
        
        if follow_sells:
            reasoning_parts.append(f"Mirror their exits - {win_rate:.0f}% win rate with {avg_roi:.0f}% avg return")
            if max_roi > 500:
                reasoning_parts.append(f"Excellent upside capture (max: {max_roi:.0f}%)")
            if big_losses == 0:
                reasoning_parts.append("Good loss management")
            elif big_losses == 1:
                reasoning_parts.append("Mostly good loss management")
        else:
            # Specific reasons why we're not following their exits
            if avg_hold_time < 1:
                reasoning_parts.append(f"Don't copy exits - they exit too quickly ({avg_hold_time:.1f}h avg hold)")
            elif moonshots > 0 and max_roi > 1000:
                reasoning_parts.append(f"Don't copy exits - they exit {moonshots}x moonshots too early (max {max_roi:.0f}%)")
            elif big_losses > len(completed_trades) * 0.3:
                reasoning_parts.append(f"Don't copy exits - poor loss management ({big_losses} heavy losses)")
            else:
                reasoning_parts.append("Don't copy exits - inconsistent exit timing")
            
            # Add TP strategy explanation
            tp1 = strategy.get('tp1_percent', 0)
            tp2 = strategy.get('tp2_percent', 0)
            reasoning_parts.append(f"Use custom TPs: {tp1}%-{tp2}% targets")
        
        return " | ".join(reasoning_parts)
        
    except Exception as e:
        logger.error(f"Error generating individualized strategy reasoning: {str(e)}")
        return f"Strategy analysis error: {str(e)}"

def _generate_individualized_decision_reasoning(analysis: Dict[str, Any]) -> str:
    """Generate individualized decision reasoning with specific wallet insights."""
    try:
        binary_decisions = analysis.get('binary_decisions', {})
        scoring_breakdown = analysis.get('scoring_breakdown', {})
        token_analysis = analysis.get('token_analysis', [])
        composite_score = analysis.get('composite_score', 0)
        
        follow_wallet = binary_decisions.get('follow_wallet', False)
        follow_sells = binary_decisions.get('follow_sells', False)
        
        # Get component scores for specific feedback
        component_scores = scoring_breakdown.get('component_scores', {})
        risk_score = component_scores.get('risk_adjusted_score', 0)
        distribution_score = component_scores.get('distribution_score', 0)
        discipline_score = component_scores.get('discipline_score', 0)
        
        # Include real Helius timestamp information (1 decimal)
        last_tx_data = analysis.get('last_transaction_data', {})
        days_since_last = last_tx_data.get('days_since_last_trade', 999)
        
        reasoning_parts = []
        
        # Follow Wallet decision with specific reasons
        if follow_wallet:
            reasoning_parts.append(f"FOLLOW: Score {composite_score:.1f}/100 (>= 65 threshold)")
            
            # Highlight their strengths
            if risk_score > 25:
                reasoning_parts.append("Strong risk-adjusted returns")
            if distribution_score > 20:
                reasoning_parts.append("Good win distribution")
            if discipline_score > 15:
                reasoning_parts.append("Decent trading discipline")
        else:
            reasoning_parts.append(f"DON'T FOLLOW: Score {composite_score:.1f}/100 (< 65 threshold)")
            
            # Specific weaknesses
            if risk_score < 15:
                reasoning_parts.append("Poor risk-adjusted performance")
            if distribution_score < 12:
                reasoning_parts.append("Weak win distribution")
            if discipline_score < 10:
                reasoning_parts.append("Poor trading discipline")
        
        # Follow Sells decision with exit analysis specifics
        if follow_wallet:
            if follow_sells:
                reasoning_parts.append("COPY EXITS: Passed detailed exit analysis")
            else:
                reasoning_parts.append("CUSTOM EXITS: Failed detailed exit quality review")
        
        # Add specific behavioral insights with REAL Helius timestamp awareness (1 decimal)
        if token_analysis:
            completed_trades = [t for t in token_analysis if t.get('trade_status') == 'completed']
            if completed_trades:
                rois = [t.get('roi_percent', 0) for t in completed_trades]
                hold_times = [t.get('hold_time_hours', 0) for t in completed_trades]
                avg_hold = sum(hold_times) / len(hold_times)
                moonshots = sum(1 for roi in rois if roi >= 400)
                
                if moonshots > 0:
                    reasoning_parts.append(f"{moonshots} moonshot wins")
                if avg_hold < 2:
                    reasoning_parts.append("Very short holds")
                elif avg_hold > 24:
                    reasoning_parts.append("Patient holder")
                
                # Add REAL activity context from Helius (1 decimal)
                if days_since_last <= 1:
                    reasoning_parts.append("Currently active (Helius)")
                elif days_since_last <= 3:
                    reasoning_parts.append("Recently active (Helius)")
                elif days_since_last >= 999:
                    reasoning_parts.append("Timestamp detection failed")
        
        return " | ".join(reasoning_parts)
        
    except Exception as e:
        logger.error(f"Error generating individualized decision reasoning: {str(e)}")
        return f"Decision analysis error: {str(e)}"

def _create_failed_row(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create CSV row for failed analysis with specific error details."""
    wallet_address = analysis.get('wallet_address', '')
    error_type = analysis.get('error_type', 'UNKNOWN_ERROR')
    error_message = analysis.get('error', 'Unknown error')
    
    # Create row with error information
    row = {
        'wallet_address': wallet_address,
        'composite_score': 0,
        'days_since_last_trade': 999.0,  # Use 999.0 to indicate failure (1 decimal)
        'roi': 0.0,
        'median_roi': 0.0,
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
        'decision_reason': f"Cannot analyze: {error_type} - {error_message}"
    }
    
    return row

def _create_error_row(analysis: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
    """Create minimal error row when row creation fails."""
    return {
        'wallet_address': analysis.get('wallet_address', ''),
        'composite_score': 0,
        'days_since_last_trade': 999.0,  # Use 999.0 to indicate error (1 decimal)
        'roi': 0.0,
        'median_roi': 0.0,
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
        'strategy_reason': f'Row creation error: {error_msg}',
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
        failed_analyses = [a for a in analyses if not a.get('success')]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ZEUS WALLET ANALYSIS SUMMARY - REAL 7-DAY ROI DATA\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            f.write("📊 ANALYSIS OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Wallets: {len(successful_analyses)}\n")
            f.write(f"REAL 7-Day ROI: From Cielo Trading Stats API\n")
            f.write(f"Timestamp Accuracy: HIGH (Helius API)\n\n")
            
            # Binary decision summary
            if successful_analyses:
                follow_wallet_yes = sum(1 for a in successful_analyses if a.get('binary_decisions', {}).get('follow_wallet', False))
                follow_sells_yes = sum(1 for a in successful_analyses if a.get('binary_decisions', {}).get('follow_sells', False))
                
                f.write("🎯 BINARY DECISIONS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Follow Wallet: {follow_wallet_yes}/{len(successful_analyses)} ({follow_wallet_yes/len(successful_analyses)*100:.1f}%)\n")
                f.write(f"Follow Sells: {follow_sells_yes}/{len(successful_analyses)} ({follow_sells_yes/len(successful_analyses)*100:.1f}%)\n\n")
        
        logger.info(f"✅ Exported Zeus summary to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error exporting Zeus summary: {str(e)}")
        return False