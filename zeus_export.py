"""
Zeus Export - CSV Export with UPDATED Column Structure
Exports Zeus analysis results with new column layout and comprehensive Cielo data integration

NEW FEATURES:
- Updated column structure per user specifications
- Comprehensive Cielo data extraction (prioritized over calculations)
- Fallback calculations using token analysis only when needed
- Cleaner, more focused CSV output
- Scoring system remains completely unchanged
"""

import os
import csv
import json
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger("zeus.export")

def export_zeus_analysis(results: Dict[str, Any], output_file: str) -> bool:
    """
    Export Zeus analysis results to CSV with updated column structure.
    
    Args:
        results: Zeus analysis results dictionary
        output_file: Output CSV file path
        
    Returns:
        bool: True if successful, False otherwise
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
        
        # Prepare CSV data with new structure
        csv_data = []
        
        for analysis in analyses:
            if not analysis.get('success'):
                csv_data.append(_create_failed_row(analysis))
                continue
            
            # Create updated analysis row
            csv_data.append(_create_updated_analysis_row(analysis))
        
        # Sort by composite score (highest first)
        csv_data.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        # Write CSV
        if csv_data:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = _get_updated_csv_fieldnames()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        logger.info(f"✅ Exported {len(csv_data)} wallet analyses with updated structure to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error exporting Zeus analysis: {str(e)}")
        return False

def _get_updated_csv_fieldnames() -> List[str]:
    """Get updated CSV column fieldnames per user specifications."""
    return [
        'wallet_address',
        'composite_score',
        'days_since_last_trade',
        'roi',  # 7-day ROI from Cielo
        'median_roi',  # 7-day median ROI from Cielo
        'usd_profit_2_days',  # Realized gains from Cielo
        'usd_profit_7_days',  # Realized gains from Cielo
        'usd_profit_30_days',  # Realized gains from Cielo
        'copy_wallet',  # YES/NO
        'copy_sells',  # YES/NO
        'tp_1',  # TP1 percentage
        'tp_2',  # TP2 percentage
        'stop_loss',  # Stop loss percentage
        'avg_sol_buy_per_token',  # Average SOL per token from Cielo
        'avg_buys_per_token',  # Average buys per token from Cielo
        'average_holding_time',  # Average holding time from Cielo
        'total_buys_30_days',  # Total buys from Cielo
        'total_sells_30_days',  # Total sells from Cielo
        'trader_pattern',
        'strategy_reason',
        'decision_reason'
    ]

def _create_updated_analysis_row(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create CSV row with updated structure and comprehensive Cielo data integration."""
    try:
        # Extract basic data
        wallet_address = analysis.get('wallet_address', '')
        binary_decisions = analysis.get('binary_decisions', {})
        strategy = analysis.get('strategy_recommendation', {})
        token_analysis = analysis.get('token_analysis', [])
        wallet_data = analysis.get('wallet_data', {})
        
        # Extract comprehensive Cielo metrics (prioritized over calculations)
        cielo_metrics = _extract_comprehensive_cielo_metrics(wallet_data, token_analysis)
        
        # Create the row with new structure
        row = {
            'wallet_address': wallet_address,
            'composite_score': round(analysis.get('composite_score', 0), 1),
            'days_since_last_trade': cielo_metrics['days_since_last_trade'],
            'roi': cielo_metrics['roi_7_day'],
            'median_roi': cielo_metrics['median_roi_7_day'],
            'usd_profit_2_days': cielo_metrics['usd_profit_2_days'],
            'usd_profit_7_days': cielo_metrics['usd_profit_7_days'],
            'usd_profit_30_days': cielo_metrics['usd_profit_30_days'],
            'copy_wallet': 'YES' if binary_decisions.get('follow_wallet', False) else 'NO',
            'copy_sells': 'YES' if binary_decisions.get('follow_sells', False) else 'NO',
            'tp_1': strategy.get('tp1_percent', 0),
            'tp_2': strategy.get('tp2_percent', 0),
            'stop_loss': strategy.get('stop_loss_percent', -35),
            'avg_sol_buy_per_token': cielo_metrics['avg_sol_buy_per_token'],
            'avg_buys_per_token': cielo_metrics['avg_buys_per_token'],
            'average_holding_time': cielo_metrics['average_holding_time'],
            'total_buys_30_days': cielo_metrics['total_buys_30_days'],
            'total_sells_30_days': cielo_metrics['total_sells_30_days'],
            'trader_pattern': _identify_detailed_trader_pattern(analysis),
            'strategy_reason': _generate_individualized_strategy_reasoning(analysis),
            'decision_reason': _generate_individualized_decision_reasoning(analysis)
        }
        
        return row
        
    except Exception as e:
        logger.error(f"Error creating updated analysis row: {str(e)}")
        return _create_error_row(analysis, str(e))

def _extract_comprehensive_cielo_metrics(wallet_data: Dict[str, Any], token_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract comprehensive metrics from Cielo data with minimal fallback calculations.
    Prioritizes Cielo API data over manual calculations.
    
    Args:
        wallet_data: Raw wallet data from Cielo API
        token_analysis: Token analysis for fallback calculations only
        
    Returns:
        Dict with all required metrics
    """
    try:
        # Initialize with defaults
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
            'average_holding_time': 0.0,
            'days_since_last_trade': 999
        }
        
        # Extract from Cielo data (prioritized)
        if isinstance(wallet_data, dict):
            # Handle nested data structure - get actual Cielo response
            cielo_data = wallet_data.get('data', wallet_data) if 'data' in wallet_data else wallet_data
            
            if isinstance(cielo_data, dict):
                logger.info(f"Extracting from Cielo data with keys: {list(cielo_data.keys())}")
                
                # ROI Metrics (7-day focus) - try multiple field name variations
                metrics['roi_7_day'] = _safe_float_extract(cielo_data, [
                    'roi_7d', 'roi_7_day', 'weekly_roi', '7d_roi', 'roi_week', 'avg_roi'
                ], 0.0)
                
                metrics['median_roi_7_day'] = _safe_float_extract(cielo_data, [
                    'median_roi_7d', 'median_roi_7_day', 'median_weekly_roi', 'median_7d_roi', 'median_roi'
                ], 0.0)
                
                # USD Profit Metrics (realized gains) - comprehensive field search
                metrics['usd_profit_2_days'] = _safe_float_extract(cielo_data, [
                    'realized_pnl_2d', 'profit_2d', 'pnl_2_day', 'realized_profit_2d', 'pnl_2d'
                ], 0.0)
                
                metrics['usd_profit_7_days'] = _safe_float_extract(cielo_data, [
                    'realized_pnl_7d', 'profit_7d', 'pnl_7_day', 'realized_profit_7d', 'pnl_7d', 'pnl'
                ], 0.0)
                
                metrics['usd_profit_30_days'] = _safe_float_extract(cielo_data, [
                    'realized_pnl_30d', 'profit_30d', 'pnl_30_day', 'realized_profit_30d', 'pnl_30d', 'total_pnl'
                ], 0.0)
                
                # Buy/Sell Counts
                metrics['total_buys_30_days'] = _safe_int_extract(cielo_data, [
                    'buy_count', 'total_buys', 'buys_30d', 'buy_transactions', 'swaps_buy'
                ], 0)
                
                metrics['total_sells_30_days'] = _safe_int_extract(cielo_data, [
                    'sell_count', 'total_sells', 'sells_30d', 'sell_transactions', 'swaps_sell'
                ], 0)
                
                # Average Holding Time (prioritize Cielo)
                metrics['average_holding_time'] = _safe_float_extract(cielo_data, [
                    'average_holding_time_hours', 'avg_hold_time_hours', 'average_holding_time_sec',
                    'avg_hold_time', 'holding_time_avg', 'mean_holding_time'
                ], 0.0)
                
                # Convert seconds to hours if needed
                if metrics['average_holding_time'] > 200:  # Likely in seconds if > 200
                    metrics['average_holding_time'] = metrics['average_holding_time'] / 3600.0
                
                # Average SOL per token - calculate from Cielo volume data
                total_buy_amount = _safe_float_extract(cielo_data, [
                    'total_buy_amount_usd', 'buy_volume_usd', 'total_volume_bought'
                ], 0.0)
                
                total_tokens = _safe_int_extract(cielo_data, [
                    'total_tokens', 'unique_tokens', 'tokens_traded', 'distinct_tokens'
                ], 0)
                
                if total_buy_amount > 0 and total_tokens > 0:
                    # Convert USD to SOL estimate (rough conversion)
                    sol_price_estimate = 100.0  # Rough SOL price estimate
                    total_sol_volume = total_buy_amount / sol_price_estimate
                    metrics['avg_sol_buy_per_token'] = total_sol_volume / total_tokens
                
                # Average buys per token
                if metrics['total_buys_30_days'] > 0 and total_tokens > 0:
                    metrics['avg_buys_per_token'] = metrics['total_buys_30_days'] / total_tokens
                
                # Days since last trade - try to get from Cielo
                last_trade_timestamp = _safe_int_extract(cielo_data, [
                    'last_trade_timestamp', 'latest_transaction', 'last_activity', 'most_recent_trade'
                ], 0)
                
                if last_trade_timestamp > 0:
                    current_time = int(time.time())
                    days_diff = (current_time - last_trade_timestamp) / 86400
                    metrics['days_since_last_trade'] = max(0, int(days_diff))
                
                logger.info(f"Extracted Cielo metrics: ROI={metrics['roi_7_day']:.1f}%, "
                           f"Profit_7d=${metrics['usd_profit_7_days']:.0f}, "
                           f"Buys={metrics['total_buys_30_days']}, "
                           f"Avg_Hold={metrics['average_holding_time']:.1f}h")
        
        # Fallback calculations only for missing critical data
        if metrics['days_since_last_trade'] == 999:  # No Cielo data for this
            metrics['days_since_last_trade'] = _calculate_days_since_last_trade_fallback(token_analysis)
        
        if metrics['avg_sol_buy_per_token'] == 0.0:  # Fallback calculation
            metrics['avg_sol_buy_per_token'] = _calculate_actual_avg_sol_buy_fallback(token_analysis)
        
        if metrics['avg_buys_per_token'] == 0.0:  # Fallback calculation
            metrics['avg_buys_per_token'] = _calculate_avg_buys_per_token_fallback(token_analysis)
        
        if metrics['average_holding_time'] == 0.0:  # Fallback calculation
            metrics['average_holding_time'] = _calculate_average_holding_time_fallback(token_analysis)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error extracting comprehensive Cielo metrics: {str(e)}")
        # Return fallback calculations for all metrics
        return _calculate_all_fallback_metrics(token_analysis)

def _safe_float_extract(data: Dict[str, Any], field_names: List[str], default: float = 0.0) -> float:
    """Safely extract float value from multiple possible field names."""
    for field_name in field_names:
        if field_name in data:
            try:
                value = data[field_name]
                if value is not None:
                    return float(value)
            except (ValueError, TypeError):
                continue
    return default

def _safe_int_extract(data: Dict[str, Any], field_names: List[str], default: int = 0) -> int:
    """Safely extract integer value from multiple possible field names."""
    for field_name in field_names:
        if field_name in data:
            try:
                value = data[field_name]
                if value is not None:
                    return int(value)
            except (ValueError, TypeError):
                continue
    return default

def _calculate_days_since_last_trade_fallback(token_analysis: List[Dict[str, Any]]) -> int:
    """Calculate days since the most recent trade using token analysis."""
    try:
        if not token_analysis:
            return 999
        
        most_recent_timestamp = 0
        
        for token in token_analysis:
            first_ts = token.get('first_timestamp', 0)
            last_ts = token.get('last_timestamp', 0)
            latest_ts = max(first_ts, last_ts)
            if latest_ts > most_recent_timestamp:
                most_recent_timestamp = latest_ts
        
        if most_recent_timestamp == 0:
            return 999
        
        current_timestamp = int(time.time())
        seconds_diff = current_timestamp - most_recent_timestamp
        days_diff = int(seconds_diff / 86400)
        
        return max(0, days_diff)
        
    except Exception as e:
        logger.error(f"Error calculating days since last trade: {str(e)}")
        return 999

def _calculate_actual_avg_sol_buy_fallback(token_analysis: List[Dict[str, Any]]) -> float:
    """Calculate the actual average SOL buy amount from token analysis."""
    try:
        sol_buys = []
        for token in token_analysis:
            sol_in = token.get('total_sol_in', 0)
            if sol_in > 0:
                sol_buys.append(sol_in)
        
        if not sol_buys:
            return 0.0
        
        avg_buy = sum(sol_buys) / len(sol_buys)
        return round(avg_buy, 3)
        
    except Exception as e:
        logger.error(f"Error calculating actual average SOL buy: {str(e)}")
        return 0.0

def _calculate_avg_buys_per_token_fallback(token_analysis: List[Dict[str, Any]]) -> float:
    """Calculate the average number of buys per token."""
    try:
        if not token_analysis:
            return 0.0
        
        total_buys = 0
        tokens_with_buys = 0
        
        for token in token_analysis:
            buy_count = token.get('buy_count', 0)
            if buy_count > 0:
                total_buys += buy_count
                tokens_with_buys += 1
        
        if tokens_with_buys == 0:
            return 0.0
        
        avg_buys = total_buys / tokens_with_buys
        return round(avg_buys, 1)
        
    except Exception as e:
        logger.error(f"Error calculating avg buys per token: {str(e)}")
        return 0.0

def _calculate_average_holding_time_fallback(token_analysis: List[Dict[str, Any]]) -> float:
    """Calculate average holding time from token analysis."""
    try:
        if not token_analysis:
            return 0.0
        
        holding_times = []
        for token in token_analysis:
            hold_time = token.get('hold_time_hours', 0)
            if hold_time > 0:
                holding_times.append(hold_time)
        
        if not holding_times:
            return 0.0
        
        avg_holding_time = sum(holding_times) / len(holding_times)
        return round(avg_holding_time, 1)
        
    except Exception as e:
        logger.error(f"Error calculating average holding time: {str(e)}")
        return 0.0

def _calculate_all_fallback_metrics(token_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate all metrics using fallback methods when Cielo data is unavailable."""
    return {
        'roi_7_day': 0.0,  # Cannot calculate without historical price data
        'median_roi_7_day': 0.0,  # Cannot calculate without historical price data
        'usd_profit_2_days': 0.0,  # Cannot calculate without USD conversion
        'usd_profit_7_days': 0.0,  # Cannot calculate without USD conversion
        'usd_profit_30_days': 0.0,  # Cannot calculate without USD conversion
        'total_buys_30_days': sum(token.get('buy_count', 0) for token in token_analysis),
        'total_sells_30_days': sum(token.get('sell_count', 0) for token in token_analysis),
        'avg_sol_buy_per_token': _calculate_actual_avg_sol_buy_fallback(token_analysis),
        'avg_buys_per_token': _calculate_avg_buys_per_token_fallback(token_analysis),
        'average_holding_time': _calculate_average_holding_time_fallback(token_analysis),
        'days_since_last_trade': _calculate_days_since_last_trade_fallback(token_analysis)
    }

def _identify_detailed_trader_pattern(analysis: Dict[str, Any]) -> str:
    """Identify detailed trader pattern with specific characteristics."""
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
        
        # Detailed pattern identification
        if avg_hold_time < 0.2:  # < 12 minutes
            return 'ultra_short_flipper'
        elif avg_hold_time < 1:  # < 1 hour
            if avg_roi > 30:
                return 'skilled_sniper'
            else:
                return 'impulsive_flipper'
        elif moonshots > 0 and roi_std > 150:
            if heavy_losses > len(completed_trades) * 0.3:
                return 'high_risk_gem_hunter'
            else:
                return 'disciplined_gem_hunter'
        elif avg_hold_time > 48:  # > 2 days
            if avg_roi > 50:
                return 'patient_position_trader'
            else:
                return 'stubborn_bag_holder'
        elif roi_std < 50 and avg_roi > 20:
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
        
        # Generate specific reasoning based on their behavior
        reasoning_parts = []
        
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
        
        reasoning_parts = []
        
        # Follow Wallet decision with specific reasons
        if follow_wallet:
            reasoning_parts.append(f"FOLLOW: Score {composite_score}/100 (>= 65 threshold)")
            
            # Highlight their strengths
            if risk_score > 25:
                reasoning_parts.append("Strong risk-adjusted returns")
            if distribution_score > 20:
                reasoning_parts.append("Good win distribution")
            if discipline_score > 15:
                reasoning_parts.append("Decent trading discipline")
        else:
            reasoning_parts.append(f"DON'T FOLLOW: Score {composite_score}/100 (< 65 threshold)")
            
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
        
        # Add specific behavioral insights
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
        
        return " | ".join(reasoning_parts)
        
    except Exception as e:
        logger.error(f"Error generating individualized decision reasoning: {str(e)}")
        return f"Decision analysis error: {str(e)}"

def _create_failed_row(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create CSV row for failed analysis with specific error details."""
    wallet_address = analysis.get('wallet_address', '')
    error_type = analysis.get('error_type', 'UNKNOWN_ERROR')
    error_message = analysis.get('error', 'Unknown error')
    
    # Create row with error information using new structure
    row = {
        'wallet_address': wallet_address,
        'composite_score': 0,
        'days_since_last_trade': 999,
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
        'average_holding_time': 0.0,
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
        'days_since_last_trade': 999,
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
        'average_holding_time': 0.0,
        'total_buys_30_days': 0,
        'total_sells_30_days': 0,
        'trader_pattern': 'error',
        'strategy_reason': f'Row creation error: {error_msg}',
        'decision_reason': f'Processing error: {error_msg}'
    }

def export_zeus_summary(results: Dict[str, Any], output_file: str) -> bool:
    """
    Export Zeus analysis summary to text file.
    
    Args:
        results: Zeus analysis results
        output_file: Output text file path
        
    Returns:
        bool: True if successful
    """
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
            f.write("ZEUS WALLET ANALYSIS SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            f.write("📊 ANALYSIS OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Wallets Requested: {results.get('total_requested', 0)}\n")
            f.write(f"Successfully Analyzed: {len(successful_analyses)}\n")
            f.write(f"Failed Analyses: {len(failed_analyses)}\n")
            f.write(f"Analysis Period: 30 days\n")
            f.write(f"Minimum Token Requirement: 6 unique trades\n")
            f.write(f"Enhanced Exit Analysis: 5-10 tokens deep study\n\n")
            
            # Binary decision summary
            if successful_analyses:
                follow_wallet_count = sum(1 for a in successful_analyses 
                                        if a.get('binary_decisions', {}).get('follow_wallet', False))
                follow_sells_count = sum(1 for a in successful_analyses 
                                       if a.get('binary_decisions', {}).get('follow_sells', False))
                
                f.write("🎯 BINARY DECISION SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Follow Wallet (YES): {follow_wallet_count}/{len(successful_analyses)} ")
                f.write(f"({follow_wallet_count/len(successful_analyses)*100:.1f}%)\n")
                f.write(f"Follow Sells (YES): {follow_sells_count}/{len(successful_analyses)} ")
                f.write(f"({follow_sells_count/len(successful_analyses)*100:.1f}%)\n\n")
        
        logger.info(f"✅ Exported Zeus summary to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error exporting Zeus summary: {str(e)}")
        return False

def export_bot_config_json(results: Dict[str, Any], output_file: str) -> bool:
    """
    Export bot configuration in JSON format.
    
    Args:
        results: Zeus analysis results
        output_file: Output JSON file path
        
    Returns:
        bool: True if successful
    """
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        analyses = results.get('analyses', [])
        successful_analyses = [a for a in analyses if a.get('success')]
        
        # Filter to only wallets we should follow
        follow_wallets = [a for a in successful_analyses 
                         if a.get('binary_decisions', {}).get('follow_wallet', False)]
        
        bot_config = {
            'generated_at': datetime.now().isoformat(),
            'zeus_version': '1.0',
            'analysis_summary': {
                'total_analyzed': len(successful_analyses),
                'follow_wallets': len(follow_wallets),
                'analysis_period_days': 30,
                'enhanced_exit_analysis': True,
                'comprehensive_cielo_integration': True
            },
            'wallets': []
        }
        
        for analysis in follow_wallets:
            wallet_config = {
                'wallet_address': analysis['wallet_address'],
                'composite_score': analysis.get('composite_score', 0),
                'binary_decisions': analysis.get('binary_decisions', {}),
                'strategy': analysis.get('strategy_recommendation', {}),
                'analysis_timestamp': analysis.get('analysis_timestamp', '')
            }
            bot_config['wallets'].append(wallet_config)
        
        # Sort by score
        bot_config['wallets'].sort(key=lambda x: x['composite_score'], reverse=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(bot_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Exported bot configuration to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error exporting bot config: {str(e)}")
        return False