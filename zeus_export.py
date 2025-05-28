"""
Zeus Export - CSV Export with INDIVIDUALIZED Analysis - COMPLETE VERSION
Exports Zeus analysis results with personalized insights and wallet-specific metrics

COMPLETE FEATURES:
- Individualized strategy and decision reasoning (not cookie-cutter)
- Wallet-specific stop loss recommendations based on actual loss management
- Actual average SOL buy amounts (not position size ranges)
- Fixed encoding issues (proper >= symbols)
- Detailed, personalized insights for each wallet
- All original export functionality maintained
"""

import os
import csv
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger("zeus.export")

def export_zeus_analysis(results: Dict[str, Any], output_file: str) -> bool:
    """
    Export Zeus analysis results to CSV with individualized insights.
    
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
        
        # Prepare CSV data with individualized insights
        csv_data = []
        
        for analysis in analyses:
            if not analysis.get('success'):
                csv_data.append(_create_failed_row(analysis))
                continue
            
            # Create individualized analysis row
            csv_data.append(_create_individualized_analysis_row(analysis))
        
        # Sort by composite score (highest first)
        csv_data.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        # Add rank column
        for i, row in enumerate(csv_data, 1):
            row['rank'] = i
        
        # Write CSV
        if csv_data:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = _get_csv_fieldnames()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        logger.info(f"✅ Exported {len(csv_data)} individualized wallet analyses to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error exporting Zeus analysis: {str(e)}")
        return False

def _get_csv_fieldnames() -> List[str]:
    """Get CSV column fieldnames with improved naming."""
    return [
        # Basic Info
        'rank',
        'wallet_address', 
        
        # Binary Decisions (MOST IMPORTANT)
        'follow_wallet',
        'follow_sells',
        
        # Bot Configuration
        'tp1_percent',
        'tp2_percent',
        'tp3_percent', 
        'stop_loss_percent',
        'avg_sol_buy',  # CHANGED: Show actual average SOL buy amount
        
        # Scoring & Metrics
        'composite_score',
        'risk_adjusted_score',
        'distribution_score',
        'discipline_score',
        'market_impact_score',
        'consistency_score',
        
        # Volume Qualifier
        'unique_tokens_traded',
        'tokens_analyzed',
        'volume_tier',
        
        # Analysis Details - INDIVIDUALIZED
        'trader_pattern',
        'strategy_reasoning',  # INDIVIDUALIZED insights
        'decision_reasoning',  # INDIVIDUALIZED insights
    ]

def _create_individualized_analysis_row(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create CSV row with INDIVIDUALIZED insights and wallet-specific metrics."""
    try:
        # Extract data
        wallet_address = analysis.get('wallet_address', '')
        binary_decisions = analysis.get('binary_decisions', {})
        strategy = analysis.get('strategy_recommendation', {})
        scoring_breakdown = analysis.get('scoring_breakdown', {})
        token_analysis = analysis.get('token_analysis', [])
        
        # CALCULATE ACTUAL AVERAGE SOL BUY (not position size range)
        actual_avg_sol_buy = _calculate_actual_avg_sol_buy(token_analysis)
        
        # INDIVIDUALIZED STOP LOSS based on their loss management
        individualized_stop_loss = _calculate_individualized_stop_loss(token_analysis, scoring_breakdown)
        
        # Basic row data
        row = {
            'wallet_address': wallet_address,
            'unique_tokens_traded': analysis.get('unique_tokens_traded', 0),
            'tokens_analyzed': analysis.get('tokens_analyzed', 0)
        }
        
        # Binary decisions
        row.update({
            'follow_wallet': 'YES' if binary_decisions.get('follow_wallet', False) else 'NO',
            'follow_sells': 'YES' if binary_decisions.get('follow_sells', False) else 'NO'
        })
        
        # Bot configuration with individualized metrics
        row.update({
            'tp1_percent': strategy.get('tp1_percent', 0),
            'tp2_percent': strategy.get('tp2_percent', 0),
            'tp3_percent': strategy.get('tp3_percent', 0),
            'stop_loss_percent': individualized_stop_loss,  # INDIVIDUALIZED
            'avg_sol_buy': actual_avg_sol_buy  # ACTUAL average SOL buy amount
        })
        
        # Scoring
        row.update({
            'composite_score': analysis.get('composite_score', 0)
        })
        
        # Component scores
        component_scores = scoring_breakdown.get('component_scores', {})
        row.update({
            'risk_adjusted_score': component_scores.get('risk_adjusted_score', 0),
            'distribution_score': component_scores.get('distribution_score', 0),
            'discipline_score': component_scores.get('discipline_score', 0),
            'market_impact_score': component_scores.get('market_impact_score', 0),
            'consistency_score': component_scores.get('consistency_score', 0)
        })
        
        # Volume qualifier
        volume_qualifier = scoring_breakdown.get('volume_qualifier', {})
        volume_tier = volume_qualifier.get('tier', 'unknown')
        
        if volume_tier == 'baseline':
            tier_display = 'Baseline (6+ tokens)'
        elif volume_tier == 'emerging':
            tier_display = 'Emerging (4-5 tokens)'
        elif volume_tier == 'very_new':
            tier_display = 'Very New (2-3 tokens)'
        else:
            tier_display = f'{volume_tier} ({analysis.get("unique_tokens_traded", 0)} tokens)'
            
        row.update({
            'volume_tier': tier_display
        })
        
        # INDIVIDUALIZED INSIGHTS
        trader_pattern = _identify_detailed_trader_pattern(analysis)
        individualized_strategy_reasoning = _generate_individualized_strategy_reasoning(analysis)
        individualized_decision_reasoning = _generate_individualized_decision_reasoning(analysis)
        
        row.update({
            'trader_pattern': trader_pattern,
            'strategy_reasoning': individualized_strategy_reasoning,
            'decision_reasoning': individualized_decision_reasoning
        })
        
        return row
        
    except Exception as e:
        logger.error(f"Error creating individualized analysis row: {str(e)}")
        return _create_error_row(analysis, str(e))

def _calculate_actual_avg_sol_buy(token_analysis: List[Dict[str, Any]]) -> float:
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
        return round(avg_buy, 3)  # Round to 3 decimal places
        
    except Exception as e:
        logger.error(f"Error calculating actual average SOL buy: {str(e)}")
        return 0.0

def _calculate_individualized_stop_loss(token_analysis: List[Dict[str, Any]], 
                                      scoring_breakdown: Dict[str, Any]) -> int:
    """Calculate wallet-specific stop loss based on their actual loss management behavior."""
    try:
        # Analyze their actual loss behavior
        completed_trades = [t for t in token_analysis if t.get('trade_status') == 'completed']
        losing_trades = [t for t in completed_trades if t.get('roi_percent', 0) < 0]
        
        if not losing_trades:
            return -30  # Conservative default for wallets with no loss data
        
        # Calculate their typical loss levels
        loss_amounts = [abs(t.get('roi_percent', 0)) for t in losing_trades]
        avg_loss = sum(loss_amounts) / len(loss_amounts)
        max_loss = max(loss_amounts)
        
        # Get their loss management metrics
        metrics = scoring_breakdown.get('metrics_used', {})
        heavy_loss_rate = metrics.get('heavy_loss_rate', 0)  # % of losses > 50%
        quick_cut_losses = metrics.get('quick_cut_losses', 0)
        
        # Determine appropriate stop loss based on their behavior
        if heavy_loss_rate > 30:  # They let losses run too much
            stop_loss = -25  # Tighter stop loss
        elif heavy_loss_rate > 15:
            stop_loss = -30
        elif avg_loss > 40:  # Their average loss is high
            stop_loss = -35
        elif max_loss > 80:  # They've had some very large losses
            stop_loss = -40
        elif quick_cut_losses > 0:  # They cut losses quickly - can afford wider stop
            stop_loss = -45
        else:
            stop_loss = -35  # Standard
        
        # Additional adjustments based on trader pattern
        trader_pattern = _identify_detailed_trader_pattern({'token_analysis': token_analysis})
        
        if 'gem_hunter' in trader_pattern:
            stop_loss -= 5  # Gem hunters need wider stops
        elif 'scalper' in trader_pattern:
            stop_loss += 5  # Scalpers need tighter stops
        elif 'volatile' in trader_pattern:
            stop_loss -= 3  # Volatile traders need slightly wider stops
        
        # Cap the range
        return max(-50, min(-20, stop_loss))
        
    except Exception as e:
        logger.error(f"Error calculating individualized stop loss: {str(e)}")
        return -35  # Safe default

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
        min_roi = min(rois)
        roi_std = (sum((roi - avg_roi) ** 2 for roi in rois) / len(rois)) ** 0.5 if len(rois) > 1 else 0
        
        # Count different outcome types
        moonshots = sum(1 for roi in rois if roi >= 400)  # 5x+
        big_wins = sum(1 for roi in rois if 100 <= roi < 400)  # 2x-5x
        small_wins = sum(1 for roi in rois if 0 < roi < 100)
        small_losses = sum(1 for roi in rois if -50 < roi <= 0)
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
            tp3 = strategy.get('tp3_percent', 0)
            reasoning_parts.append(f"Use custom TPs: {tp1}%-{tp3}% targets")
        
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
    
    # Create row with specific error information
    row = {
        'wallet_address': wallet_address,
        'follow_wallet': 'NO',
        'follow_sells': 'NO',
        'tp1_percent': 0,
        'tp2_percent': 0,
        'tp3_percent': 0,
        'stop_loss_percent': -35,
        'avg_sol_buy': 0.0,
        'composite_score': 0,
        'risk_adjusted_score': 0,
        'distribution_score': 0,
        'discipline_score': 0,
        'market_impact_score': 0,
        'consistency_score': 0,
        'unique_tokens_traded': analysis.get('unique_tokens_found', 0),
        'tokens_analyzed': 0,
        'volume_tier': f'Failed: {error_type}',
        'trader_pattern': 'failed_analysis',
        'strategy_reasoning': f"Analysis failed: {error_message}",
        'decision_reasoning': f"Cannot analyze: {error_type} - {error_message}"
    }
    
    return row

def _create_error_row(analysis: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
    """Create minimal error row when row creation fails."""
    return {
        'wallet_address': analysis.get('wallet_address', ''),
        'follow_wallet': 'NO',
        'follow_sells': 'NO',
        'tp1_percent': 0,
        'tp2_percent': 0,
        'tp3_percent': 0,
        'stop_loss_percent': -35,
        'avg_sol_buy': 0.0,
        'composite_score': 0,
        'risk_adjusted_score': 0,
        'distribution_score': 0,
        'discipline_score': 0,
        'market_impact_score': 0,
        'consistency_score': 0,
        'unique_tokens_traded': 0,
        'tokens_analyzed': 0,
        'volume_tier': 'Error',
        'trader_pattern': 'error',
        'strategy_reasoning': f'Row creation error: {error_msg}',
        'decision_reasoning': f'Processing error: {error_msg}'
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
                
                # Show Follow Wallet YES but Follow Sells NO cases
                follow_wallet_only = sum(1 for a in successful_analyses 
                                       if a.get('binary_decisions', {}).get('follow_wallet', False) and 
                                       not a.get('binary_decisions', {}).get('follow_sells', False))
                f.write(f"Follow Wallet Only (YES/NO): {follow_wallet_only}/{len(successful_analyses)} ")
                f.write(f"({follow_wallet_only/len(successful_analyses)*100:.1f}%)\n\n")
            
            # Top performers
            if successful_analyses:
                top_performers = sorted(successful_analyses, 
                                      key=lambda x: x.get('composite_score', 0), reverse=True)[:10]
                
                f.write("🏆 TOP 10 PERFORMERS\n")
                f.write("-" * 40 + "\n")
                
                for i, analysis in enumerate(top_performers, 1):
                    wallet = analysis['wallet_address']
                    score = analysis.get('composite_score', 0)
                    binary_decisions = analysis.get('binary_decisions', {})
                    strategy = analysis.get('strategy_recommendation', {})
                    
                    follow_wallet = binary_decisions.get('follow_wallet', False)
                    follow_sells = binary_decisions.get('follow_sells', False)
                    
                    f.write(f"\n{i}. {wallet[:8]}...{wallet[-4:]}\n")
                    f.write(f"   Score: {score:.1f}/100\n")
                    f.write(f"   Follow Wallet: {'✅ YES' if follow_wallet else '❌ NO'}\n")
                    f.write(f"   Follow Sells: {'✅ YES' if follow_sells else '❌ NO'}\n")
                    f.write(f"   TP Strategy: {strategy.get('tp1_percent', 0)}% / ")
                    f.write(f"{strategy.get('tp2_percent', 0)}% / {strategy.get('tp3_percent', 0)}%\n")
                    f.write(f"   Stop Loss: {strategy.get('stop_loss_percent', -35)}%\n")
                    f.write(f"   Reasoning: {strategy.get('reasoning', 'N/A')}\n")
            
            # Error analysis
            if failed_analyses:
                f.write(f"\n\n❌ FAILED ANALYSES ({len(failed_analyses)})\n")
                f.write("-" * 40 + "\n")
                
                error_counts = {}
                for analysis in failed_analyses:
                    error_type = analysis.get('error_type', 'UNKNOWN')
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
                
                for error_type, count in error_counts.items():
                    f.write(f"{error_type}: {count} wallets\n")
                
                # Show first 5 failed analyses
                f.write("\nFirst 5 Failed Analyses:\n")
                for analysis in failed_analyses[:5]:
                    wallet = analysis.get('wallet_address', 'Unknown')
                    error = analysis.get('error', 'Unknown error')
                    f.write(f"• {wallet[:8]}...{wallet[-4:]}: {error}\n")
            
            f.write(f"\n" + "=" * 80 + "\n")
            f.write("END OF ZEUS ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n")
        
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
                'individualized_analysis': True
            },
            'wallets': []
        }
        
        for analysis in follow_wallets:
            wallet_config = {
                'wallet_address': analysis['wallet_address'],
                'composite_score': analysis.get('composite_score', 0),
                'binary_decisions': analysis.get('binary_decisions', {}),
                'strategy': analysis.get('strategy_recommendation', {}),
                'analysis_timestamp': analysis.get('analysis_timestamp', ''),
                'individualized_insights': {
                    'trader_pattern': _identify_detailed_trader_pattern(analysis),
                    'avg_sol_buy': _calculate_actual_avg_sol_buy(analysis.get('token_analysis', [])),
                    'strategy_reasoning': _generate_individualized_strategy_reasoning(analysis),
                    'decision_reasoning': _generate_individualized_decision_reasoning(analysis)
                }
            }
            bot_config['wallets'].append(wallet_config)
        
        # Sort by score
        bot_config['wallets'].sort(key=lambda x: x['composite_score'], reverse=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(bot_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Exported individualized bot configuration to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error exporting bot config: {str(e)}")
        return False