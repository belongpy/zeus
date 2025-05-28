"""
Zeus Export - CSV Export with Bot Configuration
Exports Zeus analysis results to CSV with bot-friendly configuration format

Features:
- Binary decision columns (Follow Wallet Y/N, Follow Sells Y/N)
- TP/SL strategy configuration
- Scoring breakdown
- Bot-ready configuration format
"""

import os
import csv
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger("zeus.export")

def export_zeus_analysis(results: Dict[str, Any], output_file: str) -> bool:
    """
    Export Zeus analysis results to CSV with bot configuration.
    
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
        
        # Prepare CSV data
        csv_data = []
        
        for analysis in analyses:
            if not analysis.get('success'):
                # Add failed analysis with minimal data
                csv_data.append(_create_failed_row(analysis))
                continue
            
            # Create successful analysis row
            csv_data.append(_create_analysis_row(analysis))
        
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
        
        logger.info(f"✅ Exported {len(csv_data)} wallet analyses to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error exporting Zeus analysis: {str(e)}")
        return False

def _get_csv_fieldnames() -> List[str]:
    """Get CSV column fieldnames in proper order."""
    return [
        # Basic Info
        'rank',
        'wallet_address',
        'analysis_timestamp',
        'analysis_status',
        
        # Binary Decisions (MOST IMPORTANT)
        'follow_wallet',
        'follow_sells',
        
        # Bot Configuration
        'copy_entries',
        'copy_exits',
        'tp1_percent',
        'tp2_percent', 
        'tp3_percent',
        'stop_loss_percent',
        'position_size_sol',
        
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
        'volume_qualifier_tier',
        'volume_multiplier',
        
        # Analysis Details
        'analysis_days',
        'conclusive_analysis',
        'analysis_phase',
        'trader_pattern',
        
        # Strategy Details
        'strategy_reasoning',
        'decision_reasoning',
        
        # Error Handling
        'error_type',
        'error_message'
    ]

def _create_analysis_row(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create CSV row from successful analysis."""
    try:
        # Extract data
        wallet_address = analysis.get('wallet_address', '')
        binary_decisions = analysis.get('binary_decisions', {})
        strategy = analysis.get('strategy_recommendation', {})
        scoring = analysis.get('scoring_breakdown', {})
        
        # Basic info
        row = {
            'wallet_address': wallet_address,
            'analysis_timestamp': analysis.get('analysis_timestamp', ''),
            'analysis_status': 'SUCCESS',
            'analysis_days': analysis.get('analysis_days', 30),
            'unique_tokens_traded': analysis.get('unique_tokens_traded', 0),
            'tokens_analyzed': analysis.get('tokens_analyzed', 0),
            'conclusive_analysis': 'YES' if analysis.get('conclusive_analysis', True) else 'NO',
            'analysis_phase': analysis.get('analysis_phase', 'unknown')
        }
        
        # Binary decisions
        row.update({
            'follow_wallet': 'YES' if binary_decisions.get('follow_wallet', False) else 'NO',
            'follow_sells': 'YES' if binary_decisions.get('follow_sells', False) else 'NO'
        })
        
        # Bot configuration
        row.update({
            'copy_entries': 'YES' if strategy.get('copy_entries', False) else 'NO',
            'copy_exits': 'YES' if strategy.get('copy_exits', False) else 'NO',
            'tp1_percent': strategy.get('tp1_percent', 0),
            'tp2_percent': strategy.get('tp2_percent', 0),
            'tp3_percent': strategy.get('tp3_percent', 0),
            'stop_loss_percent': strategy.get('stop_loss_percent', -35),
            'position_size_sol': strategy.get('position_size_sol', '0')
        })
        
        # Scoring
        row.update({
            'composite_score': analysis.get('composite_score', 0),
            'risk_adjusted_score': scoring.get('risk_adjusted_score', 0),
            'distribution_score': scoring.get('distribution_score', 0),
            'discipline_score': scoring.get('discipline_score', 0),
            'market_impact_score': scoring.get('market_impact_score', 0),
            'consistency_score': scoring.get('consistency_score', 0)
        })
        
        # Volume qualifier
        volume_qualifier = scoring.get('volume_qualifier', {})
        row.update({
            'volume_qualifier_tier': volume_qualifier.get('tier', 'unknown'),
            'volume_multiplier': volume_qualifier.get('multiplier', 1.0)
        })
        
        # Strategy details
        row.update({
            'strategy_reasoning': strategy.get('reasoning', ''),
            'decision_reasoning': binary_decisions.get('decision_reasoning', ''),
            'trader_pattern': _identify_trader_pattern_from_analysis(analysis)
        })
        
        # No errors
        row.update({
            'error_type': '',
            'error_message': ''
        })
        
        return row
        
    except Exception as e:
        logger.error(f"Error creating analysis row: {str(e)}")
        return _create_failed_row({
            'wallet_address': analysis.get('wallet_address', ''),
            'error': f'Row creation error: {str(e)}',
            'error_type': 'ROW_CREATION_ERROR'
        })

def _create_failed_row(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create CSV row for failed analysis."""
    wallet_address = analysis.get('wallet_address', '')
    error_type = analysis.get('error_type', 'UNKNOWN_ERROR')
    error_message = analysis.get('error', 'Unknown error')
    
    # Create minimal row with error info
    row = {
        'wallet_address': wallet_address,
        'analysis_timestamp': datetime.now().isoformat(),
        'analysis_status': 'FAILED',
        'error_type': error_type,
        'error_message': error_message
    }
    
    # Fill other fields with defaults
    fieldnames = _get_csv_fieldnames()
    for field in fieldnames:
        if field not in row:
            if field in ['follow_wallet', 'follow_sells', 'copy_entries', 'copy_exits', 'conclusive_analysis']:
                row[field] = 'NO'
            elif field in ['tp1_percent', 'tp2_percent', 'tp3_percent', 'composite_score', 
                          'risk_adjusted_score', 'distribution_score', 'discipline_score',
                          'market_impact_score', 'consistency_score', 'unique_tokens_traded',
                          'tokens_analyzed', 'analysis_days', 'volume_multiplier']:
                row[field] = 0
            elif field == 'stop_loss_percent':
                row[field] = -35
            elif field == 'position_size_sol':
                row[field] = '0'
            else:
                row[field] = ''
    
    return row

def _identify_trader_pattern_from_analysis(analysis: Dict[str, Any]) -> str:
    """Identify trader pattern from analysis data."""
    try:
        token_analysis = analysis.get('token_analysis', [])
        
        if not token_analysis:
            return 'unknown'
        
        # Look for pattern indicators
        completed_trades = [t for t in token_analysis if t.get('trade_status') == 'completed']
        
        if len(completed_trades) < 2:
            return 'insufficient_data'
        
        # Calculate pattern metrics
        rois = [t.get('roi_percent', 0) for t in completed_trades]
        hold_times = [t.get('hold_time_hours', 0) for t in completed_trades]
        
        if not rois or not hold_times:
            return 'insufficient_data'
        
        avg_roi = sum(rois) / len(rois)
        avg_hold_time = sum(hold_times) / len(hold_times)
        max_roi = max(rois)
        roi_std = (sum((roi - avg_roi) ** 2 for roi in rois) / len(rois)) ** 0.5 if len(rois) > 1 else 0
        
        # Pattern identification logic
        if avg_hold_time < 0.1:  # < 6 minutes
            return 'sniper'
        elif avg_hold_time < 1:  # < 1 hour
            return 'flipper'
        elif max_roi > 400 and roi_std > 100:  # High variance, high upside
            return 'gem_hunter'
        elif avg_hold_time > 24:  # > 1 day
            return 'position_trader'
        elif roi_std < 50 and avg_roi > 20:  # Consistent profits
            return 'consistent_scalper'
        elif roi_std > 80:  # High variance
            return 'volatile_trader'
        else:
            return 'mixed_strategy'
        
    except Exception as e:
        logger.debug(f"Error identifying trader pattern: {str(e)}")
        return 'unknown'

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
            f.write(f"Smart Token Sampling: 5 → 10 tokens\n\n")
            
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
                    f.write(f"   Position Size: {strategy.get('position_size_sol', '0')} SOL\n")
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
            
            # Strategy distribution
            if successful_analyses:
                f.write(f"\n\n📋 STRATEGY DISTRIBUTION\n")
                f.write("-" * 40 + "\n")
                
                # Count by trader pattern
                patterns = {}
                for analysis in successful_analyses:
                    pattern = _identify_trader_pattern_from_analysis(analysis)
                    patterns[pattern] = patterns.get(pattern, 0) + 1
                
                for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
                    percentage = count / len(successful_analyses) * 100
                    f.write(f"{pattern}: {count} wallets ({percentage:.1f}%)\n")
            
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
        import json
        
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
                'analysis_period_days': 30
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