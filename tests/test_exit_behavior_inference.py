#!/usr/bin/env python3
"""
Unit Tests for Zeus Exit Behavior Inference Logic
Tests the core CORRECTED EXIT ANALYSIS functionality that separates 
actual trader exit behavior from final token ROI performance.
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import Zeus modules
try:
    from zeus_utils import ZeusUtils
    from zeus_analyzer import ZeusAnalyzer
    from zeus_api_manager import ZeusAPIManager
except ImportError as e:
    print(f"Warning: Could not import Zeus modules: {e}")
    print("Make sure you're running tests from the project root directory")

class TestExitBehaviorInference(unittest.TestCase):
    """Test suite for exit behavior inference logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_config = {
            'analysis': {
                'very_short_threshold_minutes': 5,
                'long_hold_threshold_hours': 24,
                'enable_corrected_exit_analysis': True
            },
            'features': {
                'enable_exit_behavior_inference': True,
                'enable_realistic_tp_sl': True
            }
        }
        
        # Mock API manager
        self.mock_api_manager = Mock(spec=ZeusAPIManager)
        
    def test_flipper_exit_behavior_inference(self):
        """Test exit behavior inference for flipper patterns (<5 minutes)."""
        # Test case 1: Flipper exits quickly, token pumps after
        result = ZeusUtils.infer_actual_exit_behavior(
            final_roi=500.0,  # Token went to 500% after they sold
            hold_time_hours=0.05,  # 3 minutes
            num_swaps=2,
            pattern='flipper'
        )
        
        self.assertLess(result['actual_exit_roi'], 100,
                       "Flipper should have much lower exit ROI than final token ROI")
        self.assertGreater(result['actual_exit_roi'], 0,
                          "Flipper should have positive exit but much less than 500%")
        self.assertEqual(result['exit_strategy'], 'quick_flip_before_pump')
        self.assertEqual(result['confidence'], 'high')
        
        # Test case 2: Flipper with small gain
        result = ZeusUtils.infer_actual_exit_behavior(
            final_roi=25.0,
            hold_time_hours=0.08,  # 5 minutes
            num_swaps=1,
            pattern='flipper'
        )
        
        self.assertLessEqual(result['actual_exit_roi'], 25.0,
                           "Small gain flipper should get most of the performance")
        self.assertGreater(result['actual_exit_roi'], 15.0,
                          "But should still be realistic for flipper")
    
    def test_multiple_swap_exit_inference(self):
        """Test exit behavior inference for multiple swap scenarios."""
        # Test case: Multiple swaps suggest partial exits
        result = ZeusUtils.infer_actual_exit_behavior(
            final_roi=300.0,  # Token did 3x
            hold_time_hours=2.0,  # 2 hours
            num_swaps=5,  # Multiple partial exits
            pattern='sniper'
        )
        
        self.assertLess(result['actual_exit_roi'], 300.0,
                       "Multiple swaps should result in lower than final ROI")
        self.assertGreater(result['actual_exit_roi'], 150.0,
                          "But should capture significant portion")
        self.assertEqual(result['exit_strategy'], 'partial_exits')
        self.assertEqual(result['confidence'], 'medium')
    
    def test_long_hold_diamond_hands(self):
        """Test exit behavior inference for long holds (diamond hands)."""
        result = ZeusUtils.infer_actual_exit_behavior(
            final_roi=1000.0,  # 10x token
            hold_time_hours=48.0,  # 2 days
            num_swaps=2,
            pattern='gem_hunter'
        )
        
        self.assertGreaterEqual(result['actual_exit_roi'], 700.0,
                               "Diamond hands should capture most of the gains")
        self.assertLessEqual(result['actual_exit_roi'], 1000.0,
                            "But might not get 100% of peak")
        self.assertEqual(result['exit_strategy'], 'diamond_hands')
    
    def test_realistic_tp_sl_calculation(self):
        """Test realistic TP/SL calculation based on patterns."""
        # Test flipper TP/SL (should be low)
        flipper_tp_sl = ZeusUtils.calculate_realistic_tp_sl('flipper')
        
        self.assertLessEqual(flipper_tp_sl['tp1'], 80,
                           "Flipper TP1 should be realistic (≤80%)")
        self.assertLessEqual(flipper_tp_sl['tp2'], 120,
                           "Flipper TP2 should be realistic (≤120%)")
        self.assertGreaterEqual(flipper_tp_sl['stop_loss'], -25,
                               "Flipper stop loss should be tight")
        
        # Test gem hunter TP/SL (can be higher)
        gem_hunter_tp_sl = ZeusUtils.calculate_realistic_tp_sl('gem_hunter')
        
        self.assertGreaterEqual(gem_hunter_tp_sl['tp1'], 80,
                               "Gem hunter TP1 can be higher")
        self.assertLessEqual(gem_hunter_tp_sl['tp1'], 300,
                           "But still realistic")
        self.assertLessEqual(gem_hunter_tp_sl['tp2'], 600,
                           "Even gem hunters have limits")
    
    def test_tp_sl_validation_for_patterns(self):
        """Test TP/SL validation prevents inflated recommendations."""
        # Test inflated flipper TP/SL gets corrected
        corrected_tp1, corrected_tp2, corrected_sl = ZeusUtils.validate_tp_sl_for_pattern(
            pattern='flipper',
            tp1=200.0,  # Way too high for flipper
            tp2=500.0,  # Way too high for flipper
            stop_loss=-60.0
        )
        
        self.assertLessEqual(corrected_tp1, 80,
                           "Inflated flipper TP1 should be corrected")
        self.assertLessEqual(corrected_tp2, 120,
                           "Inflated flipper TP2 should be corrected")
        
        # Test gem hunter TP/SL allows higher levels
        corrected_tp1, corrected_tp2, corrected_sl = ZeusUtils.validate_tp_sl_for_pattern(
            pattern='gem_hunter',
            tp1=200.0,
            tp2=400.0,
            stop_loss=-40.0
        )
        
        self.assertEqual(corrected_tp1, 200,
                        "Reasonable gem hunter TP1 should not be changed")
        self.assertEqual(corrected_tp2, 400,
                        "Reasonable gem hunter TP2 should not be changed")
    
    def test_inflation_detection(self):
        """Test detection of inflated TP/SL recommendations."""
        # Test flipper with inflated TP levels
        detection = ZeusUtils.detect_inflated_tp_sl('flipper', 200, 500)
        
        self.assertTrue(detection['inflated'],
                       "Should detect inflated flipper TP levels")
        self.assertEqual(detection['severity'], 'critical',
                        "Flipper with 200%+ TP should be critical")
        self.assertLessEqual(detection['corrected_tp1'], 80,
                           "Should provide corrected levels")
        
        # Test gem hunter with reasonable levels
        detection = ZeusUtils.detect_inflated_tp_sl('gem_hunter', 150, 300)
        
        self.assertFalse(detection['inflated'],
                        "Should not flag reasonable gem hunter levels")
    
    def test_pattern_identification_with_corrected_thresholds(self):
        """Test trader pattern identification with corrected thresholds."""
        # Test flipper identification (very short hold)
        flipper_metrics = {
            'avg_hold_time_hours': 0.05,  # 3 minutes
            'avg_roi': 30,
            'moonshot_rate': 0,
            'win_rate': 60,
            'total_trades': 10,
            'corrected_exit_analysis': True
        }
        
        pattern = ZeusUtils.identify_corrected_trader_pattern(flipper_metrics, use_corrected_thresholds=True)
        self.assertEqual(pattern, 'skilled_flipper',
                        "Short hold with good ROI should be skilled flipper")
        
        # Test position trader identification (long hold)
        position_metrics = {
            'avg_hold_time_hours': 48,  # 2 days
            'avg_roi': 120,
            'moonshot_rate': 5,
            'win_rate': 55,
            'total_trades': 8,
            'corrected_exit_analysis': True
        }
        
        pattern = ZeusUtils.identify_corrected_trader_pattern(position_metrics, use_corrected_thresholds=True)
        self.assertEqual(pattern, 'position_trader',
                        "Long hold with good ROI should be position trader")
    
    def test_safe_type_conversion(self):
        """Test safe type conversion functions."""
        # Test safe float conversion
        self.assertEqual(ZeusUtils._safe_float("123.45", 0), 123.45)
        self.assertEqual(ZeusUtils._safe_float("invalid", 0), 0)
        self.assertEqual(ZeusUtils._safe_float(None, 5.0), 5.0)
        self.assertEqual(ZeusUtils._safe_float(float('nan'), 10.0), 10.0)
        
        # Test safe int conversion
        self.assertEqual(ZeusUtils._safe_int("123", 0), 123)
        self.assertEqual(ZeusUtils._safe_int("invalid", 0), 0)
        self.assertEqual(ZeusUtils._safe_int(123.7, 0), 123)
        self.assertEqual(ZeusUtils._safe_int(None, 5), 5)
    
    def test_corrected_analysis_summary(self):
        """Test creation of corrected analysis summary."""
        sample_results = [
            {
                'success': True,
                'trade_pattern_analysis': {
                    'pattern': 'flipper',
                    'exit_analysis_corrected': True
                },
                'strategy_recommendation': {
                    'tp1_percent': 30,
                    'tp2_percent': 60
                }
            },
            {
                'success': True,
                'trade_pattern_analysis': {
                    'pattern': 'gem_hunter',
                    'exit_analysis_corrected': True
                },
                'strategy_recommendation': {
                    'tp1_percent': 150,
                    'tp2_percent': 350
                }
            }
        ]
        
        summary = ZeusUtils.create_corrected_analysis_summary(sample_results)
        
        self.assertEqual(summary['total_wallets'], 2)
        self.assertEqual(summary['successful_analyses'], 2)
        self.assertEqual(summary['pattern_distribution']['flipper'], 1)
        self.assertEqual(summary['pattern_distribution']['gem_hunter'], 1)
        self.assertEqual(summary['corrected_exit_analysis']['wallets_with_corrected_data'], 2)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with zero or negative values
        result = ZeusUtils.infer_actual_exit_behavior(
            final_roi=0.0,
            hold_time_hours=0.0,
            num_swaps=0,
            pattern='unknown'
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('actual_exit_roi', result)
        self.assertIn('exit_strategy', result)
        
        # Test with None values
        result = ZeusUtils.infer_actual_exit_behavior(
            final_roi=None,
            hold_time_hours=None,
            num_swaps=None,
            pattern=None
        )
        
        self.assertIsInstance(result, dict)
        
        # Test pattern validation with invalid inputs
        tp1, tp2, sl = ZeusUtils.validate_tp_sl_for_pattern('invalid_pattern', -100, -200, 50)
        
        self.assertGreater(tp1, 0, "Should return reasonable defaults for invalid pattern")
        self.assertGreater(tp2, tp1, "TP2 should be greater than TP1")
        self.assertLess(sl, 0, "Stop loss should be negative")

class TestZeusAnalyzerExitAnalysis(unittest.TestCase):
    """Test suite for ZeusAnalyzer exit analysis integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_api_manager = Mock(spec=ZeusAPIManager)
        self.sample_config = {
            'analysis': {
                'very_short_threshold_minutes': 5,
                'long_hold_threshold_hours': 24,
                'enable_corrected_exit_analysis': True
            }
        }
    
    @patch('zeus_analyzer.ZeusScorer')
    def test_corrected_exit_analysis_integration(self, mock_scorer):
        """Test integration of corrected exit analysis in Zeus analyzer."""
        # Mock the scorer
        mock_scorer_instance = Mock()
        mock_scorer.return_value = mock_scorer_instance
        mock_scorer_instance.calculate_composite_score.return_value = {
            'composite_score': 75.0,
            'component_scores': {}
        }
        
        analyzer = ZeusAnalyzer(self.mock_api_manager, self.sample_config)
        
        # Test token list analysis with corrected exit inference
        sample_tokens = [
            {
                'roi_percentage': 500,  # Final token ROI
                'holding_time_seconds': 180,  # 3 minutes (flipper)
                'num_swaps': 2,
                'total_pnl_usd': 100
            },
            {
                'roi_percentage': 200,  # Final token ROI
                'holding_time_seconds': 7200,  # 2 hours
                'num_swaps': 4,  # Multiple swaps
                'total_pnl_usd': 50
            }
        ]
        
        # This would normally be called within the analyzer
        result = analyzer._analyze_token_list_corrected(sample_tokens)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('exit_analysis_corrected', False))
        
        # The average exit ROI should be much lower than the average final ROI
        # because the corrected analysis infers actual exit behavior
        avg_roi = result.get('avg_roi', 0)
        avg_final_roi = sum(t['roi_percentage'] for t in sample_tokens) / len(sample_tokens)
        
        self.assertLess(avg_roi, avg_final_roi * 0.8,
                       "Corrected exit analysis should show lower ROI than final token performance")

class TestPerformanceMonitoring(unittest.TestCase):
    """Test suite for performance monitoring functionality."""
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization."""
        from zeus_cli import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        self.assertEqual(monitor.total_cost, 0)
        self.assertEqual(len(monitor.api_calls), 0)
        self.assertEqual(len(monitor.response_times), 0)
    
    def test_api_call_recording(self):
        """Test recording of API calls."""
        from zeus_cli import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Record some API calls
        monitor.record_api_call('cielo_trading_stats', 30, 2.5, True)
        monitor.record_api_call('helius_timestamp', 0, 1.2, True)
        monitor.record_api_call('cielo_token_pnl', 5, 3.1, False)
        
        self.assertEqual(monitor.total_cost, 35)
        self.assertEqual(len(monitor.api_calls), 3)
        
        # Check cielo_trading_stats stats
        cielo_stats = monitor.api_calls['cielo_trading_stats']
        self.assertEqual(cielo_stats['calls'], 1)
        self.assertEqual(cielo_stats['cost'], 30)
        self.assertEqual(cielo_stats['success'], 1)
        self.assertEqual(cielo_stats['errors'], 0)
        
        # Check failed call
        token_pnl_stats = monitor.api_calls['cielo_token_pnl']
        self.assertEqual(token_pnl_stats['errors'], 1)
    
    def test_performance_report_generation(self):
        """Test performance report generation."""
        from zeus_cli import PerformanceMonitor
        import time
        
        monitor = PerformanceMonitor()
        
        # Simulate some time passing
        time.sleep(0.1)
        
        # Record some calls
        monitor.record_api_call('test_api', 10, 1.5, True)
        monitor.record_api_call('test_api', 15, 2.0, True)
        
        report = monitor.get_performance_report()
        
        self.assertGreater(report['session_duration'], 0)
        self.assertEqual(report['total_api_calls'], 2)
        self.assertEqual(report['total_cost'], 25)
        self.assertGreater(report['avg_response_time'], 0)
        self.assertIn('test_api', report['api_breakdown'])

def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestExitBehaviorInference))
    suite.addTests(loader.loadTestsFromTestCase(TestZeusAnalyzerExitAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMonitoring))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY - Zeus Exit Behavior Inference")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    print(f"\n🎯 KEY TESTS COVERED:")
    print(f"  ✅ Exit Behavior Inference (flipper vs gem hunter)")
    print(f"  ✅ Realistic TP/SL Calculation (pattern-aware)")
    print(f"  ✅ TP/SL Validation (prevents inflation)")
    print(f"  ✅ Inflation Detection (flags unrealistic levels)")
    print(f"  ✅ Pattern Identification (corrected thresholds)")
    print(f"  ✅ Performance Monitoring (API costs & response times)")
    print(f"  ✅ Safe Type Conversion (handles invalid inputs)")
    print(f"  ✅ Edge Cases (error handling)")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)