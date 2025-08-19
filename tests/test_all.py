"""
Comprehensive test suite for sentiment analysis project.
"""
import sys
import os
import unittest
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test modules
from test_basic import TestBasicFunctionality, run_basic_tests
from test_model_integration import TestModelIntegration, TestModelPerformance, run_integration_tests
from test_config import TestDataGenerator, TestAssertions, get_test_config

class TestSuite:
    """Main test suite runner."""
    
    def __init__(self):
        self.start_time = None
        self.test_results = {}
        
    def run_all_tests(self, verbose=True):
        """Run all test suites."""
        print("🚀 Starting Comprehensive Test Suite")
        print("=" * 60)
        self.start_time = time.time()
        
        # Run console tests first
        if verbose:
            print("\n📋 Running Console Tests...")
            self._run_console_tests()
        
        # Run unit tests
        print("\n🔬 Running Unit Tests...")
        self._run_unit_tests()
        
        # Performance tests
        print("\n⚡ Running Performance Tests...")
        self._run_performance_tests()
        
        # Summary
        return self._print_summary()
        
    def _run_console_tests(self):
        """Run console-based tests."""
        try:
            print("\n1️⃣ Basic Functionality Tests:")
            run_basic_tests()
            self.test_results['console_basic'] = 'PASSED'
        except Exception as e:
            print(f"❌ Console basic tests failed: {e}")
            self.test_results['console_basic'] = 'FAILED'
            
        try:
            print("\n2️⃣ Integration Tests:")
            run_integration_tests()
            self.test_results['console_integration'] = 'PASSED'
        except Exception as e:
            print(f"❌ Console integration tests failed: {e}")
            self.test_results['console_integration'] = 'FAILED'
            
    def _run_unit_tests(self):
        """Run unit test suites."""
        # Create test loader
        loader = unittest.TestLoader()
        
        # Load test suites
        test_suites = [
            loader.loadTestsFromTestCase(TestBasicFunctionality),
            loader.loadTestsFromTestCase(TestModelIntegration),
            loader.loadTestsFromTestCase(TestModelPerformance),
        ]
        
        # Run each suite
        for i, suite in enumerate(test_suites, 1):
            print(f"\n{i}️⃣ Running Unit Test Suite {i}...")
            runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
            result = runner.run(suite)
            
            suite_name = f'unit_suite_{i}'
            if result.wasSuccessful():
                self.test_results[suite_name] = 'PASSED'
                print(f"✅ Suite {i} passed: {result.testsRun} tests")
            else:
                self.test_results[suite_name] = 'FAILED'
                print(f"❌ Suite {i} failed: {len(result.failures)} failures, {len(result.errors)} errors")
                
    def _run_performance_tests(self):
        """Run performance-specific tests."""
        from utils.inference import SentimentInference
        from test_config import TestDataGenerator
        
        try:
            print("\n📊 Testing Inference Speed...")
            inference = SentimentInference()
            
            # Single prediction speed
            test_text = "This is a test for performance measurement."
            start_time = time.time()
            result = inference.predict_single(test_text)
            single_time = time.time() - start_time
            
            print(f"  Single prediction: {single_time:.3f}s")
            
            # Batch prediction speed
            test_texts = TestDataGenerator.get_sample_texts(count=50)
            start_time = time.time()
            batch_results = inference.predict_batch(test_texts)
            batch_time = time.time() - start_time
            
            print(f"  Batch prediction (50 texts): {batch_time:.3f}s")
            print(f"  Average per text: {batch_time/len(test_texts):.3f}s")
            
            # Performance criteria
            if single_time < 1.0 and batch_time < 10.0:
                self.test_results['performance'] = 'PASSED'
                print("✅ Performance tests passed")
            else:
                self.test_results['performance'] = 'FAILED'
                print("❌ Performance tests failed - too slow")
                
        except Exception as e:
            print(f"❌ Performance tests failed: {e}")
            self.test_results['performance'] = 'FAILED'
            
    def _print_summary(self):
        """Print test summary."""
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results.values() if result == 'PASSED')
        failed = len(self.test_results) - passed
        
        print(f"Total runtime: {total_time:.2f} seconds")
        print(f"Tests passed: {passed}")
        print(f"Tests failed: {failed}")
        print()
        
        for test_name, status in self.test_results.items():
            icon = "✅" if status == "PASSED" else "❌"
            print(f"{icon} {test_name}: {status}")
            
        print("\n" + "=" * 60)
        
        if failed == 0:
            print("🎉 ALL TESTS PASSED! Project is ready to use.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
            
        return failed == 0

def run_specific_tests(test_type='all'):
    """Run specific type of tests."""
    suite = TestSuite()
    
    if test_type == 'basic':
        print("🧪 Running Basic Tests Only...")
        run_basic_tests()
        return True
    elif test_type == 'integration':
        print("🔗 Running Integration Tests Only...")
        run_integration_tests()
        return True
    elif test_type == 'performance':
        print("⚡ Running Performance Tests Only...")
        suite._run_performance_tests()
        return suite.test_results.get('performance') == 'PASSED'
    elif test_type == 'unit':
        print("🔬 Running Unit Tests Only...")
        suite._run_unit_tests()
        return all(status == 'PASSED' for status in suite.test_results.values())
    else:
        print("🚀 Running All Tests...")
        return suite.run_all_tests()

def quick_smoke_test():
    """Run a quick smoke test to verify basic functionality."""
    print("💨 Running Quick Smoke Test...")
    print("-" * 40)
    
    try:
        from utils.inference import SentimentInference
        from utils.text_preprocessor import TextPreprocessor
        from utils.model_utils import SentimentModel
        
        # Test imports
        print("✅ Imports successful")
        
        # Test basic prediction
        inference = SentimentInference()
        result = inference.predict_single("This is a test")
        assert 'score' in result and 'confidence' in result
        print("✅ Basic prediction works")
        
        # Test preprocessing
        preprocessor = TextPreprocessor()
        processed = preprocessor.preprocess_text("Test TEXT! https://example.com")
        assert len(processed) > 0
        print("✅ Text preprocessing works")
        
        # Test model
        model = SentimentModel()
        prediction = model.predict("Test")
        assert 'score' in prediction
        print("✅ Model prediction works")
        
        print("\n🎉 Smoke test PASSED - Basic functionality working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Smoke test FAILED: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run sentiment analysis tests')
    parser.add_argument('--type', choices=['all', 'basic', 'integration', 'performance', 'unit', 'smoke'], 
                       default='all', help='Type of tests to run')
    parser.add_argument('--quick', action='store_true', help='Run quick smoke test only')
    
    args = parser.parse_args()
    
    if args.quick:
        success = quick_smoke_test()
    else:
        success = run_specific_tests(args.type)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
