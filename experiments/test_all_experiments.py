"""
Test Script for All Three Experiments
This script runs quick validation tests for all three experiment scripts to ensure they work correctly.
"""

import sys
import traceback
from datetime import datetime

def test_template_comparison():
    """Test the template comparison experiment"""
    print("Testing Template Comparison Experiment...")
    try:
        from experiment_1_template_comparison import TemplateComparisonExperiment
        
        # Initialize experiment with reduced parameters for testing
        exp = TemplateComparisonExperiment()
        exp.config['evolution_cycles'] = 2  # Reduce for testing
        exp.config['samples_per_cycle'] = 5  # Reduce for testing
        
        # Test data loading
        train_data, test_data = exp.load_and_prepare_data()
        print(f"  [OK] Data loaded: {len(train_data)} train, {len(test_data)} test samples")
        
        # Test agent setup
        baseline_agents = exp.setup_baseline_agents()
        adaptutor_agent = exp.setup_adaptutor_agent()
        print(f"  [OK] Agents setup: {len(baseline_agents)} baseline + 1 AdaPTutor agent")
        
        # Test one cycle of evolution
        sample_data = train_data.sample(n=3)
        results = exp.run_cycle(adaptutor_agent, sample_data, 0, 'adaptutor_rl_il')
        print(f"  [OK] Evolution cycle test completed")
        
        print("  [OK] Template Comparison Experiment: PASSED")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Template Comparison Experiment: FAILED")
        print(f"    Error: {str(e)}")
        traceback.print_exc()
        return False

def test_llm_comparison():
    """Test the LLM comparison experiment"""
    print("\nTesting LLM Comparison Experiment...")
    try:
        from experiment_2_llm_comparison import LLMComparisonExperiment
        
        # Initialize experiment with reduced parameters for testing
        exp = LLMComparisonExperiment()
        exp.config['generations'] = 2  # Reduce for testing
        exp.config['samples_per_generation'] = 3  # Reduce for testing
        
        # Test data loading
        train_data, test_data = exp.load_and_prepare_data()
        print(f"  [OK] Data loaded: {len(train_data)} train, {len(test_data)} test samples")
        
        # Test agent setup
        baseline_agents = exp.setup_baseline_agents()
        adaptutor_agent = exp.setup_adaptutor_agent()
        print(f"  [OK] Agents setup: {len(baseline_agents)} baseline + 1 AdaPTutor agent")
        
        # Test one generation
        sample_data = train_data.sample(n=3)
        results = exp.run_generation(adaptutor_agent, sample_data, 0, 'adaptutor_ga')
        print(f"  [OK] Generation test completed")
        
        print("  [OK] LLM Comparison Experiment: PASSED")
        return True
        
    except Exception as e:
        print(f"  [FAIL] LLM Comparison Experiment: FAILED")
        print(f"    Error: {str(e)}")
        traceback.print_exc()
        return False

def test_ablation_study():
    """Test the ablation study experiment"""
    print("\nTesting Ablation Study Experiment...")
    try:
        from experiment_3_ablation_study import AblationStudyExperiment
        
        # Initialize experiment with reduced parameters for testing
        exp = AblationStudyExperiment()
        exp.config['evolution_cycles'] = 2  # Reduce for testing
        exp.config['ga_generations'] = 2   # Reduce for testing
        exp.config['samples_per_cycle'] = 3  # Reduce for testing
        
        # Test data loading
        train_data, test_data = exp.load_and_prepare_data()
        print(f"  [OK] Data loaded: {len(train_data)} train, {len(test_data)} test samples")
        
        # Test template agent setup
        template_agents = exp.setup_template_agents()
        print(f"  [OK] Template agents setup: {len(template_agents)} agents")
        
        # Test LLM agent setup
        llm_agents = exp.setup_llm_agents()
        print(f"  [OK] LLM agents setup: {len(llm_agents)} agents")
        
        # Test template agent evaluation
        sample_data = test_data.sample(n=3)
        test_results = exp.evaluate_template_agent(template_agents['rl_il_full'], sample_data, 'rl_il_full')
        print(f"  [OK] Template agent evaluation test completed")
        
        # Test LLM agent evaluation
        test_results = exp.evaluate_llm_agent(llm_agents['ga_evolution'], sample_data, 'ga_evolution')
        print(f"  [OK] LLM agent evaluation test completed")
        
        print("  [OK] Ablation Study Experiment: PASSED")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Ablation Study Experiment: FAILED")
        print(f"    Error: {str(e)}")
        traceback.print_exc()
        return False

def test_data_consistency():
    """Test that all experiments use consistent data"""
    print("\nTesting Data Consistency Across Experiments...")
    try:
        from experiment_1_template_comparison import TemplateComparisonExperiment
        from experiment_2_llm_comparison import LLMComparisonExperiment
        from experiment_3_ablation_study import AblationStudyExperiment
        
        # Load data from all experiments
        exp1 = TemplateComparisonExperiment()
        exp2 = LLMComparisonExperiment()
        exp3 = AblationStudyExperiment()
        
        train1, test1 = exp1.load_and_prepare_data()
        train2, test2 = exp2.load_and_prepare_data()
        train3, test3 = exp3.load_and_prepare_data()
        
        # Check consistency
        assert len(train1) == len(train2) == len(train3), "Training data sizes don't match"
        assert len(test1) == len(test2) == len(test3), "Test data sizes don't match"
        
        print(f"  [OK] Data consistency verified: {len(train1)} train, {len(test1)} test samples")
        print("  [OK] Data Consistency Test: PASSED")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Data Consistency Test: FAILED")
        print(f"    Error: {str(e)}")
        traceback.print_exc()
        return False

def test_results_saving():
    """Test that results can be saved properly"""
    print("\nTesting Results Saving...")
    try:
        import json
        import os
        
        # Test JSON serialization with sample results
        sample_results = {
            'experiment_id': 'test_experiment',
            'timestamp': datetime.now().isoformat(),
            'config': {'test_param': 1.0},
            'results': {
                'agent1': {
                    'quality_scores': [0.5, 0.6, 0.7],
                    'response_times': [0.1, 0.2, 0.15]
                }
            }
        }
        
        # Create test results directory
        test_dir = 'test_results'
        os.makedirs(test_dir, exist_ok=True)
        
        # Save and load test results
        test_file = f"{test_dir}/test_results.json"
        with open(test_file, 'w') as f:
            json.dump(sample_results, f, indent=2)
        
        with open(test_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results['experiment_id'] == sample_results['experiment_id']
        
        # Clean up
        os.remove(test_file)
        os.rmdir(test_dir)
        
        print("  [OK] Results saving and loading verified")
        print("  [OK] Results Saving Test: PASSED")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Results Saving Test: FAILED")
        print(f"    Error: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("EXPERIMENT VALIDATION TESTS")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        test_template_comparison,
        test_llm_comparison,
        test_ablation_study,
        test_data_consistency,
        test_results_saving
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] {test_func.__name__}: FAILED with exception")
            print(f"    Error: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n[OK] ALL TESTS PASSED! Experiments are ready to run.")
        return True
    else:
        print(f"\n[FAIL] {failed} test(s) failed. Please fix the issues before running experiments.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)