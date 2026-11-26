"""
Demo LLM Evolution Script
Demonstrates the evolution of LLM responses over time with comprehensive metrics and analysis
"""

import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# Import the integrated system
from integrated_llm_system import IntegratedLLMSystem

class LLMEvolutionDemo:
    def __init__(self):
        """Initialize the evolution demo"""
        self.system = None
        self.demo_results = []
        self.evolution_snapshots = []
        
    def run_evolution_demo(self, 
                          target_input: str = "I do not understand neural networks",
                          evolution_cycles: int = 8,
                          requests_per_cycle: int = 3):
        """
        Run a comprehensive evolution demonstration
        
        Args:
            target_input: The input to track evolution for
            evolution_cycles: Number of evolution cycles to run
            requests_per_cycle: Number of requests per evolution cycle
        """
        print("🚀 LLM Evolution Demonstration")
        print("=" * 80)
        print(f"Target Input: '{target_input}'")
        print(f"Evolution Cycles: {evolution_cycles}")
        print(f"Requests per Cycle: {requests_per_cycle}")
        print("=" * 80)
        
        # Initialize system
        print("\n📋 Initializing Integrated LLM System...")
        self.system = IntegratedLLMSystem(
            evolution_config={
                'population_size': 6,
                'mutation_rate': 0.4,
                'crossover_rate': 0.8,
                'elite_ratio': 0.25,
                'evolution_frequency': requests_per_cycle  # Evolve after each cycle
            }
        )
        
        # Prepare diverse training inputs to drive evolution
        training_inputs = [
            "I don't understand machine learning",
            "Can you explain deep learning?",
            "What are artificial neural networks?",
            "Help me understand backpropagation",
            "How do convolutional neural networks work?",
            "What's the difference between AI and ML?",
            "Explain gradient descent algorithm",
            "I'm confused about overfitting in models"
        ]
        
        print(f"\n🎯 Starting Evolution Demo with {len(training_inputs)} training inputs")
        
        # Run evolution cycles
        for cycle in range(evolution_cycles):
            print(f"\n{'🔄 EVOLUTION CYCLE ' + str(cycle + 1):=^80}")
            
            # Process training inputs to drive evolution
            cycle_start_time = time.time()
            
            # Use a mix of training inputs and our target input
            cycle_inputs = training_inputs[:requests_per_cycle-1] + [target_input]
            np.random.shuffle(cycle_inputs)
            
            cycle_results = []
            for i, input_text in enumerate(cycle_inputs):
                print(f"\n📝 Processing request {i+1}/{len(cycle_inputs)}: {input_text[:50]}...")
                
                result = self.system.process_request(
                    input_text=input_text,
                    enable_evolution=True
                )
                cycle_results.append(result)
                
                # If this is our target input, save detailed snapshot
                if input_text == target_input:
                    snapshot = self._create_evolution_snapshot(result, cycle + 1)
                    self.evolution_snapshots.append(snapshot)
                    
                    # Safely extract quality score for display
                    quality_score = 0.0
                    if 'quality_metrics' in result and 'overall_quality' in result['quality_metrics']:
                        quality_score = result['quality_metrics']['overall_quality']
                    elif 'error' not in result:
                        quality_score = snapshot['quality_score']
                    
                    print(f"📊 Target response quality: {quality_score:.3f}")
            
            cycle_time = time.time() - cycle_start_time
            
            # Cycle summary
            cycle_qualities = []
            for r in cycle_results:
                if 'quality_metrics' in r and 'overall_quality' in r['quality_metrics']:
                    cycle_qualities.append(r['quality_metrics']['overall_quality'])
                else:
                    cycle_qualities.append(0.0)  # Default for error cases
            avg_quality = np.mean(cycle_qualities) if cycle_qualities else 0.0
            
            print(f"\n✅ Cycle {cycle + 1} completed in {cycle_time:.2f}s")
            print(f"   Average Quality: {avg_quality:.3f}")
            print(f"   Evolution Step: {self.system.evolution_step}")
            
            self.demo_results.extend(cycle_results)
        
        # Generate comprehensive analysis
        self._generate_evolution_analysis()
        
        # Save results
        self._save_demo_results()
        
        print(f"\n🎉 Evolution Demo Completed!")
        print(f"Total Requests: {len(self.demo_results)}")
        print(f"Evolution Cycles: {evolution_cycles}")
        print(f"Final Evolution Step: {self.system.evolution_step}")
    
    def _create_evolution_snapshot(self, result: Dict[str, Any], cycle: int) -> Dict[str, Any]:
        """Create a detailed snapshot of evolution progress"""
        # Handle cases where result might have errors or missing data
        if 'error' in result:
            return {
                'cycle': cycle,
                'evolution_step': self.system.evolution_step,
                'timestamp': result.get('timestamp', ''),
                'input_text': result.get('input_text', ''),
                'response': f"Error: {result['error']}",
                'response_time': 0.0,
                'quality_score': 0.0,
                'quality_dimensions': {},
                'system_prompt_used': '',
                'evolution_stage': 'error',
                'processing_time': result.get('processing_time', 0.0)
            }
        
        # Extract llm_response data safely
        llm_response = result.get('llm_response', {})
        quality_metrics = result.get('quality_metrics', {})
        
        return {
            'cycle': cycle,
            'evolution_step': self.system.evolution_step,
            'timestamp': result.get('timestamp', ''),
            'input_text': result.get('input_text', ''),
            'response': llm_response.get('response', 'No response available'),
            'response_time': llm_response.get('response_time', 0.0),
            'quality_score': quality_metrics.get('overall_quality', 0.0),
            'quality_dimensions': quality_metrics.get('quality_dimensions', {}),
            'system_prompt_used': llm_response.get('system_prompt', ''),
            'evolution_stage': llm_response.get('evolution_stage', 'basic'),
            'processing_time': result.get('processing_time', 0.0)
        }
    
    def _generate_evolution_analysis(self):
        """Generate comprehensive evolution analysis"""
        print(f"\n{'📈 EVOLUTION ANALYSIS':=^80}")
        
        if not self.evolution_snapshots:
            print("❌ No evolution snapshots available for analysis")
            return
        
        # Quality progression analysis
        print("\n🎯 Quality Progression Analysis:")
        print("-" * 50)
        
        for i, snapshot in enumerate(self.evolution_snapshots):
            cycle = snapshot['cycle']
            quality = snapshot['quality_score']
            stage = snapshot['evolution_stage']
            
            print(f"Cycle {cycle:2d} | Quality: {quality:.3f} | Stage: {stage:15s} | Step: {snapshot['evolution_step']:2d}")
        
        # Calculate improvement metrics
        if len(self.evolution_snapshots) >= 2:
            initial_quality = self.evolution_snapshots[0]['quality_score']
            final_quality = self.evolution_snapshots[-1]['quality_score']
            improvement = final_quality - initial_quality
            improvement_pct = (improvement / initial_quality) * 100 if initial_quality > 0 else 0
            
            print(f"\n📊 Quality Improvement:")
            print(f"   Initial Quality: {initial_quality:.3f}")
            print(f"   Final Quality:   {final_quality:.3f}")
            print(f"   Improvement:     {improvement:+.3f} ({improvement_pct:+.1f}%)")
        
        # Response evolution comparison
        print(f"\n🔍 Response Evolution Comparison:")
        print("-" * 50)
        
        if len(self.evolution_snapshots) >= 2:
            print("INITIAL RESPONSE (Cycle 1):")
            print(f"Quality: {self.evolution_snapshots[0]['quality_score']:.3f}")
            print(f"Stage: {self.evolution_snapshots[0]['evolution_stage']}")
            print(f"Response: {self.evolution_snapshots[0]['response'][:200]}...")
            
            print(f"\nFINAL RESPONSE (Cycle {self.evolution_snapshots[-1]['cycle']}):")
            print(f"Quality: {self.evolution_snapshots[-1]['quality_score']:.3f}")
            print(f"Stage: {self.evolution_snapshots[-1]['evolution_stage']}")
            print(f"Response: {self.evolution_snapshots[-1]['response'][:200]}...")
        
        # Quality dimensions analysis
        print(f"\n📋 Quality Dimensions Evolution:")
        print("-" * 50)
        
        if self.evolution_snapshots:
            dimensions = ['relevance', 'helpfulness', 'clarity', 'completeness', 'engagement']
            
            # Check if quality_dimensions exist in snapshots
            initial_snapshot = self.evolution_snapshots[0]
            final_snapshot = self.evolution_snapshots[-1]
            
            # Get available dimensions from both snapshots
            initial_dims = initial_snapshot.get('quality_dimensions', {})
            final_dims = final_snapshot.get('quality_dimensions', {})
            
            if not initial_dims and not final_dims:
                print("   No quality dimensions data available for analysis")
            else:
                print(f"{'Dimension':<12} | {'Initial':<7} | {'Final':<7} | {'Change':<8}")
                print("-" * 45)
                
                # Only analyze dimensions that exist in both snapshots
                available_dimensions = []
                for dim in dimensions:
                    if dim in initial_dims and dim in final_dims:
                        available_dimensions.append(dim)
                
                if available_dimensions:
                    for dim in available_dimensions:
                        try:
                            initial = float(initial_dims[dim])
                            final = float(final_dims[dim])
                            change = final - initial
                            
                            print(f"{dim.capitalize():<12} | {initial:<7.3f} | {final:<7.3f} | {change:+7.3f}")
                        except (ValueError, TypeError):
                            print(f"{dim.capitalize():<12} | {'N/A':<7} | {'N/A':<7} | {'N/A':<8}")
                else:
                    print("   No matching quality dimensions found between initial and final snapshots")
                    
                    # Show what dimensions are available in each snapshot
                    if initial_dims:
                        print(f"   Initial snapshot dimensions: {list(initial_dims.keys())}")
                    if final_dims:
                        print(f"   Final snapshot dimensions: {list(final_dims.keys())}")
        
        # System performance analysis
        print(f"\n⚡ System Performance Analysis:")
        print("-" * 50)
        
        response_times = [s['response_time'] for s in self.evolution_snapshots]
        processing_times = [s['processing_time'] for s in self.evolution_snapshots]
        
        print(f"Average Response Time: {np.mean(response_times):.3f}s")
        print(f"Average Processing Time: {np.mean(processing_times):.3f}s")
        print(f"Response Time Range: {np.min(response_times):.3f}s - {np.max(response_times):.3f}s")
        
        # Get system report
        if self.system:
            system_report = self.system.get_system_report()
            print(f"\n🏆 Final System Status:")
            print(f"   Total Requests: {system_report['system_info']['total_requests']}")
            print(f"   Evolution Cycles: {system_report['performance_metrics']['evolution_cycles']}")
            print(f"   Quality Improvement: {system_report['performance_metrics']['quality_improvement']:.3f}")
    
    def _save_demo_results(self):
        """Save demo results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save evolution snapshots
        snapshots_file = f"llm_evolution_snapshots_{timestamp}.json"
        with open(snapshots_file, 'w') as f:
            json.dump(self.evolution_snapshots, f, indent=2, default=str)
        print(f"📁 Evolution snapshots saved to: {snapshots_file}")
        
        # Save complete demo results
        demo_file = f"llm_evolution_demo_{timestamp}.json"
        demo_data = {
            'demo_info': {
                'timestamp': timestamp,
                'total_requests': len(self.demo_results),
                'evolution_cycles': len(self.evolution_snapshots),
                'target_input': self.evolution_snapshots[0]['input_text'] if self.evolution_snapshots else None
            },
            'evolution_snapshots': self.evolution_snapshots,
            'all_results': self.demo_results
        }
        
        with open(demo_file, 'w') as f:
            json.dump(demo_data, f, indent=2, default=str)
        print(f"📁 Complete demo results saved to: {demo_file}")
        
        # Save system state
        if self.system:
            self.system.save_system_state()
    
    def compare_responses(self, input_text: str, num_comparisons: int = 3):
        """
        Compare responses at different evolution stages
        
        Args:
            input_text: Input to generate responses for
            num_comparisons: Number of evolution stages to compare
        """
        print(f"\n{'🔬 RESPONSE COMPARISON ANALYSIS':=^80}")
        print(f"Input: '{input_text}'")
        print(f"Comparing {num_comparisons} evolution stages")
        print("=" * 80)
        
        if not self.system:
            self.system = IntegratedLLMSystem()
        
        comparisons = []
        
        # Generate responses at different evolution stages
        for stage in range(num_comparisons):
            # Set evolution step
            evolution_step = stage * 10  # Simulate different evolution stages
            self.system.llm_agent.set_evolution_step(evolution_step)
            
            print(f"\n🎭 Generating response at Evolution Step {evolution_step}...")
            
            result = self.system.process_request(
                input_text=input_text,
                enable_evolution=False  # Don't evolve during comparison
            )
            
            # Handle cases where response generation fails
            if 'llm_response' not in result or result['llm_response'] is None:
                print(f"   ⚠️  Response generation failed at step {evolution_step}")
                comparison = {
                    'evolution_step': evolution_step,
                    'response': f"Error: Response generation failed",
                    'quality_score': 0.0,
                    'quality_dimensions': {},
                    'evolution_stage': 'error',
                    'response_time': 0.0
                }
            else:
                comparison = {
                    'evolution_step': evolution_step,
                    'response': result['llm_response'].get('response', 'Error: No response content'),
                    'quality_score': result.get('quality_metrics', {}).get('overall_quality', 0.0),
                    'quality_dimensions': result.get('quality_metrics', {}).get('quality_dimensions', {}),
                    'evolution_stage': result['llm_response'].get('evolution_stage', 'basic'),
                    'response_time': result['llm_response'].get('response_time', 0.0)
                }
            
            comparisons.append(comparison)
            
            print(f"   Quality Score: {comparison['quality_score']:.3f}")
            print(f"   Evolution Stage: {comparison['evolution_stage']}")
            print(f"   Response: {comparison['response'][:100]}...")
        
        # Display comparison results
        print(f"\n{'📊 COMPARISON RESULTS':=^80}")
        
        for i, comp in enumerate(comparisons):
            print(f"\n🎯 EVOLUTION STEP {comp['evolution_step']} ({comp['evolution_stage'].upper()}):")
            print(f"Quality Score: {comp['quality_score']:.3f}")
            print(f"Response Time: {comp['response_time']:.3f}s")
            print("Quality Dimensions:")
            for dim, score in comp['quality_dimensions'].items():
                print(f"  {dim.capitalize()}: {score:.3f}")
            print(f"Response: {comp['response']}")
            print("-" * 80)
        
        return comparisons

def main():
    """Run the LLM Evolution Demo"""
    print("🌟 Welcome to the LLM Evolution Demonstration!")
    print("This demo shows how the LLM agent evolves and improves over time.")
    print()
    
    # Create demo instance
    demo = LLMEvolutionDemo()
    
    # Run evolution demo with the specified example
    demo.run_evolution_demo(
        target_input="I do not understand neural networks",
        evolution_cycles=6,
        requests_per_cycle=4
    )
    
    print("\n" + "=" * 80)
    print("🎯 BONUS: Response Comparison at Different Evolution Stages")
    print("=" * 80)
    
    # Compare responses at different stages
    demo.compare_responses(
        input_text="I do not understand neural networks",
        num_comparisons=3
    )
    
    print("\n🎉 Demo completed successfully!")
    print("Check the generated JSON files for detailed results and analysis.")

if __name__ == "__main__":
    main()