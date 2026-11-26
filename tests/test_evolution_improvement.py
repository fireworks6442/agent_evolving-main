"""
Test Evolution Improvement System
Demonstrates how the LLM Evolution System starts with poor parameters and evolves to better ones
"""

import numpy as np
import random
import time
from datetime import datetime
from llm_evolution_system import LLMEvolutionSystem
import matplotlib.pyplot as plt
import json

class MockLLMAgent:
    """Mock LLM Agent for testing evolution"""
    
    def __init__(self):
        self.evolution_params = {
            'temperature': 1.1,           # Start with poor parameters
            'explanation_depth': 0.15,
            'technical_level': 0.2,
            'top_p': 0.65,
            'max_tokens': 80,
            'frequency_penalty': 0.1,
            'presence_penalty': 0.1,
            'engagement_factor': 0.3,
            'system_prompt_style': 0,
            'response_format': 0
        }
        self.evolution_step = 0
    
    def update_evolution_params(self, params):
        """Update evolution parameters"""
        self.evolution_params.update(params)
    
    def set_evolution_step(self, step):
        """Set evolution step"""
        self.evolution_step = step
    
    def simulate_response_quality(self, params):
        """Simulate response quality based on parameters"""
        # Quality heuristics based on parameter values
        temp = params.get('temperature', 0.7)
        depth = params.get('explanation_depth', 0.5)
        tech = params.get('technical_level', 0.5)
        top_p = params.get('top_p', 0.9)
        tokens = params.get('max_tokens', 400)
        engagement = params.get('engagement_factor', 0.6)
        
        # Calculate quality dimensions based on parameters
        # Relevance: better with moderate temperature and high top_p
        relevance = 0.3
        if 0.3 <= temp <= 0.7:
            relevance += 0.3
        if top_p >= 0.85:
            relevance += 0.2
        if temp < 0.5:
            relevance += 0.1
        
        # Helpfulness: better with higher explanation depth and moderate technical level
        helpfulness = 0.2
        if depth >= 0.6:
            helpfulness += 0.4
        if 0.4 <= tech <= 0.7:
            helpfulness += 0.2
        if tokens >= 300:
            helpfulness += 0.1
        
        # Clarity: better with lower temperature and moderate depth
        clarity = 0.25
        if temp <= 0.6:
            clarity += 0.3
        if 0.4 <= depth <= 0.7:
            clarity += 0.2
        if tech <= 0.6:
            clarity += 0.15
        
        # Completeness: better with more tokens and higher depth
        completeness = 0.4
        if tokens >= 400:
            completeness += 0.3
        if depth >= 0.5:
            completeness += 0.2
        
        # Engagement: based on engagement factor and moderate parameters
        engagement_score = 0.3
        if engagement >= 0.6:
            engagement_score += 0.3
        if 0.4 <= temp <= 0.8:
            engagement_score += 0.2
        
        # Add some randomness to simulate real-world variation
        noise_factor = 0.1
        relevance += random.uniform(-noise_factor, noise_factor)
        helpfulness += random.uniform(-noise_factor, noise_factor)
        clarity += random.uniform(-noise_factor, noise_factor)
        completeness += random.uniform(-noise_factor, noise_factor)
        engagement_score += random.uniform(-noise_factor, noise_factor)
        
        # Ensure bounds [0, 1]
        return {
            'relevance': max(0.0, min(1.0, relevance)),
            'helpfulness': max(0.0, min(1.0, helpfulness)),
            'clarity': max(0.0, min(1.0, clarity)),
            'completeness': max(0.0, min(1.0, completeness)),
            'engagement': max(0.0, min(1.0, engagement_score))
        }

def run_evolution_improvement_test():
    """Run the evolution improvement test"""
    print("=" * 80)
    print("🧬 LLM EVOLUTION IMPROVEMENT TEST")
    print("=" * 80)
    print("Starting with DELIBERATELY POOR parameters and evolving to better ones...")
    print()
    
    # Initialize mock agent and evolution system
    agent = MockLLMAgent()
    evolution_system = LLMEvolutionSystem(agent, population_size=8, mutation_rate=0.6)
    
    # Track evolution progress
    generation_data = []
    parameter_history = []
    quality_history = []
    
    # Test queries that will be used for evaluation
    test_queries = [
        "I don't understand neural networks",
        "Explain machine learning algorithms",
        "What is deep learning?",
        "How do neural networks work?",
        "Can you help me understand AI?",
        "What are the basics of machine learning?",
        "Explain artificial intelligence concepts",
        "How does backpropagation work?"
    ]
    
    print("📊 INITIAL POPULATION ANALYSIS")
    print("-" * 50)
    
    # Analyze initial population
    initial_params = evolution_system.get_best_parameters()
    print(f"Initial Best Parameters:")
    for param, value in initial_params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.3f}")
        else:
            print(f"  {param}: {value}")
    
    # Test initial quality
    initial_quality = agent.simulate_response_quality(initial_params)
    print(f"\nInitial Quality Scores:")
    for dim, score in initial_quality.items():
        print(f"  {dim}: {score:.3f}")
    
    overall_initial = np.mean(list(initial_quality.values()))
    print(f"  Overall Quality: {overall_initial:.3f}")
    print()
    
    # Run evolution for multiple generations
    num_generations = 10
    experiences_per_generation = 5
    
    print("🔄 EVOLUTION PROCESS")
    print("-" * 50)
    
    for generation in range(num_generations):
        print(f"\n🧬 Generation {generation + 1}/{num_generations}")
        
        # Get current best parameters
        current_best = evolution_system.get_best_parameters()
        
        # Simulate experiences with current parameters
        generation_qualities = []
        for i in range(experiences_per_generation):
            query = random.choice(test_queries)
            
            # Simulate response quality
            quality_scores = agent.simulate_response_quality(current_best)
            generation_qualities.append(quality_scores)
            
            # Add experience to evolution system
            evolution_system.add_experience(current_best, quality_scores)
        
        # Calculate average quality for this generation
        avg_quality = {}
        for dim in ['relevance', 'helpfulness', 'clarity', 'completeness', 'engagement']:
            scores = [q[dim] for q in generation_qualities]
            avg_quality[dim] = np.mean(scores)
        
        overall_quality = np.mean(list(avg_quality.values()))
        
        # Store data for analysis
        generation_data.append({
            'generation': generation + 1,
            'parameters': current_best.copy(),
            'quality': avg_quality.copy(),
            'overall_quality': overall_quality
        })
        
        # Print generation summary
        print(f"  Parameters: temp={current_best['temperature']:.3f}, "
              f"depth={current_best['explanation_depth']:.3f}, "
              f"tech={current_best['technical_level']:.3f}")
        print(f"  Quality: rel={avg_quality['relevance']:.3f}, "
              f"help={avg_quality['helpfulness']:.3f}, "
              f"clar={avg_quality['clarity']:.3f}")
        print(f"  Overall Quality: {overall_quality:.3f}")
        
        # Evolve parameters for next generation
        if generation < num_generations - 1:
            evolved_params = evolution_system.evolve_parameters()
            agent.update_evolution_params(evolved_params)
    
    print("\n" + "=" * 80)
    print("📈 EVOLUTION RESULTS ANALYSIS")
    print("=" * 80)
    
    # Compare initial vs final
    final_data = generation_data[-1]
    initial_data = generation_data[0]
    
    print("\n🔍 PARAMETER EVOLUTION:")
    print("-" * 40)
    critical_params = ['temperature', 'explanation_depth', 'technical_level', 'top_p']
    
    for param in critical_params:
        initial_val = initial_data['parameters'][param]
        final_val = final_data['parameters'][param]
        change = final_val - initial_val
        change_pct = (change / initial_val) * 100 if initial_val != 0 else 0
        
        print(f"{param:18}: {initial_val:.3f} → {final_val:.3f} "
              f"(Δ{change:+.3f}, {change_pct:+.1f}%)")
    
    print("\n🎯 QUALITY IMPROVEMENT:")
    print("-" * 40)
    
    for dim in ['relevance', 'helpfulness', 'clarity', 'completeness', 'engagement']:
        initial_score = initial_data['quality'][dim]
        final_score = final_data['quality'][dim]
        improvement = final_score - initial_score
        improvement_pct = (improvement / initial_score) * 100 if initial_score != 0 else 0
        
        print(f"{dim:12}: {initial_score:.3f} → {final_score:.3f} "
              f"(+{improvement:.3f}, +{improvement_pct:.1f}%)")
    
    # Overall improvement
    initial_overall = initial_data['overall_quality']
    final_overall = final_data['overall_quality']
    total_improvement = final_overall - initial_overall
    total_improvement_pct = (total_improvement / initial_overall) * 100
    
    print(f"\n🏆 OVERALL IMPROVEMENT:")
    print(f"Initial Quality: {initial_overall:.3f}")
    print(f"Final Quality:   {final_overall:.3f}")
    print(f"Improvement:     +{total_improvement:.3f} (+{total_improvement_pct:.1f}%)")
    
    # Evolution summary
    print(f"\n📊 EVOLUTION SUMMARY:")
    summary = evolution_system.get_evolution_summary()
    print(f"Generations:        {summary['generation']}")
    print(f"Best Fitness:       {summary['best_fitness']:.4f}")
    print(f"Improvement Rate:   {summary['improvement_rate']:.4f}")
    print(f"Convergence Status: {summary['convergence_status']}")
    print(f"Experience Count:   {summary['experience_count']}")
    print(f"Population Diversity: {summary['population_diversity']:.4f}")
    
    # Show parameter changes over generations
    print(f"\n📈 PARAMETER EVOLUTION OVER TIME:")
    print("-" * 50)
    print("Gen | Temp  | Depth | Tech  | Top_P | Overall")
    print("-" * 50)
    
    for data in generation_data[::2]:  # Show every 2nd generation
        gen = data['generation']
        params = data['parameters']
        quality = data['overall_quality']
        print(f"{gen:2d}  | {params['temperature']:.3f} | "
              f"{params['explanation_depth']:.3f} | "
              f"{params['technical_level']:.3f} | "
              f"{params['top_p']:.3f} | {quality:.3f}")
    
    # Demonstrate parameter diversity
    print(f"\n🔄 PARAMETER CHANGES BETWEEN GENERATIONS:")
    print("-" * 50)
    
    for i in range(1, min(6, len(generation_data))):
        prev_params = generation_data[i-1]['parameters']
        curr_params = generation_data[i]['parameters']
        
        changes = []
        for param in critical_params:
            change = abs(curr_params[param] - prev_params[param])
            changes.append(f"{param[:4]}:{change:.3f}")
        
        print(f"Gen {i} → {i+1}: {', '.join(changes)}")
    
    # Success criteria
    print(f"\n✅ SUCCESS CRITERIA CHECK:")
    print("-" * 30)
    
    # Check if parameters actually changed
    param_changes = []
    for param in critical_params:
        change = abs(final_data['parameters'][param] - initial_data['parameters'][param])
        param_changes.append(change)
    
    avg_param_change = np.mean(param_changes)
    print(f"Average Parameter Change: {avg_param_change:.3f} {'✅' if avg_param_change > 0.1 else '❌'}")
    
    # Check if quality improved
    quality_improved = total_improvement > 0.05
    print(f"Quality Improvement > 5%: {total_improvement_pct:.1f}% {'✅' if quality_improved else '❌'}")
    
    # Check if evolution is working
    evolution_working = summary['improvement_rate'] > 0
    print(f"Positive Evolution Rate: {summary['improvement_rate']:.4f} {'✅' if evolution_working else '❌'}")
    
    # Final verdict
    success = avg_param_change > 0.1 and quality_improved and evolution_working
    print(f"\n🎯 EVOLUTION TEST RESULT: {'SUCCESS ✅' if success else 'NEEDS IMPROVEMENT ❌'}")
    
    if success:
        print("The evolution system successfully:")
        print("  • Started with poor parameters")
        print("  • Evolved parameters across generations")
        print("  • Achieved significant quality improvements")
        print("  • Demonstrated clear parameter diversity")
    else:
        print("The evolution system needs improvement in:")
        if avg_param_change <= 0.1:
            print("  • Parameter evolution (parameters not changing enough)")
        if not quality_improved:
            print("  • Quality improvement (less than 5% improvement)")
        if not evolution_working:
            print("  • Evolution mechanism (negative or zero improvement rate)")
    
    return generation_data, success

def create_evolution_visualization(generation_data):
    """Create visualization of evolution progress"""
    try:
        import matplotlib.pyplot as plt
        
        generations = [d['generation'] for d in generation_data]
        overall_quality = [d['overall_quality'] for d in generation_data]
        
        # Quality dimensions
        relevance = [d['quality']['relevance'] for d in generation_data]
        helpfulness = [d['quality']['helpfulness'] for d in generation_data]
        clarity = [d['quality']['clarity'] for d in generation_data]
        
        # Parameters
        temperature = [d['parameters']['temperature'] for d in generation_data]
        depth = [d['parameters']['explanation_depth'] for d in generation_data]
        tech = [d['parameters']['technical_level'] for d in generation_data]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall quality evolution
        ax1.plot(generations, overall_quality, 'b-o', linewidth=2, markersize=6)
        ax1.set_title('Overall Quality Evolution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Quality Score')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Quality dimensions
        ax2.plot(generations, relevance, 'r-o', label='Relevance', linewidth=2)
        ax2.plot(generations, helpfulness, 'g-o', label='Helpfulness', linewidth=2)
        ax2.plot(generations, clarity, 'b-o', label='Clarity', linewidth=2)
        ax2.set_title('Quality Dimensions Evolution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Quality Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Parameter evolution
        ax3.plot(generations, temperature, 'r-s', label='Temperature', linewidth=2)
        ax3.plot(generations, depth, 'g-^', label='Explanation Depth', linewidth=2)
        ax3.plot(generations, tech, 'b-d', label='Technical Level', linewidth=2)
        ax3.set_title('Parameter Evolution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Parameter Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.2)
        
        # Improvement rate
        if len(overall_quality) > 1:
            improvement_rates = []
            for i in range(1, len(overall_quality)):
                rate = overall_quality[i] - overall_quality[i-1]
                improvement_rates.append(rate)
            
            ax4.bar(generations[1:], improvement_rates, alpha=0.7, 
                   color=['green' if r > 0 else 'red' for r in improvement_rates])
            ax4.set_title('Generation-to-Generation Improvement', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Quality Change')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('evolution_improvement_results.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved as 'evolution_improvement_results.png'")
        
    except ImportError:
        print("\n📊 Matplotlib not available - skipping visualization")

def main():
    """Main function to run the evolution improvement test"""
    print("🚀 Starting Evolution Improvement Test...")
    print()
    
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Run the test
    generation_data, success = run_evolution_improvement_test()
    
    # Create visualization
    create_evolution_visualization(generation_data)
    
    # Save results to JSON
    results = {
        'test_timestamp': datetime.now().isoformat(),
        'success': success,
        'generation_data': generation_data,
        'summary': {
            'total_generations': len(generation_data),
            'initial_quality': generation_data[0]['overall_quality'],
            'final_quality': generation_data[-1]['overall_quality'],
            'improvement': generation_data[-1]['overall_quality'] - generation_data[0]['overall_quality']
        }
    }
    
    with open('evolution_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to 'evolution_test_results.json'")
    print("\n🎉 Evolution Improvement Test Complete!")
    
    return success

if __name__ == "__main__":
    main()