#!/usr/bin/env python3
"""
Test script to validate the improved LLM evolution system.
This script tests that the enhanced system properly improves quality dimensions,
especially Relevance, Helpfulness, and Clarity.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import openai
import time

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_evolution_system import LLMEvolutionSystem
from llm_metrics_framework import LLMMetricsFramework

class MockLLMAgent:
    """Mock LLM agent for testing purposes."""
    
    def __init__(self):
        self.parameters = {
            'temperature': 0.7,
            'max_tokens': 150,
            'top_p': 0.9,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'system_prompt_style': 2,
            'response_format': 1,
            'explanation_depth': 0.5,
            'technical_level': 0.5,
            'engagement_factor': 0.7
        }
    
    def update_parameters(self, new_params):
        """Update agent parameters."""
        self.parameters.update(new_params)
    
    def generate_response(self, input_text, urgency='medium', category='general'):
        """Generate a mock response with quality that varies based on parameters."""
        # Extract key parameters that affect quality
        temp = self.parameters.get('temperature', 0.7)
        depth = self.parameters.get('explanation_depth', 0.5)
        tech_level = self.parameters.get('technical_level', 0.5)
        engagement = self.parameters.get('engagement_factor', 0.7)
        top_p = self.parameters.get('top_p', 0.9)
        
        # Calculate quality factors based on parameters
        # Optimal ranges for different aspects:
        # - Relevance: temperature 0.3-0.7, top_p 0.8-0.95
        # - Helpfulness: explanation_depth 0.6-0.9, technical_level matched to content
        # - Clarity: temperature 0.2-0.6, explanation_depth 0.4-0.8
        
        relevance_factor = 1.0
        helpfulness_factor = 1.0
        clarity_factor = 1.0
        
        # Temperature effects
        if 0.3 <= temp <= 0.7:
            relevance_factor *= 1.2
            clarity_factor *= 1.1
        elif temp > 0.9:
            relevance_factor *= 0.7
            clarity_factor *= 0.6
        
        # Top_p effects
        if 0.8 <= top_p <= 0.95:
            relevance_factor *= 1.1
        elif top_p < 0.7:
            relevance_factor *= 0.8
        
        # Explanation depth effects
        if depth >= 0.6:
            helpfulness_factor *= 1.3
            clarity_factor *= 1.1
        elif depth < 0.3:
            helpfulness_factor *= 0.6
            clarity_factor *= 0.7
        
        # Technical level matching
        if category == 'technical' and tech_level >= 0.7:
            helpfulness_factor *= 1.2
            relevance_factor *= 1.1
        elif category == 'educational' and 0.3 <= tech_level <= 0.6:
            clarity_factor *= 1.2
            helpfulness_factor *= 1.1
        
        # Generate response content based on quality factors
        base_quality = {
            'relevance': min(0.9, 0.4 * relevance_factor),
            'helpfulness': min(0.9, 0.3 * helpfulness_factor),
            'clarity': min(0.9, 0.4 * clarity_factor),
            'completeness': min(0.9, 0.5 * (depth + tech_level)),
            'engagement': min(0.9, 0.4 * engagement)
        }
        
        # Create response with varying quality
        if base_quality['relevance'] > 0.6 and base_quality['helpfulness'] > 0.5:
            response = f"Here's a comprehensive answer to '{input_text[:30]}...': [Detailed technical explanation with step-by-step guidance, relevant examples, and practical implementation details. Quality factors: R={relevance_factor:.2f}, H={helpfulness_factor:.2f}, C={clarity_factor:.2f}]"
        elif base_quality['clarity'] > 0.5:
            response = f"Regarding '{input_text[:30]}...': [Clear explanation with good structure and examples. Some technical details included. Quality factors: R={relevance_factor:.2f}, H={helpfulness_factor:.2f}, C={clarity_factor:.2f}]"
        else:
            response = f"About '{input_text[:30]}...': [Basic response with limited detail and relevance. Quality factors: R={relevance_factor:.2f}, H={helpfulness_factor:.2f}, C={clarity_factor:.2f}]"
        
        # Store quality factors for debugging
        self.last_quality_factors = base_quality
        
        return response

class RealLLMAgent:
    """Real LLM agent using OpenAI API for testing purposes."""
    
    def __init__(self):
        self.parameters = {
            'temperature': 0.7,
            'max_tokens': 150,
            'top_p': 0.9,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'system_prompt_style': 2,
            'response_format': 1,
            'explanation_depth': 0.5,
            'technical_level': 0.5,
            'engagement_factor': 0.7
        }
        
        # Initialize OpenAI client
        self.client = openai.OpenAI()
    
    def update_parameters(self, new_params):
        """Update agent parameters."""
        self.parameters.update(new_params)
    
    def _create_system_prompt(self, category, urgency):
        """Create system prompt based on parameters."""
        style = self.parameters.get('system_prompt_style', 2)
        depth = self.parameters.get('explanation_depth', 0.5)
        tech_level = self.parameters.get('technical_level', 0.5)
        engagement = self.parameters.get('engagement_factor', 0.7)
        
        # Base prompts for different styles
        base_prompts = [
            "You are a helpful assistant. Provide clear and accurate responses.",
            "You are an expert assistant. Give detailed and professional answers.",
            "You are a knowledgeable assistant. Balance clarity with thoroughness.",
            "You are a friendly expert. Make complex topics accessible and engaging.",
            "You are a comprehensive assistant. Provide complete and well-structured responses."
        ]
        
        base_prompt = base_prompts[min(style, len(base_prompts) - 1)]
        
        # Add depth instructions
        if depth >= 0.7:
            base_prompt += " Provide detailed explanations with examples and context."
        elif depth <= 0.3:
            base_prompt += " Keep responses concise and to the point."
        else:
            base_prompt += " Provide balanced explanations with key details."
        
        # Add technical level instructions
        if tech_level >= 0.7:
            base_prompt += " Use technical terminology and advanced concepts when appropriate."
        elif tech_level <= 0.3:
            base_prompt += " Use simple language and avoid technical jargon."
        else:
            base_prompt += " Balance technical accuracy with accessibility."
        
        # Add engagement instructions
        if engagement >= 0.7:
            base_prompt += " Make your response engaging and interactive."
        elif engagement <= 0.3:
            base_prompt += " Focus on factual information delivery."
        else:
            base_prompt += " Maintain a professional yet approachable tone."
        
        # Add category-specific instructions
        if category == 'technical':
            base_prompt += " Focus on practical implementation and best practices."
        elif category == 'educational':
            base_prompt += " Structure your response for learning and understanding."
        elif category == 'debugging':
            base_prompt += " Provide step-by-step troubleshooting guidance."
        
        # Add urgency-specific instructions
        if urgency == 'high':
            base_prompt += " Prioritize immediate actionable solutions."
        elif urgency == 'low':
            base_prompt += " Provide comprehensive background and context."
        
        return base_prompt
    
    def generate_response(self, input_text, urgency='medium', category='general'):
        """Generate a real response using OpenAI API."""
        try:
            system_prompt = self._create_system_prompt(category, urgency)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                temperature=self.parameters.get('temperature', 0.7),
                max_tokens=int(self.parameters.get('max_tokens', 150)),  # Ensure integer
                top_p=self.parameters.get('top_p', 0.9),
                frequency_penalty=self.parameters.get('frequency_penalty', 0.0),
                presence_penalty=self.parameters.get('presence_penalty', 0.0)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"⚠️ API call failed: {e}")
            # Fallback to mock response
            return f"[API Error - Mock Response] Regarding '{input_text[:30]}...': This is a fallback response due to API issues."

def run_evolution_test():
    """Run a comprehensive test of the evolution system."""
    print("🧪 Starting Evolution System Test with Real OpenAI API")
    print("=" * 50)
    
    # Initialize systems
    real_agent = RealLLMAgent()
    evolution_system = LLMEvolutionSystem(real_agent)  # Pass agent as required parameter
    metrics_framework = LLMMetricsFramework()
    
    # Test scenarios with expected quality patterns
    test_scenarios = [
        {
            'input': 'How do I implement a binary search algorithm in Python?',
            'urgency': 'medium',
            'category': 'technical',
            'expected_high_relevance': True,
            'expected_high_helpfulness': True
        },
        {
            'input': 'What are the best practices for database optimization?',
            'urgency': 'high',
            'category': 'technical',
            'expected_high_relevance': True,
            'expected_high_helpfulness': True
        },
        {
            'input': 'Explain machine learning concepts for beginners',
            'urgency': 'low',
            'category': 'educational',
            'expected_high_clarity': True,
            'expected_high_engagement': True
        },
        {
            'input': 'Debug this error: TypeError in line 42',
            'urgency': 'high',
            'category': 'debugging',
            'expected_high_helpfulness': True,
            'expected_high_relevance': True
        }
    ]
    
    # Track quality evolution over generations
    generation_metrics = []
    
    print("📊 Running evolution cycles...")
    
    for generation in range(5):  # Run 5 generations
        print(f"\n🔄 Generation {generation + 1}")
        
        # Get current best parameters
        if evolution_system.population:
            best_params = evolution_system.get_best_parameters()
            real_agent.update_parameters(best_params)
            print(f"   🎛️ Current parameters: temp={best_params.get('temperature', 0.7):.2f}, "
                  f"depth={best_params.get('explanation_depth', 0.5):.2f}, "
                  f"tech={best_params.get('technical_level', 0.5):.2f}")
        
        generation_qualities = {
            'relevance': [],
            'helpfulness': [],
            'clarity': [],
            'completeness': [],
            'engagement': []
        }
        
        # Test each scenario with current parameters
        for i, scenario in enumerate(test_scenarios):
            print(f"   📝 Testing scenario {i+1}/4...")
            
            # Add delay to respect API rate limits
            if i > 0:
                time.sleep(1)
            
            response = real_agent.generate_response(
                scenario['input'], 
                scenario['urgency'], 
                scenario['category']
            )
            
            print(f"   📤 Response: {response[:100]}...")
            
            # Evaluate response quality
            quality_scores = metrics_framework.evaluate_response_quality(
                input_text=scenario['input'],
                response=response,
                urgency_level=scenario['urgency'],
                content_category=scenario['category'],
                evolution_step=generation
            )
            
            # Store individual dimension scores
            for dim in generation_qualities:
                generation_qualities[dim].append(quality_scores['quality_dimensions'][dim])
            
            # Add experience to evolution system
            evolution_system.add_experience(
                parameters=real_agent.parameters.copy(),
                quality_scores=quality_scores,
                response_time=0.5,  # Mock response time
                user_feedback="good" if quality_scores['overall_quality'] > 0.7 else None
            )
        
        # Calculate average quality for this generation
        avg_qualities = {dim: np.mean(scores) for dim, scores in generation_qualities.items()}
        avg_qualities['generation'] = generation + 1
        generation_metrics.append(avg_qualities)
        
        # Print generation summary
        print(f"   📈 Avg Relevance: {avg_qualities['relevance']:.3f}")
        print(f"   🤝 Avg Helpfulness: {avg_qualities['helpfulness']:.3f}")
        print(f"   💡 Avg Clarity: {avg_qualities['clarity']:.3f}")
        print(f"   ✅ Avg Completeness: {avg_qualities['completeness']:.3f}")
        print(f"   🎯 Avg Engagement: {avg_qualities['engagement']:.3f}")
        
        # Evolve parameters for next generation
        if generation < 4:  # Don't evolve after last generation
            print("   🧬 Evolving parameters...")
            evolution_system.evolve_parameters()
    
    return generation_metrics

def analyze_results(generation_metrics):
    """Analyze and visualize the evolution results."""
    print("\n📊 EVOLUTION RESULTS ANALYSIS")
    print("=" * 50)
    
    # Calculate improvements
    initial = generation_metrics[0]
    final = generation_metrics[-1]
    
    dimensions = ['relevance', 'helpfulness', 'clarity', 'completeness', 'engagement']
    improvements = {}
    
    print("\n🎯 Quality Dimension Evolution:")
    print("-" * 50)
    print(f"{'Dimension':<12} | {'Initial':<7} | {'Final':<7} | {'Change':<8}")
    print("-" * 50)
    
    for dim in dimensions:
        initial_val = initial[dim]
        final_val = final[dim]
        change = final_val - initial_val
        improvements[dim] = change
        
        change_str = f"{change:+.3f}"
        print(f"{dim.capitalize():<12} | {initial_val:<7.3f} | {final_val:<7.3f} | {change_str:<8}")
    
    # Check if critical dimensions improved
    critical_dims = ['relevance', 'helpfulness', 'clarity']
    critical_improved = all(improvements[dim] > 0 for dim in critical_dims)
    
    print(f"\n🎉 Critical Dimensions (Relevance, Helpfulness, Clarity) Improved: {'✅ YES' if critical_improved else '❌ NO'}")
    
    # Overall assessment
    total_improvement = sum(improvements.values())
    positive_dims = sum(1 for change in improvements.values() if change > 0)
    
    print(f"📈 Total Quality Improvement: {total_improvement:+.3f}")
    print(f"✅ Dimensions Improved: {positive_dims}/{len(dimensions)}")
    
    # Create visualization
    create_evolution_plot(generation_metrics)
    
    return improvements, critical_improved

def create_evolution_plot(generation_metrics):
    """Create a plot showing quality evolution over generations."""
    generations = [m['generation'] for m in generation_metrics]
    dimensions = ['relevance', 'helpfulness', 'clarity', 'completeness', 'engagement']
    
    plt.figure(figsize=(12, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, dim in enumerate(dimensions):
        values = [m[dim] for m in generation_metrics]
        plt.plot(generations, values, marker='o', linewidth=2, 
                label=dim.capitalize(), color=colors[i])
    
    plt.title('LLM Quality Dimensions Evolution Over Generations', fontsize=16, fontweight='bold')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Quality Score', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('evolution_results.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Evolution plot saved as 'evolution_results.png'")
    plt.close()

def main():
    """Main test function."""
    print("🚀 LLM Evolution System Improvement Test")
    print("Testing enhanced reward calculation and evolution strategy")
    print("=" * 60)
    
    try:
        # Run evolution test
        generation_metrics = run_evolution_test()
        
        # Analyze results
        improvements, critical_improved = analyze_results(generation_metrics)
        
        # Final assessment
        print("\n" + "=" * 60)
        print("🏆 FINAL ASSESSMENT")
        print("=" * 60)
        
        if critical_improved:
            print("✅ SUCCESS: All critical dimensions (Relevance, Helpfulness, Clarity) improved!")
            print("🎯 The enhanced evolution system is working correctly.")
        else:
            print("❌ NEEDS IMPROVEMENT: Some critical dimensions did not improve.")
            print("🔧 Further optimization of the evolution system may be needed.")
        
        # Save detailed results
        results = {
            'generation_metrics': generation_metrics,
            'improvements': improvements,
            'critical_improved': critical_improved,
            'test_timestamp': datetime.now().isoformat()
        }
        
        with open('evolution_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to 'evolution_test_results.json'")
        
        return critical_improved
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)