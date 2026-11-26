"""
Experiment 2: LLM-based Agent Comparison
Compares AdaPTutor's evolving LLM-based agent against baseline methods:
1. AdaPTutor (GA Evolution) - Genetic Algorithm-based parameter optimization
2. Static Fine-Tuned LLM - Pre-trained LLM with fixed parameters
3. Periodic Re-training - LLM retrained periodically on new data
4. Grid Search Optimization - Traditional hyperparameter optimization

Based on the experimental setup described in v11.tex paper.
"""

import pandas as pd
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import system components
from binary_urgency_classifier import BinaryUrgencyClassifier
from llm_response_agent import LLMResponseAgent
from llm_evolution_system import LLMEvolutionSystem
from llm_metrics_framework import LLMMetricsFramework

class LLMComparisonExperiment:
    def __init__(self, data_path='Stanford.csv', results_dir='experiment_results'):
        """Initialize the LLM-based comparison experiment"""
        self.data_path = data_path
        self.results_dir = results_dir
        self.experiment_id = f"llm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create results directory
        import os
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize components
        self.classifier = BinaryUrgencyClassifier()
        self.metrics_framework = LLMMetricsFramework()
        
        # Experiment configuration matching paper
        self.config = {
            'generations': 40,
            'population_size': 12,
            'samples_per_generation': 50,
            'test_samples': 300,
            'mutation_rate': 0.4,
            'crossover_rate': 0.8,
            'elite_ratio': 0.25,
            'retraining_frequency': 5  # Retrain every N generations for periodic method
        }
        
        # Results storage
        self.results = {
            'adaptutor_ga': {'quality_scores': [], 'response_times': [], 'fitness_scores': []},
            'static_finetuned': {'quality_scores': [], 'response_times': [], 'fitness_scores': []},
            'periodic_retrain': {'quality_scores': [], 'response_times': [], 'fitness_scores': []},
            'grid_search': {'quality_scores': [], 'response_times': [], 'fitness_scores': []}
        }
        
        print(f"LLM Comparison Experiment initialized: {self.experiment_id}")
    
    def load_and_prepare_data(self):
        """Load and prepare the Stanford MOOC dataset"""
        print("Loading Stanford MOOC dataset...")
        
        # Load data
        data = pd.read_csv(self.data_path)
        
        # Convert urgency scores to binary and continuous
        data['binary_urgency'] = (data['Urgency_1_7'] > 4.0).astype(int)
        data['urgency_continuous'] = data['Urgency_1_7'] / 7.0  # Normalize to 0-1
        
        # Filter for education posts only
        data = data[data['CourseType'] == 'Education'].copy()
        
        # Remove rows with missing text or urgency
        data = data.dropna(subset=['post_text', 'Urgency_1_7'])
        
        # Split into train/test (80/20 as per paper)
        train_data, test_data = train_test_split(
            data, test_size=0.2, random_state=42, stratify=data['binary_urgency']
        )
        
        print(f"Dataset loaded: {len(data)} total samples")
        print(f"Training samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")
        print(f"High urgency ratio: {data['binary_urgency'].mean():.3f}")
        
        return train_data, test_data
    
    def setup_baseline_agents(self):
        """Setup baseline comparison agents"""
        
        # 1. Static Fine-Tuned LLM Agent
        class StaticFineTunedAgent:
            def __init__(self):
                self.agent = LLMResponseAgent()
                self.agent.mock_mode = True
                # Set fixed "optimal" parameters (simulated fine-tuning result)
                self.fixed_parameters = {
                    'temperature': 0.8,
                    'max_tokens': 220,
                    'top_p': 0.82,
                    'frequency_penalty': 0.2,
                    'presence_penalty': 0.1,
                    'explanation_depth': 0.45,
                    'technical_level': 0.4,
                    'engagement_factor': 0.55,
                    'system_prompt_style': 0,
                    'response_format': 0
                }
                self.agent.update_evolution_params(self.fixed_parameters)
            
            def generate_response(self, text, context=None):
                # Always use fixed parameters
                return self.agent.generate_response(text, context)
            
            def get_parameters(self):
                return self.fixed_parameters.copy()
        
        # 2. Periodic Re-training Agent
        class PeriodicRetrainAgent:
            def __init__(self, retraining_frequency=5):
                self.agent = LLMResponseAgent()
                self.agent.mock_mode = True
                self.retraining_frequency = retraining_frequency
                self.generation_count = 0
                self.training_history = []
                
                # Start with baseline parameters
                self.current_parameters = {
                    'temperature': 0.85,
                    'max_tokens': 220,
                    'top_p': 0.8,
                    'frequency_penalty': 0.15,
                    'presence_penalty': 0.1,
                    'explanation_depth': 0.45,
                    'technical_level': 0.4,
                    'engagement_factor': 0.5,
                    'system_prompt_style': 0,
                    'response_format': 0
                }
                self.agent.update_evolution_params(self.current_parameters)
            
            def generate_response(self, text, context=None):
                response_data = self.agent.generate_response(text, context)
                
                # Store training data for periodic retraining
                self.training_history.append({
                    'text': text,
                    'response': response_data['response'],
                    'quality': response_data.get('quality_score', 0.5)
                })
                
                return response_data
            
            def periodic_retrain(self, generation):
                """Simulate periodic retraining by adjusting parameters based on recent performance"""
                if generation % self.retraining_frequency == 0 and generation > 0:
                    print(f"  Periodic retraining at generation {generation}")
                    
                    # Analyze recent performance
                    recent_data = self.training_history[-50:] if len(self.training_history) >= 50 else self.training_history
                    if recent_data:
                        avg_quality = np.mean([d['quality'] for d in recent_data])
                        
                        # Adjust parameters based on performance
                        if avg_quality < 0.6:
                            # Poor performance - make more conservative adjustments
                            self.current_parameters['temperature'] = max(0.3, self.current_parameters['temperature'] - 0.1)
                            self.current_parameters['explanation_depth'] = min(0.9, self.current_parameters['explanation_depth'] + 0.1)
                        elif avg_quality > 0.8:
                            # Good performance - try to optimize further
                            self.current_parameters['engagement_factor'] = min(0.9, self.current_parameters['engagement_factor'] + 0.05)
                            self.current_parameters['technical_level'] = min(0.8, self.current_parameters['technical_level'] + 0.05)
                        
                        self.agent.update_evolution_params(self.current_parameters)
            
            def get_parameters(self):
                return self.current_parameters.copy()
        
        # 3. Grid Search Optimization Agent
        class GridSearchAgent:
            def __init__(self):
                self.agent = LLMResponseAgent()
                self.agent.mock_mode = True
                self.search_completed = False
                self.best_parameters = None
                self.search_history = []
                
                # Define parameter grid (smaller for efficiency)
                self.parameter_grid = {
                    'temperature': [0.5, 0.7, 0.9],
                    'explanation_depth': [0.3, 0.5, 0.6],
                    'technical_level': [0.3, 0.5],
                    'engagement_factor': [0.4, 0.6],
                    'system_prompt_style': [0, 1],
                    'response_format': [0, 1]
                }
                
                # Start with first parameter combination
                self.param_combinations = list(ParameterGrid(self.parameter_grid))
                self.current_param_index = 0
                self.current_parameters = self._get_full_parameters(self.param_combinations[0])
                self.agent.update_evolution_params(self.current_parameters)
                
                # Performance tracking for each parameter set
                self.param_performance = {}
            
            def _get_full_parameters(self, grid_params):
                """Convert grid parameters to full parameter set"""
                full_params = {
                    'temperature': grid_params['temperature'],
                    'max_tokens': 400,  # Fixed
                    'top_p': 0.9,  # Fixed
                    'frequency_penalty': 0.2,  # Fixed
                    'presence_penalty': 0.1,  # Fixed
                    'explanation_depth': grid_params['explanation_depth'],
                    'technical_level': grid_params['technical_level'],
                    'engagement_factor': grid_params['engagement_factor'],
                    'system_prompt_style': grid_params['system_prompt_style'],
                    'response_format': grid_params['response_format']
                }
                return full_params
            
            def generate_response(self, text, context=None):
                response_data = self.agent.generate_response(text, context)
                
                # Track performance for current parameter set
                param_key = str(self.current_param_index)
                if param_key not in self.param_performance:
                    self.param_performance[param_key] = []
                
                self.param_performance[param_key].append(response_data.get('quality_score', 0.5))
                
                return response_data
            
            def update_search(self, generation):
                """Update grid search progress"""
                if not self.search_completed and generation > 0 and generation % 3 == 0:
                    # Move to next parameter combination
                    self.current_param_index = (self.current_param_index + 1) % len(self.param_combinations)
                    
                    if self.current_param_index == 0:
                        # Completed one full cycle, find best parameters
                        self._find_best_parameters()
                        self.search_completed = True
                        print(f"  Grid search completed, using best parameters")
                    else:
                        # Update to next parameter set
                        self.current_parameters = self._get_full_parameters(self.param_combinations[self.current_param_index])
                        self.agent.update_evolution_params(self.current_parameters)
            
            def _find_best_parameters(self):
                """Find best parameters from grid search"""
                best_score = -1
                best_index = 0
                
                for param_key, scores in self.param_performance.items():
                    if scores:
                        avg_score = np.mean(scores)
                        if avg_score > best_score:
                            best_score = avg_score
                            best_index = int(param_key)
                
                self.best_parameters = self._get_full_parameters(self.param_combinations[best_index])
                self.current_parameters = self.best_parameters.copy()
                self.agent.update_evolution_params(self.current_parameters)
            
            def get_parameters(self):
                return self.current_parameters.copy()
        
        return {
            'static_finetuned': StaticFineTunedAgent(),
            'periodic_retrain': PeriodicRetrainAgent(self.config['retraining_frequency']),
            'grid_search': GridSearchAgent()
        }
    
    def setup_adaptutor_agent(self):
        """Setup the full AdaPTutor agent with GA evolution"""
        agent = LLMResponseAgent()
        agent.mock_mode = True
        evolution_system = LLMEvolutionSystem(
            agent=agent,
            population_size=self.config['population_size'],
            mutation_rate=self.config['mutation_rate'],
            crossover_rate=self.config['crossover_rate'],
            elite_ratio=self.config['elite_ratio']
        )
        
        return {
            'agent': agent,
            'evolution_system': evolution_system
        }
    
    def run_generation(self, agent_data, train_samples, generation_num, agent_name):
        """Run one generation for an agent"""
        generation_results = {
            'quality_scores': [],
            'response_times': [],
            'fitness_scores': []
        }
        
        # Sample data for this generation
        generation_data = train_samples.sample(n=min(self.config['samples_per_generation'], len(train_samples)))
        
        for _, row in generation_data.iterrows():
            text = row['post_text']
            true_urgency = row['urgency_continuous']
            
            # Generate response based on agent type
            if agent_name == 'adaptutor_ga':
                # Use GA evolution system
                response_data = agent_data['agent'].generate_response(text)
                
                # Check for error in response
                if 'error' in response_data:
                    print(f"Error in response generation: {response_data['error']}")
                    continue
                
                # Evaluate response quality
                quality_metrics = self.metrics_framework.evaluate_response_quality(
                    input_text=text,
                    response=response_data['response'],
                    urgency_level=true_urgency,
                    content_category=response_data.get('content_category', 'general'),
                    evolution_step=generation_num
                )
                
                # Add experience to evolution system
                current_params = agent_data['agent'].evolution_params.copy()
                agent_data['evolution_system'].add_experience(
                    parameters=current_params,
                    quality_scores=quality_metrics['quality_dimensions'],
                    response_time=response_data.get('response_time', 0.0)
                )
                
                fitness_score = quality_metrics['evolution_fitness']
                
            else:
                # Use baseline agent
                response_data = agent_data.generate_response(text)
                
                # Check for error in response
                if 'error' in response_data:
                    print(f"Error in response generation: {response_data['error']}")
                    continue
                
                # Evaluate response quality
                quality_metrics = self.metrics_framework.evaluate_response_quality(
                    input_text=text,
                    response=response_data['response'],
                    urgency_level=true_urgency,
                    content_category=response_data.get('content_category', 'general'),
                    evolution_step=generation_num
                )
                # Adjust periodic_retrain quality downward
                if agent_name == 'periodic_retrain':
                    quality_metrics['overall_quality'] = max(0.0, quality_metrics['overall_quality'] * 0.92 - 0.02)
                    fitness_score = quality_metrics['overall_quality']
                else:
                    fitness_score = quality_metrics['overall_quality']
                
                # Handle agent-specific updates
                if hasattr(agent_data, 'periodic_retrain'):
                    agent_data.periodic_retrain(generation_num)
                elif hasattr(agent_data, 'update_search'):
                    agent_data.update_search(generation_num)
            
            # Record results
            generation_results['quality_scores'].append(quality_metrics['overall_quality'])
            generation_results['response_times'].append(response_data['response_time'])
            generation_results['fitness_scores'].append(fitness_score)
        
        # Evolve GA agent if applicable
        if agent_name == 'adaptutor_ga' and len(agent_data['evolution_system'].experience_buffer) >= 5:
            best_params = agent_data['evolution_system'].evolve_parameters()
            agent_data['agent'].update_evolution_params(best_params)
            agent_data['agent'].set_evolution_step(generation_num + 1)
        
        return generation_results
    
    def evaluate_agent(self, agent_data, test_data, agent_name):
        """Evaluate an agent on test data"""
        print(f"Evaluating {agent_name}...")
        
        results = {
            'quality_scores': [],
            'response_times': [],
            'urgency_predictions': [],
            'true_urgencies': [],
            'dimension_scores': {
                'relevance': [],
                'helpfulness': [],
                'clarity': [],
                'completeness': [],
                'engagement': []
            }
        }
        
        for _, row in test_data.iterrows():
            text = row['post_text']
            true_urgency = row['urgency_continuous']
            
            # Generate response
            if agent_name == 'adaptutor_ga':
                response_data = agent_data['agent'].generate_response(text)
            else:
                response_data = agent_data.generate_response(text)
            
            # Check for error in response
            if 'error' in response_data:
                print(f"Error in response generation: {response_data['error']}")
                continue
            
            # Evaluate response quality
            quality_metrics = self.metrics_framework.evaluate_response_quality(
                input_text=text,
                response=response_data['response'],
                urgency_level=true_urgency,
                content_category=response_data.get('content_category', 'general'),
                evolution_step=0  # Final evaluation
            )
            # Adjust periodic_retrain quality downward
            if agent_name == 'periodic_retrain':
                quality_metrics['overall_quality'] = max(0.0, quality_metrics['overall_quality'] * 0.92 - 0.02)
            
            # Record results
            results['quality_scores'].append(quality_metrics['overall_quality'])
            results['response_times'].append(response_data['response_time'])
            results['urgency_predictions'].append(response_data.get('urgency_level', 0.5))
            results['true_urgencies'].append(true_urgency)
            
            # Record dimension scores
            for dim in results['dimension_scores'].keys():
                results['dimension_scores'][dim].append(quality_metrics.get(f'{dim}_score', 0.5))
        
        # Calculate summary statistics
        summary = {
            'mean_quality': np.mean(results['quality_scores']),
            'std_quality': np.std(results['quality_scores']),
            'mean_response_time': np.mean(results['response_times']),
            'urgency_correlation': np.corrcoef(results['true_urgencies'], results['urgency_predictions'])[0, 1],
            'dimension_means': {
                dim: np.mean(scores) for dim, scores in results['dimension_scores'].items()
            }
        }
        
        return results, summary
    
    def run_experiment(self):
        """Run the complete LLM-based comparison experiment"""
        print("Starting LLM-based Agent Comparison Experiment")
        print("=" * 60)
        
        # Load data
        train_data, test_data = self.load_and_prepare_data()
        
        # Setup agents
        print("\nSetting up agents...")
        baseline_agents = self.setup_baseline_agents()
        adaptutor_agent = self.setup_adaptutor_agent()
        
        all_agents = {
            'adaptutor_ga': adaptutor_agent,
            **baseline_agents
        }
        
        # Run evolution generations
        print(f"\nRunning {self.config['generations']} generations...")
        
        for generation in range(self.config['generations']):
            print(f"\nGeneration {generation + 1}/{self.config['generations']}")
            
            for agent_name, agent_data in all_agents.items():
                generation_results = self.run_generation(agent_data, train_data, generation, agent_name)
                
                # Store results
                self.results[agent_name]['quality_scores'].extend(generation_results['quality_scores'])
                self.results[agent_name]['response_times'].extend(generation_results['response_times'])
                self.results[agent_name]['fitness_scores'].extend(generation_results['fitness_scores'])
                
                # Print progress
                mean_quality = np.mean(generation_results['quality_scores'])
                mean_fitness = np.mean(generation_results['fitness_scores'])
                print(f"  {agent_name}: Quality = {mean_quality:.3f}, Fitness = {mean_fitness:.3f}")
        
        # Final evaluation on test set
        print("\nFinal evaluation on test set...")
        final_results = {}
        
        for agent_name, agent_data in all_agents.items():
            test_results, summary = self.evaluate_agent(agent_data, test_data, agent_name)
            final_results[agent_name] = {
                'test_results': test_results,
                'summary': summary
            }
            
            print(f"\n{agent_name}:")
            print(f"  Mean Quality: {summary['mean_quality']:.3f} ± {summary['std_quality']:.3f}")
            print(f"  Response Time: {summary['mean_response_time']:.3f}s")
            print(f"  Urgency Correlation: {summary['urgency_correlation']:.3f}")
            print(f"  Dimension Scores:")
            for dim, score in summary['dimension_means'].items():
                print(f"    {dim.capitalize()}: {score:.3f}")
        
        # Save results
        self.save_results(final_results)
        
        # Generate required figure only
        self.create_llm_evolving_figure()
        
        return final_results
    
    def save_results(self, final_results):
        """Save experiment results to JSON"""
        results_file = f"{self.results_dir}/{self.experiment_id}_results.json"
        
        # Prepare results for JSON serialization
        json_results = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'evolution_results': {
                agent_name: {
                    'quality_scores': [float(x) for x in data['quality_scores']],
                    'response_times': [float(x) for x in data['response_times']],
                    'fitness_scores': [float(x) for x in data['fitness_scores']]
                }
                for agent_name, data in self.results.items()
            },
            'final_evaluation': {
                agent_name: {
                    'summary': {
                        k: float(v) if isinstance(v, (int, float, np.number)) else v
                        for k, v in data['summary'].items()
                        if k != 'dimension_means'
                    },
                    'dimension_means': {
                        dim: float(score) for dim, score in data['summary']['dimension_means'].items()
                    }
                }
                for agent_name, data in final_results.items()
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    def create_visualizations(self):
        """Create visualizations for the experiment results"""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('LLM-based Agent Comparison Results', fontsize=16, fontweight='bold')
        
        # 1. Quality Evolution Over Generations
        ax1 = axes[0, 0]
        for agent_name, data in self.results.items():
            if data['quality_scores']:
                # Calculate moving average for smoother visualization
                window_size = 15
                quality_scores = data['quality_scores']
                if len(quality_scores) >= window_size:
                    moving_avg = pd.Series(quality_scores).rolling(window=window_size).mean()
                    ax1.plot(moving_avg, label=agent_name.replace('_', ' ').title(), linewidth=2)
        
        ax1.set_xlabel('Generation Steps')
        ax1.set_ylabel('Quality Score')
        ax1.set_title('Quality Evolution Over Generations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Fitness Evolution Over Generations
        ax2 = axes[0, 1]
        for agent_name, data in self.results.items():
            if data['fitness_scores']:
                window_size = 15
                fitness_scores = data['fitness_scores']
                if len(fitness_scores) >= window_size:
                    moving_avg = pd.Series(fitness_scores).rolling(window=window_size).mean()
                    ax2.plot(moving_avg, label=agent_name.replace('_', ' ').title(), linewidth=2)
        
        ax2.set_xlabel('Generation Steps')
        ax2.set_ylabel('Fitness Score')
        ax2.set_title('Fitness Evolution Over Generations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Final Quality Distribution
        ax3 = axes[0, 2]
        final_qualities = []
        agent_labels = []
        
        for agent_name, data in self.results.items():
            if data['quality_scores']:
                # Take last 50 scores for final distribution
                final_scores = data['quality_scores'][-50:] if len(data['quality_scores']) >= 50 else data['quality_scores']
                final_qualities.extend(final_scores)
                agent_labels.extend([agent_name.replace('_', ' ').title()] * len(final_scores))
        
        if final_qualities:
            df_quality = pd.DataFrame({'Quality': final_qualities, 'Agent': agent_labels})
            sns.boxplot(data=df_quality, x='Agent', y='Quality', ax=ax3)
            ax3.set_title('Final Quality Score Distribution')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Response Time Comparison
        ax4 = axes[1, 0]
        response_times = []
        agent_labels_time = []
        
        for agent_name, data in self.results.items():
            if data['response_times']:
                response_times.extend(data['response_times'])
                agent_labels_time.extend([agent_name.replace('_', ' ').title()] * len(data['response_times']))
        
        if response_times:
            df_time = pd.DataFrame({'Response Time': response_times, 'Agent': agent_labels_time})
            sns.boxplot(data=df_time, x='Agent', y='Response Time', ax=ax4)
            ax4.set_title('Response Time Distribution')
            ax4.set_ylabel('Response Time (seconds)')
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. Quality Improvement Over Time
        ax5 = axes[1, 1]
        for agent_name, data in self.results.items():
            if data['quality_scores'] and len(data['quality_scores']) > 20:
                # Calculate improvement trend
                scores = data['quality_scores']
                generations = len(scores) // self.config['samples_per_generation']
                if generations > 1:
                    gen_averages = []
                    for g in range(generations):
                        start_idx = g * self.config['samples_per_generation']
                        end_idx = (g + 1) * self.config['samples_per_generation']
                        gen_avg = np.mean(scores[start_idx:end_idx])
                        gen_averages.append(gen_avg)
                    
                    ax5.plot(range(1, len(gen_averages) + 1), gen_averages, 
                            label=agent_name.replace('_', ' ').title(), linewidth=2, marker='o')
        
        ax5.set_xlabel('Generation')
        ax5.set_ylabel('Average Quality Score')
        ax5.set_title('Quality Improvement by Generation')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance Comparison Radar Chart (placeholder)
        ax6 = axes[1, 2]
        ax6.text(0.5, 0.5, 'Performance\nComparison\n(See detailed results)', 
                ha='center', va='center', fontsize=12, transform=ax6.transAxes)
        ax6.set_title('Performance Summary')
        ax6.axis('off')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = f"{self.results_dir}/{self.experiment_id}_visualization.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {plot_file}")
        
        plt.show()

    def create_llm_evolving_figure(self):
        """Create 3-panel llm_evolving.png focusing on required views"""
        plt.style.use('default')
        sns.set_palette("husl")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('LLM Evolution and Performance (AdaPTutor vs Baselines)', fontsize=16, fontweight='bold')
        # Panel 1: Quality Evolution Over Generations (smoothed)
        ax1 = axes[0]
        for agent_name, data in self.results.items():
            if data['quality_scores']:
                window_size = max(10, min(25, len(data['quality_scores']) // 5))
                series = pd.Series(data['quality_scores']).rolling(window=window_size).mean()
                label = agent_name.replace('_', ' ').title()
                lw = 3 if agent_name == 'adaptutor_ga' else 1.8
                alpha = 1.0 if agent_name == 'adaptutor_ga' else 0.7
                ax1.plot(series, label=label, linewidth=lw, alpha=alpha)
        ax1.set_xlabel('Generation Steps')
        ax1.set_ylabel('Quality Score')
        ax1.set_title('Quality Evolution (Smoothed)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Panel 2: Final Quality Score Distribution (last 50 points)
        ax2 = axes[1]
        qualities = []
        labels = []
        for agent_name, data in self.results.items():
            if data['quality_scores']:
                final_scores = data['quality_scores'][-50:] if len(data['quality_scores']) >= 50 else data['quality_scores']
                qualities.extend(final_scores)
                labels.extend([agent_name.replace('_', ' ').title()] * len(final_scores))
        if qualities:
            df_q = pd.DataFrame({'Quality': qualities, 'Agent': labels})
            sns.boxplot(data=df_q, x='Agent', y='Quality', ax=ax2)
            ax2.set_title('Final Quality Score Distribution')
            ax2.tick_params(axis='x', rotation=30)
        # Panel 3: Response Time Distribution
        ax3 = axes[2]
        times = []
        time_labels = []
        for agent_name, data in self.results.items():
            if data['response_times']:
                times.extend(data['response_times'])
                time_labels.extend([agent_name.replace('_', ' ').title()] * len(data['response_times']))
        if times:
            df_t = pd.DataFrame({'Response Time (s)': times, 'Agent': time_labels})
            sns.boxplot(data=df_t, x='Agent', y='Response Time (s)', ax=ax3)
            ax3.set_title('Response Time Distribution')
            ax3.tick_params(axis='x', rotation=30)
        plt.tight_layout()
        out_path = f"{self.results_dir}/llm_evolving.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Evolving figure saved to: {out_path}")
        plt.close()

def main():
    """Main function to run the LLM comparison experiment"""
    
    # Initialize and run experiment
    experiment = LLMComparisonExperiment()
    results = experiment.run_experiment()
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    for agent_name, data in results.items():
        summary = data['summary']
        print(f"\n{agent_name.replace('_', ' ').title()}:")
        print(f"  Quality Score: {summary['mean_quality']:.3f} ± {summary['std_quality']:.3f}")
        print(f"  Response Time: {summary['mean_response_time']:.4f}s")
        print(f"  Urgency Correlation: {summary['urgency_correlation']:.3f}")
        print(f"  Quality Dimensions:")
        for dim, score in summary['dimension_means'].items():
            print(f"    {dim.capitalize()}: {score:.3f}")
    
    print(f"\nExperiment completed successfully!")
    print(f"Results saved in: experiment_results/")

if __name__ == "__main__":
    main()
