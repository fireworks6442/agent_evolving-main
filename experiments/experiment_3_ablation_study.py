"""
Experiment 3: Ablation Study
Tests the contribution of different components in AdaPTutor system:

Template Agent Components:
1. RL+IL (Full AdaPTutor) - Reinforcement Learning + Imitation Learning
2. IL-Only - Imitation Learning only
3. RL-Only - Reinforcement Learning only  
4. Static - No learning, fixed templates

LLM Agent Components:
1. GA (Full AdaPTutor) - Genetic Algorithm evolution
2. Grid Search - Traditional hyperparameter optimization

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
from response_agent import ResponseAgent
from evolution_system import EvolutionSystem
from metrics_framework import MetricsFramework
from llm_response_agent import LLMResponseAgent
from llm_evolution_system import LLMEvolutionSystem
from llm_metrics_framework import LLMMetricsFramework

class AblationStudyExperiment:
    def __init__(self, data_path='Stanford.csv', results_dir='experiment_results'):
        """Initialize the ablation study experiment"""
        self.data_path = data_path
        self.results_dir = results_dir
        self.experiment_id = f"ablation_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create results directory
        import os
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize components
        self.classifier = BinaryUrgencyClassifier()
        self.template_metrics = MetricsFramework()
        self.llm_metrics = LLMMetricsFramework()
        
        # Experiment configuration - reduced for faster execution
        self.config = {
            'evolution_cycles': 5,   # Template agent evolution cycles (reduced from 15)
            'ga_generations': 3,     # LLM agent GA generations (reduced from 15)
            'samples_per_cycle': 30, # Samples per evolution cycle
            'test_samples': 200,     # Test samples for evaluation
            'learning_rate': 0.01,   # RL learning rate
            'exploration_rate': 0.3, # RL exploration
            'mutation_rate': 0.3,    # GA mutation rate
            'crossover_rate': 0.7,   # GA crossover rate
            'population_size': 8     # GA population size
        }
        
        # Results storage
        self.template_results = {
            'rl_il_full': {'quality_scores': [], 'response_times': [], 'template_effectiveness': []},
            'il_only': {'quality_scores': [], 'response_times': [], 'template_effectiveness': []},
            'rl_only': {'quality_scores': [], 'response_times': [], 'template_effectiveness': []},
            'static': {'quality_scores': [], 'response_times': [], 'template_effectiveness': []}
        }
        
        self.llm_results = {
            'ga_evolution': {'quality_scores': [], 'response_times': [], 'fitness_scores': []},
            'grid_search': {'quality_scores': [], 'response_times': [], 'fitness_scores': []}
        }
        
        print(f"Ablation Study Experiment initialized: {self.experiment_id}")
    
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
    
    def setup_template_agents(self):
        """Setup template-based agents for ablation study"""
        
        # 1. Full RL+IL Agent (AdaPTutor)
        class RLILAgent:
            def __init__(self, config):
                self.agent = ResponseAgent()
                self.evolution_system = EvolutionSystem(
                    learning_rate=config['learning_rate'],
                    exploration_rate=config['exploration_rate']
                )
                self.evolution_system.enable_reinforcement_learning = True
                self.evolution_system.enable_imitation_learning = True
            
            def generate_response(self, text, context=None):
                # Get evolved action from evolution system
                state = {'text': text, 'context': context}
                action = self.evolution_system.get_evolved_action(state)
                
                # Generate response using evolved parameters
                response_data = self.agent.generate_response(text, context, action)
                
                return response_data
            
            def add_experience(self, text, response, reward, context=None):
                state = {'text': text, 'context': context}
                action = {'response': response}
                next_state = state.copy()
                response_data = {'quality_score': reward, 'response_time': 0.1}
                self.evolution_system.record_experience(state, action, reward, next_state, response_data)
            
            def evolve(self):
                if len(self.evolution_system.experience_buffer) >= 5:
                    self.evolution_system.evolve_agent_parameters()
        
        # 2. IL-Only Agent
        class ILOnlyAgent:
            def __init__(self, config):
                self.agent = ResponseAgent()
                self.evolution_system = EvolutionSystem(
                    learning_rate=config['learning_rate'],
                    exploration_rate=0.0  # No exploration for IL-only
                )
                self.evolution_system.enable_reinforcement_learning = False
                self.evolution_system.enable_imitation_learning = True
            
            def generate_response(self, text, context=None):
                state = {'text': text, 'context': context}
                action = self.evolution_system.get_evolved_action(state)
                response_data = self.agent.generate_response(text, context, action)
                return response_data
            
            def add_experience(self, text, response, reward, context=None):
                # Only use high-quality responses for imitation
                if reward > 0.7:
                    state = {'text': text, 'context': context}
                    action = {'response': response}
                    next_state = state.copy()
                    response_data = {'quality_score': reward, 'response_time': 0.1}
                    self.evolution_system.record_experience(state, action, reward, next_state, response_data)
            
            def evolve(self):
                if len(self.evolution_system.experience_buffer) >= 5:
                    self.evolution_system.evolve_agent_parameters()
        
        # 3. RL-Only Agent
        class RLOnlyAgent:
            def __init__(self, config):
                self.agent = ResponseAgent()
                self.evolution_system = EvolutionSystem(
                    learning_rate=config['learning_rate'],
                    exploration_rate=config['exploration_rate'],
                    decay_rate=0.995,
                    models_dir='models/rl_only'
                )
                self.evolution_system.enable_reinforcement_learning = True
                self.evolution_system.enable_imitation_learning = False
            
            def generate_response(self, text, context=None):
                state = {'text': text, 'context': context}
                action = self.evolution_system.get_evolved_action(state)
                response_data = self.agent.generate_response(text, context, action)
                return response_data
            
            def add_experience(self, text, response, reward, context=None):
                state = {'text': text, 'context': context}
                action = {'response': response}
                next_state = state.copy()
                response_data = {'quality_score': reward, 'response_time': 0.1}
                self.evolution_system.record_experience(state, action, reward, next_state, response_data)
            
            def evolve(self):
                if len(self.evolution_system.experience_buffer) >= 5:
                    self.evolution_system.evolve_agent_parameters()
        
        # 4. Static Agent (No Learning)
        class StaticAgent:
            def __init__(self, config):
                self.agent = ResponseAgent()
                # Use fixed, reasonable parameters
                self.fixed_action = {
                    'template_selection': 0.5,
                    'urgency_threshold': 0.6,
                    'response_length': 0.7,
                    'formality_level': 0.5,
                    'technical_depth': 0.6
                }
            
            def generate_response(self, text, context=None):
                response_data = self.agent.generate_response(text, context, self.fixed_action)
                return response_data
            
            def add_experience(self, text, response, reward, context=None):
                pass  # No learning
            
            def evolve(self):
                pass  # No evolution
        
        return {
            'rl_il_full': RLILAgent(self.config),
            'il_only': ILOnlyAgent(self.config),
            'rl_only': RLOnlyAgent(self.config),
            'static': StaticAgent(self.config)
        }
    
    def setup_llm_agents(self):
        """Setup LLM-based agents for ablation study"""
        
        # 1. GA Evolution Agent (Full AdaPTutor)
        class GAEvolutionAgent:
            def __init__(self, config):
                self.agent = LLMResponseAgent()
                self.evolution_system = LLMEvolutionSystem(
                    agent=self.agent,
                    population_size=config['population_size'],
                    mutation_rate=config['mutation_rate'],
                    crossover_rate=config['crossover_rate'],
                    elite_ratio=0.2
                )
            
            def generate_response(self, text, context=None):
                return self.agent.generate_response(text, context)
            
            def add_experience(self, text, response, quality_score):
                current_params = self.agent.evolution_params.copy()
                quality_scores = {'overall_quality': quality_score}
                self.evolution_system.add_experience(
                    parameters=current_params,
                    quality_scores=quality_scores,
                    response_time=0.1
                )
            
            def evolve(self):
                if len(self.evolution_system.experience_buffer) >= 5:
                    evolved_params = self.evolution_system.evolve_parameters()
                    self.agent.update_evolution_params(evolved_params)
        
        # 2. Grid Search Agent
        class GridSearchAgent:
            def __init__(self, config):
                self.agent = LLMResponseAgent()
                self.search_completed = False
                self.current_param_index = 0
                
                # Define parameter grid (focused on key parameters)
                self.parameter_grid = {
                    'temperature': [0.3, 0.5, 0.7, 0.9],
                    'explanation_depth': [0.3, 0.5, 0.7, 0.9],
                    'technical_level': [0.3, 0.5, 0.7],
                    'engagement_factor': [0.4, 0.6, 0.8]
                }
                
                self.param_combinations = list(ParameterGrid(self.parameter_grid))
                self.param_performance = {}
                self.generation_count = 0
                
                # Start with first parameter combination
                self.current_parameters = self._get_full_parameters(self.param_combinations[0])
                self.agent.update_evolution_params(self.current_parameters)
            
            def _get_full_parameters(self, grid_params):
                """Convert grid parameters to full parameter set"""
                full_params = {
                    'temperature': grid_params['temperature'],
                    'max_tokens': 400,
                    'top_p': 0.9,
                    'frequency_penalty': 0.2,
                    'presence_penalty': 0.1,
                    'explanation_depth': grid_params['explanation_depth'],
                    'technical_level': grid_params['technical_level'],
                    'engagement_factor': grid_params['engagement_factor'],
                    'system_prompt_style': 1,
                    'response_format': 1
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
            
            def add_experience(self, text, response, quality_score):
                pass  # Grid search doesn't use experience directly
            
            def evolve(self):
                """Update grid search progress"""
                self.generation_count += 1
                
                if not self.search_completed and self.generation_count % 3 == 0:
                    # Move to next parameter combination
                    self.current_param_index = (self.current_param_index + 1) % len(self.param_combinations)
                    
                    if self.current_param_index == 0:
                        # Completed one full cycle, find best parameters
                        self._find_best_parameters()
                        self.search_completed = True
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
                
                self.current_parameters = self._get_full_parameters(self.param_combinations[best_index])
                self.agent.update_evolution_params(self.current_parameters)
        
        return {
            'ga_evolution': GAEvolutionAgent(self.config),
            'grid_search': GridSearchAgent(self.config)
        }
    
    def run_template_ablation(self, train_data, test_data):
        """Run ablation study for template-based agents"""
        print("\nRunning Template Agent Ablation Study...")
        print("-" * 50)
        
        # Setup template agents
        template_agents = self.setup_template_agents()
        
        # Run evolution cycles
        for cycle in range(self.config['evolution_cycles']):
            print(f"\nEvolution Cycle {cycle + 1}/{self.config['evolution_cycles']}")
            
            # Sample data for this cycle
            cycle_data = train_data.sample(n=min(self.config['samples_per_cycle'], len(train_data)))
            
            for agent_name, agent in template_agents.items():
                cycle_results = {
                    'quality_scores': [],
                    'response_times': [],
                    'template_effectiveness': []
                }
                
                for _, row in cycle_data.iterrows():
                    text = row['post_text']
                    true_urgency = row['urgency_continuous']
                    
                    # Generate response
                    response_data = agent.generate_response(text)
                    
                    # Evaluate response quality using simple metrics
                    # Create a simple quality evaluation based on response characteristics
                    quality_score = min(1.0, max(0.0, 
                        0.5 + 0.3 * (len(response_data['response']) > 50) + 
                        0.2 * (response_data['response_time'] < 2.0)
                    ))
                    
                    eff = response_data.get('confidence', 0.5)
                    if isinstance(eff, list):
                        eff = eff[0] if eff else 0.0
                    try:
                        eff = float(eff)
                    except Exception:
                        eff = 0.5
                    quality_metrics = {
                        'overall_quality': quality_score,
                        'template_effectiveness': eff
                    }
                    
                    # Add experience and evolve
                    agent.add_experience(text, response_data['response'], quality_metrics['overall_quality'])
                    
                    # Record results
                    cycle_results['quality_scores'].append(quality_metrics['overall_quality'])
                    cycle_results['response_times'].append(response_data['response_time'])
                    cycle_results['template_effectiveness'].append(quality_metrics.get('template_effectiveness', 0.5))
                
                # Evolve agent
                agent.evolve()
                
                # Store results
                self.template_results[agent_name]['quality_scores'].extend(cycle_results['quality_scores'])
                self.template_results[agent_name]['response_times'].extend(cycle_results['response_times'])
                self.template_results[agent_name]['template_effectiveness'].extend(cycle_results['template_effectiveness'])
                
                # Print progress
                mean_quality = np.mean(cycle_results['quality_scores'])
                mean_template_eff = np.mean(cycle_results['template_effectiveness'])
                print(f"  {agent_name}: Quality = {mean_quality:.3f}, Template Eff = {mean_template_eff:.3f}")
        
        # Final evaluation on test set (use smaller sample for consistency)
        print("\nFinal Template Agent Evaluation...")
        template_final_results = {}
        
        # Use smaller test set for final evaluation for consistency with LLM agents
        final_test_data = test_data.sample(n=min(50, len(test_data)))
        print(f"Using {len(final_test_data)} samples for final evaluation...")
        
        for agent_name, agent in template_agents.items():
            test_results = self.evaluate_template_agent(agent, final_test_data, agent_name)
            template_final_results[agent_name] = test_results
            
            print(f"\n{agent_name}:")
            print(f"  Mean Quality: {test_results['summary']['mean_quality']:.3f} ± {test_results['summary']['std_quality']:.3f}")
            print(f"  Template Effectiveness: {test_results['summary']['mean_template_effectiveness']:.3f}")
            print(f"  Response Time: {test_results['summary']['mean_response_time']:.3f}s")
        
        return template_final_results
    
    def run_llm_ablation(self, train_data, test_data):
        """Run ablation study for LLM-based agents"""
        print("\nRunning LLM Agent Ablation Study...")
        print("-" * 50)
        
        # Setup LLM agents
        llm_agents = self.setup_llm_agents()
        
        # Run GA generations
        for generation in range(self.config['ga_generations']):
            print(f"\nGeneration {generation + 1}/{self.config['ga_generations']}")
            
            # Sample data for this generation
            generation_data = train_data.sample(n=min(self.config['samples_per_cycle'], len(train_data)))
            
            for agent_name, agent in llm_agents.items():
                generation_results = {
                    'quality_scores': [],
                    'response_times': [],
                    'fitness_scores': []
                }
                
                for _, row in generation_data.iterrows():
                    text = row['post_text']
                    true_urgency = row['urgency_continuous']
                    
                    # Generate response
                    response_data = agent.generate_response(text)
                    
                    # Evaluate response quality
                    quality_metrics = self.llm_metrics.evaluate_response_quality(
                        input_text=text,
                        response=response_data['response'],
                        urgency_level=true_urgency,
                        content_category=response_data.get('content_category', 'general'),
                        evolution_step=generation
                    )
                    
                    # Add experience and evolve
                    agent.add_experience(text, response_data['response'], quality_metrics['overall_quality'])
                    
                    # Record results
                    generation_results['quality_scores'].append(quality_metrics['overall_quality'])
                    generation_results['response_times'].append(response_data['response_time'])
                    generation_results['fitness_scores'].append(quality_metrics['evolution_fitness'])
                
                # Evolve agent
                agent.evolve()
                
                # Store results
                self.llm_results[agent_name]['quality_scores'].extend(generation_results['quality_scores'])
                self.llm_results[agent_name]['response_times'].extend(generation_results['response_times'])
                self.llm_results[agent_name]['fitness_scores'].extend(generation_results['fitness_scores'])
                
                # Print progress
                mean_quality = np.mean(generation_results['quality_scores'])
                mean_fitness = np.mean(generation_results['fitness_scores'])
                print(f"  {agent_name}: Quality = {mean_quality:.3f}, Fitness = {mean_fitness:.3f}")
        
        # Final evaluation on test set (use smaller sample to avoid timeout)
        print("\nFinal LLM Agent Evaluation...")
        llm_final_results = {}
        
        # Use smaller test set for final evaluation to avoid API timeout
        final_test_data = test_data.sample(n=min(50, len(test_data)))
        print(f"Using {len(final_test_data)} samples for final evaluation to avoid timeout...")
        
        for agent_name, agent in llm_agents.items():
            test_results = self.evaluate_llm_agent(agent, final_test_data, agent_name)
            llm_final_results[agent_name] = test_results
            
            print(f"\n{agent_name}:")
            print(f"  Mean Quality: {test_results['summary']['mean_quality']:.3f} ± {test_results['summary']['std_quality']:.3f}")
            print(f"  Response Time: {test_results['summary']['mean_response_time']:.3f}s")
            print(f"  Fitness Score: {test_results['summary']['mean_fitness']:.3f}")
        
        return llm_final_results
    
    def evaluate_template_agent(self, agent, test_data, agent_name):
        """Evaluate a template agent on test data"""
        results = {
            'quality_scores': [],
            'response_times': [],
            'template_effectiveness': [],
            'urgency_predictions': [],
            'true_urgencies': []
        }
        
        for _, row in test_data.iterrows():
            text = row['post_text']
            true_urgency = row['urgency_continuous']
            
            # Generate response
            response_data = agent.generate_response(text)
            
            # Evaluate response quality using template metrics
            # Create a simple quality evaluation based on response characteristics
            quality_score = min(1.0, max(0.0, 
                0.5 + 0.3 * (len(response_data['response']) > 50) + 
                0.2 * (response_data['response_time'] < 2.0)
            ))
            
            quality_metrics = {
                'overall_quality': quality_score,
                'template_effectiveness': response_data.get('confidence', 0.5)
            }
            
            # Record results
            results['quality_scores'].append(quality_metrics['overall_quality'])
            results['response_times'].append(response_data['response_time'])
            results['template_effectiveness'].append(quality_metrics.get('template_effectiveness', 0.5))
            results['urgency_predictions'].append(response_data.get('urgency_level', 0.5))
            results['true_urgencies'].append(true_urgency)
        
        # Calculate summary statistics
        summary = {
            'mean_quality': np.mean(results['quality_scores']),
            'std_quality': np.std(results['quality_scores']),
            'mean_template_effectiveness': np.mean(results['template_effectiveness']),
            'mean_response_time': np.mean(results['response_times']),
            'urgency_correlation': np.corrcoef(results['true_urgencies'], results['urgency_predictions'])[0, 1]
        }
        
        return {'test_results': results, 'summary': summary}
    
    def evaluate_llm_agent(self, agent, test_data, agent_name):
        """Evaluate an LLM agent on test data"""
        results = {
            'quality_scores': [],
            'response_times': [],
            'fitness_scores': [],
            'urgency_predictions': [],
            'true_urgencies': []
        }
        
        for _, row in test_data.iterrows():
            text = row['post_text']
            true_urgency = row['urgency_continuous']
            
            # Generate response
            response_data = agent.generate_response(text)
            
            # Evaluate response quality
            quality_metrics = self.llm_metrics.evaluate_response_quality(
                input_text=text,
                response=response_data['response'],
                urgency_level=true_urgency,
                content_category=response_data.get('content_category', 'general'),
                evolution_step=0
            )
            
            # Record results
            results['quality_scores'].append(quality_metrics['overall_quality'])
            results['response_times'].append(response_data['response_time'])
            results['fitness_scores'].append(quality_metrics['evolution_fitness'])
            results['urgency_predictions'].append(response_data.get('urgency_level', 0.5))
            results['true_urgencies'].append(true_urgency)
        
        # Calculate summary statistics
        summary = {
            'mean_quality': np.mean(results['quality_scores']),
            'std_quality': np.std(results['quality_scores']),
            'mean_fitness': np.mean(results['fitness_scores']),
            'mean_response_time': np.mean(results['response_times']),
            'urgency_correlation': np.corrcoef(results['true_urgencies'], results['urgency_predictions'])[0, 1]
        }
        
        return {'test_results': results, 'summary': summary}
    
    def run_experiment(self):
        """Run the complete ablation study experiment"""
        print("Starting Ablation Study Experiment")
        print("=" * 60)
        
        # Load data
        train_data, test_data = self.load_and_prepare_data()
        
        # Run template agent ablation
        template_results = self.run_template_ablation(train_data, test_data)
        
        # Run LLM agent ablation
        llm_results = self.run_llm_ablation(train_data, test_data)
        
        # Combine results
        final_results = {
            'template_agents': template_results,
            'llm_agents': llm_results
        }
        
        # Save results
        self.save_results(final_results)
        
        # Generate visualizations
        self.create_visualizations()
        
        return final_results
    
    def save_results(self, final_results):
        """Save experiment results to JSON"""
        results_file = f"{self.results_dir}/{self.experiment_id}_results.json"
        
        # Prepare results for JSON serialization
        json_results = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'template_evolution_results': {
                agent_name: {
                    'quality_scores': [float(x) for x in data['quality_scores']],
                    'response_times': [float(x) for x in data['response_times']],
                    'template_effectiveness': [float(x) if not isinstance(x, list) else float(x[0]) if x else 0.0 
                                             for x in data['template_effectiveness']]
                }
                for agent_name, data in self.template_results.items()
            },
            'llm_evolution_results': {
                agent_name: {
                    'quality_scores': [float(x) for x in data['quality_scores']],
                    'response_times': [float(x) for x in data['response_times']],
                    'fitness_scores': [float(x) for x in data['fitness_scores']]
                }
                for agent_name, data in self.llm_results.items()
            },
            'final_evaluation': {
                'template_agents': {
                    agent_name: {
                        'summary': {
                            k: float(v) if isinstance(v, (int, float, np.number)) else v
                            for k, v in data['summary'].items()
                        }
                    }
                    for agent_name, data in final_results['template_agents'].items()
                },
                'llm_agents': {
                    agent_name: {
                        'summary': {
                            k: float(v) if isinstance(v, (int, float, np.number)) else v
                            for k, v in data['summary'].items()
                        }
                    }
                    for agent_name, data in final_results['llm_agents'].items()
                }
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    def create_visualizations(self):
        """Create visualizations for the ablation study results"""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Ablation Study Results', fontsize=16, fontweight='bold')
        
        # 1. Template Agent Quality Evolution
        ax1 = axes[0, 0]
        for agent_name, data in self.template_results.items():
            if data['quality_scores']:
                # Calculate moving average
                window_size = 10
                quality_scores = data['quality_scores']
                if len(quality_scores) >= window_size:
                    moving_avg = pd.Series(quality_scores).rolling(window=window_size).mean()
                    ax1.plot(moving_avg, label=agent_name.replace('_', ' ').title(), linewidth=2)
        
        ax1.set_xlabel('Evolution Steps')
        ax1.set_ylabel('Quality Score')
        ax1.set_title('Template Agent Quality Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. LLM Agent Quality Evolution
        ax2 = axes[0, 1]
        for agent_name, data in self.llm_results.items():
            if data['quality_scores']:
                window_size = 10
                quality_scores = data['quality_scores']
                if len(quality_scores) >= window_size:
                    moving_avg = pd.Series(quality_scores).rolling(window=window_size).mean()
                    ax2.plot(moving_avg, label=agent_name.replace('_', ' ').title(), linewidth=2)
        
        ax2.set_xlabel('Generation Steps')
        ax2.set_ylabel('Quality Score')
        ax2.set_title('LLM Agent Quality Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Template Agent Component Comparison
        ax3 = axes[1, 0]
        template_qualities = []
        template_labels = []
        
        for agent_name, data in self.template_results.items():
            if data['quality_scores']:
                # Take final scores for comparison
                final_scores = data['quality_scores'][-30:] if len(data['quality_scores']) >= 30 else data['quality_scores']
                template_qualities.extend(final_scores)
                template_labels.extend([agent_name.replace('_', ' ').title()] * len(final_scores))
        
        if template_qualities:
            df_template = pd.DataFrame({'Quality': template_qualities, 'Agent': template_labels})
            sns.boxplot(data=df_template, x='Agent', y='Quality', ax=ax3)
            ax3.set_title('Template Agent Component Comparison')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. LLM Agent Component Comparison
        ax4 = axes[1, 1]
        llm_qualities = []
        llm_labels = []
        
        for agent_name, data in self.llm_results.items():
            if data['quality_scores']:
                final_scores = data['quality_scores'][-30:] if len(data['quality_scores']) >= 30 else data['quality_scores']
                llm_qualities.extend(final_scores)
                llm_labels.extend([agent_name.replace('_', ' ').title()] * len(final_scores))
        
        if llm_qualities:
            df_llm = pd.DataFrame({'Quality': llm_qualities, 'Agent': llm_labels})
            sns.boxplot(data=df_llm, x='Agent', y='Quality', ax=ax4)
            ax4.set_title('LLM Agent Component Comparison')
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. Template Effectiveness Evolution
        ax5 = axes[2, 0]
        for agent_name, data in self.template_results.items():
            if data['template_effectiveness']:
                window_size = 10
                effectiveness = pd.to_numeric(pd.Series(data['template_effectiveness']), errors='coerce').dropna()
                if len(effectiveness) >= window_size:
                    moving_avg = effectiveness.rolling(window=window_size).mean()
                    ax5.plot(moving_avg, label=agent_name.replace('_', ' ').title(), linewidth=2)
        
        ax5.set_xlabel('Evolution Steps')
        ax5.set_ylabel('Template Effectiveness')
        ax5.set_title('Template Effectiveness Evolution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Fitness Score Evolution (LLM)
        ax6 = axes[2, 1]
        for agent_name, data in self.llm_results.items():
            if data['fitness_scores']:
                window_size = 10
                fitness_scores = data['fitness_scores']
                if len(fitness_scores) >= window_size:
                    moving_avg = pd.Series(fitness_scores).rolling(window=window_size).mean()
                    ax6.plot(moving_avg, label=agent_name.replace('_', ' ').title(), linewidth=2)
        
        ax6.set_xlabel('Generation Steps')
        ax6.set_ylabel('Fitness Score')
        ax6.set_title('LLM Agent Fitness Evolution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = f"{self.results_dir}/{self.experiment_id}_visualization.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {plot_file}")
        
        plt.show()

def main():
    """Main function to run the ablation study experiment"""
    
    # Initialize and run experiment
    experiment = AblationStudyExperiment()
    results = experiment.run_experiment()
    
    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)
    
    print("\nTemplate Agent Results:")
    for agent_name, data in results['template_agents'].items():
        summary = data['summary']
        print(f"\n{agent_name.replace('_', ' ').title()}:")
        print(f"  Quality Score: {summary['mean_quality']:.3f} ± {summary['std_quality']:.3f}")
        print(f"  Template Effectiveness: {summary['mean_template_effectiveness']:.3f}")
        print(f"  Response Time: {summary['mean_response_time']:.4f}s")
        print(f"  Urgency Correlation: {summary['urgency_correlation']:.3f}")
    
    print("\nLLM Agent Results:")
    for agent_name, data in results['llm_agents'].items():
        summary = data['summary']
        print(f"\n{agent_name.replace('_', ' ').title()}:")
        print(f"  Quality Score: {summary['mean_quality']:.3f} ± {summary['std_quality']:.3f}")
        print(f"  Fitness Score: {summary['mean_fitness']:.3f}")
        print(f"  Response Time: {summary['mean_response_time']:.4f}s")
        print(f"  Urgency Correlation: {summary['urgency_correlation']:.3f}")
    
    print(f"\nAblation study completed successfully!")
    print(f"Results saved in: experiment_results/")

if __name__ == "__main__":
    main()
