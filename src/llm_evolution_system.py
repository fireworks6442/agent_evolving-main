"""
LLM Evolution System Module
Implements genetic algorithm-based parameter evolution for LLM Response Agents
"""

import json
import numpy as np
import random
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from collections import deque, defaultdict
import copy
import math

class LLMEvolutionSystem:
    def __init__(self, agent, population_size: int = 8, mutation_rate: float = 0.8, 
                 crossover_rate: float = 0.9, elite_ratio: float = 0.2):
        """
        Initialize the LLM Evolution System with ENHANCED parameters for better evolution
        
        Args:
            agent: The LLM agent to evolve
            population_size: Size of the parameter population
            mutation_rate: Probability of parameter mutation (INCREASED for better exploration)
            crossover_rate: Probability of crossover between parents (INCREASED)
            elite_ratio: Fraction of population to keep as elites (REDUCED for more exploration)
        """
        self.agent = agent
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.adaptive_mutation_rate = mutation_rate
        self.best_fitness_history = []
        
        # ENHANCED parameter ranges with WIDER exploration space for dramatic improvements
        self.parameter_ranges = {
            'temperature': (0.1, 1.5),        # MUCH WIDER range for dramatic changes
            'max_tokens': (50, 1500),         # Much wider response length range
            'top_p': (0.3, 0.99),            # Wider range starting lower
            'frequency_penalty': (0.0, 1.0), # FULL penalty range
            'presence_penalty': (0.0, 1.0),  # FULL penalty range
            'explanation_depth': (0.1, 1.0), # FULL range for explanation depth
            'technical_level': (0.1, 1.0),   # FULL range for technical content
            'engagement_factor': (0.1, 1.0), # FULL engagement range
            'system_prompt_style': (0, 5),   # MORE prompt style options
            'response_format': (0, 4)        # MORE response format options
        }
        self.adaptive_mutation_rate = mutation_rate
        self.best_fitness_history = []
        
        # ENHANCED parameter ranges with WIDER exploration space
        self.parameter_ranges = {
            'temperature': (0.1, 1.2),        # WIDER range for better exploration
            'max_tokens': (50, 1000),         # Wider response length range
            'top_p': (0.6, 0.99),            # Wider range for diversity
            'frequency_penalty': (0.0, 0.8), # Wider penalty range
            'presence_penalty': (0.0, 0.7),  # Wider penalty range
            'explanation_depth': (0.1, 1.0), # FULL range for explanation depth
            'technical_level': (0.1, 1.0),   # FULL range for technical content
            'engagement_factor': (0.2, 1.0), # Wider engagement range
            'system_prompt_style': (0, 4),   # More prompt style options
            'response_format': (0, 3)        # More response format options
        }
        
        # Evolution tracking
        self.generation = 0
        self.population = []
        self.fitness_scores = []
        self.experience_buffer = deque(maxlen=200)
        self.fitness_history = []
        
        # Enhanced quality tracking
        self.quality_history = deque(maxlen=100)
        self.dimension_history = {
            'relevance': deque(maxlen=50),
            'helpfulness': deque(maxlen=50),
            'clarity': deque(maxlen=50),
            'completeness': deque(maxlen=50),
            'engagement': deque(maxlen=50)
        }
        
        # Convergence tracking
        self.convergence_threshold = 0.01
        self.stagnation_counter = 0
        self.max_stagnation = 5
        
        # Dimension-specific optimization weights - FOCUS ON DECLINING DIMENSIONS
        self.dimension_weights = {
            'relevance': 0.35,     # HIGH priority for declining dimension
            'helpfulness': 0.35,   # HIGH priority for declining dimension
            'clarity': 0.25,       # MEDIUM priority for declining dimension
            'completeness': 0.03,  # LOW priority for stable dimension
            'engagement': 0.02     # LOW priority
        }
        
        # Quality targets for each dimension - REALISTIC TARGETS
        self.quality_targets = {
            'relevance': 0.75,     # Target for improvement
            'helpfulness': 0.70,   # Target for improvement
            'clarity': 0.75,       # Target for improvement
            'completeness': 0.85,  # Keep higher for stable dimension
            'engagement': 0.65     # Moderate target
        }
        
        # Initialize population with ENHANCED diversity
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize population with MAXIMUM diversity and strategic parameter distribution"""
        self.population = []
        self.fitness_scores = []
        
        for i in range(self.population_size):
            individual = {}
            
            # STRATEGIC initialization for better parameter exploration
            if i == 0:
                # DELIBERATELY POOR parameters for demonstration
                individual = {
                    'temperature': 1.1,           # Very high temperature (poor quality)
                    'explanation_depth': 0.15,    # Very low depth (poor helpfulness)
                    'technical_level': 0.2,       # Low technical level
                    'top_p': 0.65,               # Low top_p (poor diversity)
                    'max_tokens': 80,            # Very short responses
                    'frequency_penalty': 0.1,
                    'presence_penalty': 0.1,
                    'engagement_factor': 0.3,    # Low engagement
                    'system_prompt_style': 0,
                    'response_format': 0
                }
            elif i == 1:
                # MODERATE parameters
                individual = {
                    'temperature': 0.7,
                    'explanation_depth': 0.5,
                    'technical_level': 0.5,
                    'top_p': 0.85,
                    'max_tokens': 400,
                    'frequency_penalty': 0.3,
                    'presence_penalty': 0.3,
                    'engagement_factor': 0.6,
                    'system_prompt_style': 1,
                    'response_format': 1
                }
            elif i == 2:
                # GOOD parameters (target)
                individual = {
                    'temperature': 0.4,
                    'explanation_depth': 0.8,
                    'technical_level': 0.6,
                    'top_p': 0.92,
                    'max_tokens': 600,
                    'frequency_penalty': 0.4,
                    'presence_penalty': 0.3,
                    'engagement_factor': 0.8,
                    'system_prompt_style': 2,
                    'response_format': 2
                }
            else:
                # RANDOM initialization with FULL range exploration
                for param, (min_val, max_val) in self.parameter_ranges.items():
                    if param in ['system_prompt_style', 'response_format']:
                        individual[param] = random.randint(int(min_val), int(max_val))
                    else:
                        # Use FULL parameter range for maximum diversity
                        individual[param] = random.uniform(min_val, max_val)
            
            self.population.append(individual)
            self.fitness_scores.append(0.0)
        
        return self.population

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get evolution summary with comprehensive metrics"""
        if not self.fitness_history:
            return {
                'generation': self.generation,
                'best_fitness': 0.0,
                'improvement_rate': 0.0,
                'convergence_status': 'no_data',
                'experience_count': len(self.experience_buffer)
            }
        
        return {
            'generation': self.generation,
            'best_fitness': max(self.fitness_history) if self.fitness_history else 0.0,
            'improvement_rate': self._calculate_improvement_rate(),
            'convergence_status': self._assess_convergence(),
            'experience_count': len(self.experience_buffer),
            'population_diversity': self._calculate_population_diversity(),
            'dimension_progress': self._calculate_dimension_progress()
        }
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate improvement rate over recent generations"""
        if len(self.fitness_history) < 5:
            return 0.0
        
        recent_fitness = self.fitness_history[-5:]
        if len(recent_fitness) < 2:
            return 0.0
        
        # Linear trend over recent generations
        x = np.arange(len(recent_fitness))
        slope = np.polyfit(x, recent_fitness, 1)[0]
        return float(slope)
    
    def _assess_convergence(self) -> str:
        """Assess convergence status of the evolution"""
        if len(self.fitness_history) < 10:
            return 'insufficient_data'
        
        recent_fitness = self.fitness_history[-10:]
        fitness_std = np.std(recent_fitness)
        
        if fitness_std < self.convergence_threshold:
            return 'converged'
        elif fitness_std < self.convergence_threshold * 3:
            return 'converging'
        else:
            return 'evolving'
    
    def _calculate_population_diversity(self) -> float:
        """Calculate diversity within the current population"""
        if len(self.population) < 2:
            return 0.0
        
        diversities = []
        for i, ind1 in enumerate(self.population):
            for j, ind2 in enumerate(self.population[i+1:], i+1):
                similarity = self._calculate_parameter_similarity(ind1, ind2)
                diversity = 1 - similarity
                diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.0
    
    def _calculate_dimension_progress(self) -> Dict[str, float]:
        """Calculate progress for each quality dimension"""
        progress = {}
        
        for dimension in self.dimension_history.keys():
            if len(self.dimension_history[dimension]) >= 5:
                scores = list(self.dimension_history[dimension])
                if len(scores) >= 2:
                    # Calculate trend
                    x = np.arange(len(scores))
                    slope = np.polyfit(x, scores, 1)[0]
                    progress[dimension] = float(slope)
                else:
                    progress[dimension] = 0.0
            else:
                progress[dimension] = 0.0
        
        return progress
    
    def add_experience(self, parameters: Dict[str, Any], quality_scores: Dict[str, float], 
                      user_feedback: Optional[str] = None, response_time: float = 0.0):
        """
        Add experience with enhanced quality tracking
        
        Args:
            parameters: Parameter set used
            quality_scores: Dictionary of quality dimension scores
            user_feedback: Optional user feedback
            response_time: Response generation time
        """
        # Calculate dimension-specific rewards
        dimension_rewards = {}
        total_reward = 0.0
        
        for dimension, score in quality_scores.items():
            if dimension in self.quality_targets:
                target = self.quality_targets[dimension]
                weight = self.dimension_weights.get(dimension, 0.1)
                
                # Enhanced reward calculation
                if score >= target:
                    # Exponential reward for exceeding targets
                    reward = weight * (1.0 + (score - target) * 3.0)
                else:
                    # Penalty for being below target, but not too harsh
                    penalty_factor = max(0.3, score / target)
                    reward = weight * penalty_factor
                
                dimension_rewards[dimension] = reward
                total_reward += reward
        
        # Bonus for balanced performance across all dimensions
        if len(quality_scores) >= 3:
            # Extract only numeric quality dimension scores
            dimension_scores = []
            for key, value in quality_scores.items():
                if key in ['relevance', 'helpfulness', 'clarity', 'completeness', 'engagement'] and isinstance(value, (int, float)):
                    dimension_scores.append(value)
                elif key == 'quality_dimensions' and isinstance(value, dict):
                    # Handle nested quality_dimensions structure
                    for dim_key, dim_value in value.items():
                        if isinstance(dim_value, (int, float)):
                            dimension_scores.append(dim_value)
            
            if dimension_scores:
                # Ensure all scores are numeric before calculating min/max
                numeric_scores = []
                for score in dimension_scores:
                    try:
                        numeric_scores.append(float(score))
                    except (ValueError, TypeError):
                        continue
                
                if numeric_scores:
                    min_score = min(numeric_scores)
                    max_score = max(numeric_scores)
                    balance_bonus = max(0, 1.0 - (max_score - min_score)) * 0.2
                    total_reward += balance_bonus
        
        # Response time penalty (encourage efficiency)
        if response_time > 0:
            time_penalty = min(response_time / 10.0, 0.1)  # Max 0.1 penalty
            total_reward -= time_penalty
        
        # User feedback bonus
        feedback_bonus = 0.0
        if user_feedback:
            feedback_lower = user_feedback.lower()
            if any(word in feedback_lower for word in ['good', 'great', 'helpful', 'clear', 'perfect']):
                feedback_bonus = 0.3
            elif any(word in feedback_lower for word in ['bad', 'poor', 'unclear', 'unhelpful']):
                feedback_bonus = -0.2
        
        total_reward += feedback_bonus
        
        experience = {
            'timestamp': datetime.now().isoformat(),
            'parameters': parameters.copy(),
            'quality_scores': quality_scores.copy(),
            'dimension_rewards': dimension_rewards,
            'total_reward': total_reward,
            'user_feedback': user_feedback,
            'response_time': response_time,
            'generation': self.generation
        }
        
        self.experience_buffer.append(experience)
        
        # Update fitness scores for matching individuals
        self._update_fitness_scores()
    
    def _update_fitness_scores(self):
        """Update fitness scores based on recent experiences"""
        if not self.experience_buffer:
            return
        
        for i, individual in enumerate(self.population):
            fitness = self._calculate_fitness(individual)
            self.fitness_scores[i] = fitness
    
    def _calculate_fitness(self, individual: Dict[str, Any]) -> float:
        """
        Enhanced fitness calculation that rewards quality improvements
        
        Args:
            individual: Parameter set to evaluate
            
        Returns:
            Fitness score (higher is better)
        """
        if not self.experience_buffer:
            # Base fitness for new parameters
            return self._calculate_parameter_quality_heuristic(individual)
        
        # Find experiences with similar parameters
        similar_experiences = []
        for exp in self.experience_buffer:
            similarity = self._calculate_parameter_similarity(individual, exp['parameters'])
            if similarity > 0.3:  # Reasonable threshold for similarity
                similar_experiences.append((exp, similarity))
        
        if not similar_experiences:
            # Use heuristic for unexplored parameter space
            return self._calculate_parameter_quality_heuristic(individual)
        
        # Calculate fitness based on similar experiences
        total_fitness = 0.0
        total_weight = 0.0
        
        # Weight critical dimensions more heavily
        dimension_weights = {
            'relevance': 0.3,
            'helpfulness': 0.3,
            'clarity': 0.25,
            'completeness': 0.1,
            'engagement': 0.05
        }
        
        for exp, similarity in similar_experiences:
            # Quality scores are stored directly in the experience
            quality_dims = exp['quality_scores']
            age_weight = self._calculate_age_weight(exp)
            
            # Calculate weighted quality score for this experience
            exp_quality = 0.0
            for dim, weight in dimension_weights.items():
                if dim in quality_dims:
                    score = quality_dims[dim]
                    # Reward high scores more strongly
                    if score >= 0.7:
                        adjusted_score = score * 1.5  # Bonus for high quality
                    elif score >= 0.5:
                        adjusted_score = score
                    else:
                        adjusted_score = score * 0.7  # Penalty for low quality
                    
                    exp_quality += adjusted_score * weight
            
            # Weight by similarity and age
            experience_weight = similarity * age_weight
            total_fitness += exp_quality * experience_weight
            total_weight += experience_weight
        
        if total_weight > 0:
            base_fitness = total_fitness / total_weight
        else:
            base_fitness = 0.5
        
        # Add diversity bonus to encourage exploration
        diversity_bonus = self._calculate_diversity_bonus(individual)
        
        # Add parameter quality heuristic
        param_quality = self._calculate_parameter_quality_heuristic(individual)
        
        # Combine all components
        final_fitness = (base_fitness * 0.7 + 
                        diversity_bonus * 0.15 + 
                        param_quality * 0.15)
        
        return max(final_fitness, 0.01)  # Ensure positive fitness
    
    def _calculate_parameter_quality_heuristic(self, individual: Dict[str, Any]) -> float:
        """Calculate heuristic fitness based on parameter values"""
        quality_score = 0.5  # Base score
        
        # Temperature heuristics
        temp = individual.get('temperature', 0.5)
        if 0.3 <= temp <= 0.7:  # Good range for most tasks
            quality_score += 0.1
        elif temp < 0.2 or temp > 0.9:  # Extreme values
            quality_score -= 0.1
        
        # Explanation depth heuristics
        depth = individual.get('explanation_depth', 0.5)
        if 0.4 <= depth <= 0.8:  # Good range for helpfulness
            quality_score += 0.1
        
        # Technical level heuristics
        tech = individual.get('technical_level', 0.5)
        if 0.3 <= tech <= 0.7:  # Moderate technical level
            quality_score += 0.1
        
        # Top-p heuristics
        top_p = individual.get('top_p', 0.9)
        if 0.8 <= top_p <= 0.95:  # Good range for coherence
            quality_score += 0.05
        
        return max(0.1, min(1.0, quality_score))
    
    def _calculate_parameter_similarity(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
        """Calculate similarity between two parameter sets with RELAXED matching for better experience sharing"""
        if not params1 or not params2:
            return 0.0
        
        similarities = []
        
        for param in self.parameter_ranges.keys():
            if param in params1 and param in params2:
                val1, val2 = params1[param], params2[param]
                
                # Ensure both values are numeric before calculation
                try:
                    val1 = float(val1)
                    val2 = float(val2)
                except (ValueError, TypeError):
                    # Skip this parameter if values can't be converted to numbers
                    continue
                
                param_range = self.parameter_ranges[param][1] - self.parameter_ranges[param][0]
                
                if param_range > 0:
                    # RELAXED similarity calculation for better experience sharing
                    diff = abs(val1 - val2) / param_range
                    
                    # LINEAR similarity decay (instead of exponential) for more lenient matching
                    similarity = max(0, 1.0 - diff * 2.0)  # Linear decay, more forgiving
                    
                    # MODERATE weights for parameters (not too extreme)
                    if param in ['temperature', 'explanation_depth', 'technical_level']:
                        similarity *= 1.5  # Moderate weight for critical parameters
                    elif param in ['top_p', 'max_tokens']:
                        similarity *= 1.2  # Slight weight for important parameters
                    
                    similarities.append(similarity)
        
        if not similarities:
            return 0.0
        
        # Simple average without excessive bonuses
        return min(np.mean(similarities), 1.0)
    
    def _calculate_age_weight(self, experience: Dict[str, Any]) -> float:
        """Calculate age-based weight for experience (more recent = higher weight)"""
        try:
            exp_time = datetime.fromisoformat(experience['timestamp'])
            current_time = datetime.now()
            age_hours = (current_time - exp_time).total_seconds() / 3600
            
            # Exponential decay with 24-hour half-life
            weight = math.exp(-age_hours / 24)
            return max(weight, 0.1)  # Minimum weight
        except:
            return 0.5  # Default weight if timestamp parsing fails
    
    def _calculate_diversity_bonus(self, individual: Dict[str, Any]) -> float:
        """Calculate diversity bonus to encourage exploration"""
        if len(self.population) <= 1:
            return 0.0
        
        # Calculate average distance to other individuals
        distances = []
        for other in self.population:
            if other != individual:
                similarity = self._calculate_parameter_similarity(individual, other)
                distance = 1 - similarity
                distances.append(distance)
        
        if distances:
            avg_distance = np.mean(distances)
            # Bonus for being different (up to 0.1)
            return min(avg_distance * 0.2, 0.1)
        
        return 0.0
    
    def _calculate_parameter_quality_bonus(self, individual: Dict[str, Any]) -> float:
        """Calculate bonus for good parameter combinations"""
        bonus = 0.0
        
        # Reward balanced temperature and top_p
        temp = individual.get('temperature', 0.7)
        top_p = individual.get('top_p', 0.9)
        if 0.3 <= temp <= 0.8 and 0.8 <= top_p <= 0.95:
            bonus += 0.05
        
        # Reward appropriate explanation depth and technical level balance
        exp_depth = individual.get('explanation_depth', 0.5)
        tech_level = individual.get('technical_level', 0.5)
        if abs(exp_depth - tech_level) < 0.3:  # Balanced
            bonus += 0.03
        
        # Reward moderate penalties (avoid extremes)
        freq_pen = individual.get('frequency_penalty', 0.0)
        pres_pen = individual.get('presence_penalty', 0.0)
        if 0.1 <= freq_pen <= 0.5 and 0.1 <= pres_pen <= 0.4:
            bonus += 0.02
        
        return bonus
    
    def evolve_parameters(self) -> Dict[str, Any]:
        """Evolve the parameter population using enhanced genetic algorithm with stronger selection pressure"""
        if len(self.population) < 2:
            return self._initialize_population()[0]
        
        # Calculate fitness for all individuals with enhanced quality focus
        fitness_scores = []
        quality_performances = []
        
        for individual in self.population:
            fitness = self._calculate_fitness(individual)
            fitness_scores.append(fitness)
            
            # Track quality performance for each individual
            quality_perf = self._evaluate_individual_quality_performance(individual)
            quality_performances.append(quality_perf)
        
        if not any(fitness_scores):
            return self.population[0]
        
        # Enhanced selection with stronger pressure for quality improvement
        new_population = []
        
        # 1. Keep best individual (elitism)
        best_idx = np.argmax(fitness_scores)
        best_individual = self.population[best_idx].copy()
        new_population.append(best_individual)
        
        # 2. Generate new individuals through crossover and mutation
        while len(new_population) < self.population_size:
            # Tournament selection for parents
            parent1 = self._tournament_selection(fitness_scores)
            parent2 = self._tournament_selection(fitness_scores)
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation with adaptive rates based on quality performance
            child = self._adaptive_mutation_enhanced(child, quality_performances)
            
            new_population.append(child)
        
        # Update population and increment generation
        self.population = new_population
        self.generation += 1
        
        # Update fitness scores for new population
        self.fitness_scores = [self._calculate_fitness(ind) for ind in self.population]
        
        # Update adaptive mutation rate based on progress
        self._update_adaptive_mutation_rate()
        
        # Print evolution progress
        best_idx = np.argmax(self.fitness_scores)
        best_fitness = self.fitness_scores[best_idx]
        print(f"Generation {self.generation}: Best fitness = {best_fitness:.4f}")
        print(f"Best parameters: {self.population[best_idx]}")
        
        # Return best individual from new population
        return self.population[best_idx].copy()
    
    def _update_adaptive_mutation_rate(self):
        """Update mutation rate - KEEP HIGH for better exploration"""
        if not self.fitness_scores:
            return
            
        current_best = max(self.fitness_scores)
        self.best_fitness_history.append(current_best)
        
        # Keep only recent history
        if len(self.best_fitness_history) > 10:
            self.best_fitness_history = self.best_fitness_history[-10:]
        
        # MAINTAIN HIGH mutation rate for continuous exploration
        if len(self.best_fitness_history) >= 5:
            recent_improvement = (self.best_fitness_history[-1] - 
                                self.best_fitness_history[-5])
            
            if recent_improvement < 0.01:  # Stagnation
                # INCREASE mutation rate significantly
                self.adaptive_mutation_rate = min(0.9, self.adaptive_mutation_rate * 1.3)
            else:
                # Only slightly decrease mutation rate (keep exploration high)
                self.adaptive_mutation_rate = max(0.4, self.adaptive_mutation_rate * 0.98)

    def _adaptive_mutation_enhanced(self, individual: Dict[str, Any], quality_performances: List[Dict[str, float]]) -> Dict[str, Any]:
        """ENHANCED adaptive mutation that GUARANTEES parameter changes"""
        mutated = individual.copy()
        
        # FORCE mutation of at least 2-3 parameters to ensure changes
        params_to_mutate = list(self.parameter_ranges.keys())
        random.shuffle(params_to_mutate)
        
        # GUARANTEE mutation of critical parameters
        critical_params = ['temperature', 'explanation_depth', 'technical_level']
        forced_mutations = random.sample(critical_params, min(2, len(critical_params)))
        
        # Add some random parameters
        additional_mutations = random.sample(
            [p for p in params_to_mutate if p not in forced_mutations], 
            min(2, len(params_to_mutate) - len(forced_mutations))
        )
        
        params_to_mutate = forced_mutations + additional_mutations
        
        # STRONG mutation for guaranteed changes
        mutation_strength = 0.25
        
        for param in params_to_mutate:
            param_range = self.parameter_ranges[param]
            current_val = mutated.get(param, (param_range[0] + param_range[1]) / 2)
            
            if param in ['system_prompt_style', 'response_format']:
                # Integer parameters - force change
                options = list(range(int(param_range[0]), int(param_range[1]) + 1))
                if current_val in options:
                    options.remove(current_val)  # Remove current value to force change
                if options:
                    mutated[param] = random.choice(options)
            else:
                # Float parameters - significant mutation
                range_size = param_range[1] - param_range[0]
                mutation_amount = random.uniform(-mutation_strength * range_size, 
                                               mutation_strength * range_size)
                
                new_val = current_val + mutation_amount
                
                # Ensure bounds
                new_val = max(param_range[0], min(param_range[1], new_val))
                
                # FORCE significant change (at least 8% of range)
                min_change = range_size * 0.08
                if abs(new_val - current_val) < min_change:
                    if new_val < (param_range[0] + param_range[1]) / 2:
                        new_val = min(param_range[1], current_val + min_change)
                    else:
                        new_val = max(param_range[0], current_val - min_change)
                
                mutated[param] = new_val
        
        return mutated
    
    def _evaluate_individual_quality_performance(self, individual: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate how well an individual performs on each quality dimension"""
        # Find similar experiences to estimate performance
        similar_experiences = []
        for exp in self.experience_buffer:
            similarity = self._calculate_parameter_similarity(individual, exp['parameters'])
            if similarity > 0.3:  # Lower threshold for more data
                similar_experiences.append((exp, similarity))
        
        if not similar_experiences:
            # Return neutral performance if no similar experiences
            return {dim: 0.5 for dim in ['relevance', 'helpfulness', 'clarity', 'completeness', 'engagement']}
        
        # Weight experiences by similarity and calculate average performance
        dimension_scores = {dim: [] for dim in ['relevance', 'helpfulness', 'clarity', 'completeness', 'engagement']}
        
        for exp, similarity in similar_experiences:
            quality_dims = exp['quality_scores']
            for dim in dimension_scores:
                if dim in quality_dims:
                    weighted_score = quality_dims[dim] * similarity
                    dimension_scores[dim].append(weighted_score)
        
        # Calculate weighted averages
        performance = {}
        for dim, scores in dimension_scores.items():
            if scores:
                performance[dim] = np.mean(scores)
            else:
                performance[dim] = 0.5
        
        return performance
    
    def _select_quality_elites(self, fitness_scores: List[float], quality_performances: List[Dict[str, float]], count: int) -> List[int]:
        """Select elite individuals with bias towards quality improvement"""
        # Calculate composite scores combining fitness and critical dimension performance
        composite_scores = []
        critical_dims = ['relevance', 'helpfulness', 'clarity']
        
        for i, (fitness, quality_perf) in enumerate(zip(fitness_scores, quality_performances)):
            # Base fitness score
            composite = fitness
            
            # Add bonus for strong performance in critical dimensions
            critical_avg = np.mean([quality_perf.get(dim, 0.5) for dim in critical_dims])
            if critical_avg > 0.6:
                composite += (critical_avg - 0.6) * 2.0  # Strong bonus for good critical performance
            
            # Add bonus for balanced performance across all dimensions
            all_scores = list(quality_perf.values())
            if all_scores:
                balance_score = 1.0 - (max(all_scores) - min(all_scores))  # Higher when more balanced
                composite += balance_score * 0.5
            
            composite_scores.append(composite)
        
        # Select top performers
        elite_indices = np.argsort(composite_scores)[-count:]
        return elite_indices.tolist()
    
    def _enhanced_tournament_selection(self, fitness_scores: List[float], quality_performances: List[Dict[str, float]]) -> Dict[str, Any]:
        """Enhanced tournament selection with quality bias"""
        tournament_size = min(5, len(self.population))  # Larger tournament for stronger selection pressure
        candidates = np.random.choice(len(self.population), tournament_size, replace=False)
        
        best_idx = candidates[0]
        best_score = fitness_scores[best_idx]
        best_quality = quality_performances[best_idx]
        
        for idx in candidates[1:]:
            candidate_fitness = fitness_scores[idx]
            candidate_quality = quality_performances[idx]
            
            # Compare based on fitness and critical dimension performance
            critical_dims = ['relevance', 'helpfulness', 'clarity']
            candidate_critical = np.mean([candidate_quality.get(dim, 0.5) for dim in critical_dims])
            best_critical = np.mean([best_quality.get(dim, 0.5) for dim in critical_dims])
            
            # Prefer candidates with better critical dimension performance
            if candidate_critical > best_critical + 0.05:  # Significant improvement in critical dims
                best_idx = idx
                best_score = candidate_fitness
                best_quality = candidate_quality
            elif abs(candidate_critical - best_critical) < 0.05 and candidate_fitness > best_score:
                # Similar critical performance, prefer higher fitness
                best_idx = idx
                best_score = candidate_fitness
                best_quality = candidate_quality
        
        return self.population[best_idx]
    
    def _quality_focused_crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced crossover that preserves quality-improving parameter combinations"""
        child = {}
        
        # Evaluate which parent performs better on each dimension
        perf1 = self._evaluate_individual_quality_performance(parent1)
        perf2 = self._evaluate_individual_quality_performance(parent2)
        
        for param in self.parameter_ranges.keys():
            if param in parent1 and param in parent2:
                # For parameters that strongly affect critical dimensions, choose from better parent
                if param in ['temperature', 'explanation_depth', 'technical_level', 'top_p']:
                    critical_dims = ['relevance', 'helpfulness', 'clarity']
                    avg_perf1 = np.mean([perf1.get(dim, 0.5) for dim in critical_dims])
                    avg_perf2 = np.mean([perf2.get(dim, 0.5) for dim in critical_dims])
                    
                    if avg_perf1 > avg_perf2:
                        child[param] = parent1[param]
                    else:
                        child[param] = parent2[param]
                else:
                    # For other parameters, use standard crossover
                    if np.random.random() < 0.5:
                        child[param] = parent1[param]
                    else:
                        child[param] = parent2[param]
        
        return child
    
    def _adaptive_mutation(self, individual: Dict[str, Any], quality_performances: List[Dict[str, float]]) -> Dict[str, Any]:
        """Adaptive mutation that targets underperforming quality dimensions"""
        mutated = individual.copy()
        
        # Identify underperforming dimensions
        individual_perf = self._evaluate_individual_quality_performance(individual)
        critical_dims = ['relevance', 'helpfulness', 'clarity']
        
        # Calculate which dimensions need improvement
        underperforming_dims = []
        for dim in critical_dims:
            if individual_perf.get(dim, 0.5) < 0.6:  # Below acceptable threshold
                underperforming_dims.append(dim)
        
        # Apply targeted mutations based on underperforming dimensions
        mutation_rate = 0.3 if underperforming_dims else 0.1  # Higher mutation for underperforming
        
        for param in self.parameter_ranges.keys():
            if np.random.random() < mutation_rate:
                param_range = self.parameter_ranges[param]
                current_val = mutated.get(param, (param_range[0] + param_range[1]) / 2)
                
                # Determine mutation strategy based on underperforming dimensions
                if 'relevance' in underperforming_dims and param in ['temperature', 'top_p']:
                    # For relevance: prefer moderate temperature (0.3-0.7) and high top_p (0.8-0.95)
                    if param == 'temperature':
                        target_range = (0.3, 0.7)
                    else:  # top_p
                        target_range = (0.8, 0.95)
                    
                    # Mutate towards target range
                    target_center = (target_range[0] + target_range[1]) / 2
                    mutation_strength = 0.2
                    direction = 1 if target_center > current_val else -1
                    mutation = direction * np.random.uniform(0, mutation_strength)
                    
                elif 'helpfulness' in underperforming_dims and param in ['explanation_depth', 'technical_level']:
                    # For helpfulness: prefer higher explanation depth and appropriate technical level
                    if param == 'explanation_depth':
                        target_val = 0.7  # Higher depth for more helpful responses
                    else:  # technical_level
                        target_val = 0.6  # Moderate technical level
                    
                    mutation_strength = 0.15
                    direction = 1 if target_val > current_val else -1
                    mutation = direction * np.random.uniform(0, mutation_strength)
                    
                elif 'clarity' in underperforming_dims and param in ['temperature', 'explanation_depth']:
                    # For clarity: prefer lower temperature and moderate explanation depth
                    if param == 'temperature':
                        target_val = 0.4  # Lower temperature for clearer responses
                    else:  # explanation_depth
                        target_val = 0.6  # Moderate depth for clarity
                    
                    mutation_strength = 0.15
                    direction = 1 if target_val > current_val else -1
                    mutation = direction * np.random.uniform(0, mutation_strength)
                    
                else:
                    # Standard mutation for other parameters
                    mutation_strength = 0.1
                    mutation = np.random.uniform(-mutation_strength, mutation_strength)
                
                # Apply mutation with bounds checking
                new_val = current_val + mutation
                new_val = max(param_range[0], min(param_range[1], new_val))
                mutated[param] = new_val
        
        return mutated

    def get_best_parameters(self) -> Dict[str, Any]:
        """Get the best parameters from the current population"""
        if not self.population:
            return self._initialize_population()[0]
        
        # Calculate fitness for all individuals
        fitness_scores = [self._calculate_fitness(individual) for individual in self.population]
        
        if not any(fitness_scores):
            return self.population[0]
        
        # Return the individual with highest fitness
        best_index = np.argmax(fitness_scores)
        return self.population[best_index].copy()

    def _tournament_selection(self, fitness_scores: List[float], tournament_size: int = 3) -> Dict[str, Any]:
        """Select individual using tournament selection"""
        tournament_indices = random.sample(range(len(self.population)), 
                                         min(tournament_size, len(self.population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced crossover operation with parameter-aware blending - returns single child"""
        child = parent1.copy()
        
        for param in self.parameter_ranges.keys():
            if param in parent1 and param in parent2:
                if random.random() < 0.5:
                    # Take parameter from parent2
                    child[param] = parent2[param]
                else:
                    # Blend parameters (for continuous values)
                    if param not in ['system_prompt_style', 'response_format']:
                        alpha = random.uniform(0.3, 0.7)  # Blend factor
                        val = parent1[param] * alpha + parent2[param] * (1 - alpha)
                        
                        # Ensure values are within bounds
                        min_val, max_val = self.parameter_ranges[param]
                        child[param] = max(min_val, min(max_val, val))
        
        return child
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mutation to an individual"""
        mutated = individual.copy()
        
        for param, value in mutated.items():
            if random.random() < self.adaptive_mutation_rate:
                if param in self.parameter_ranges:
                    min_val, max_val = self.parameter_ranges[param]
                    
                    if isinstance(value, int):
                        # For integer parameters
                        mutation_strength = max(1, int((max_val - min_val) * 0.1))
                        mutated[param] = max(min_val, min(max_val, 
                                           value + random.randint(-mutation_strength, mutation_strength)))
                    else:
                        # For float parameters
                        mutation_strength = (max_val - min_val) * 0.1
                        mutated[param] = max(min_val, min(max_val, 
                                           value + random.uniform(-mutation_strength, mutation_strength)))
        
        return mutated

    def save_evolution_state(self, filepath: Optional[str] = None) -> Optional[str]:
        """Save the current evolution state to file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"llm_evolution_state_{timestamp}.json"
        
        state_data = {
            'timestamp': datetime.now().isoformat(),
            'population_size': self.population_size,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'elite_ratio': self.elite_ratio,
            'adaptive_mutation_rate': self.adaptive_mutation_rate,
            'parameter_ranges': self.parameter_ranges,
            'population': self.population,
            'fitness_scores': self.fitness_scores,
            'experience_buffer': list(self.experience_buffer),
            'best_fitness_history': self.best_fitness_history,
            'generation': getattr(self, 'generation', 0)
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            print(f"LLM Evolution state saved to {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving LLM evolution state: {e}")
            return None

def main():
    """
    Test the LLM Evolution System
    """
    print("Testing LLM Evolution System...")
    
    # This would normally import and use the LLMResponseAgent
    # For testing, we'll create a mock agent
    class MockAgent:
        def __init__(self):
            self.evolution_params = {
                'temperature': 0.7,
                'max_tokens': 500,
                'top_p': 1.0,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0
            }
            self.evolution_step = 0
        
        def update_evolution_params(self, params):
            self.evolution_params.update(params)
        
        def set_evolution_step(self, step):
            self.evolution_step = step
    
    # Initialize mock agent and evolution system
    agent = MockAgent()
    evolution_system = LLMEvolutionSystem(agent, population_size=6)
    
    # Simulate some experiences
    print("\nSimulating experiences...")
    
    experiences = [
        ("I don't understand neural networks", 0.6),
        ("URGENT: Code not working!", 0.8),
        ("Explain machine learning", 0.7),
        ("What is the deadline?", 0.5),
        ("Help with debugging", 0.9),
        ("Simple question about AI", 0.4)
    ]
    
    for i, (text, quality) in enumerate(experiences):
        mock_response = {
            'response': f"Mock response {i+1}",
            'urgency_classification': {'final_urgency': 'medium'},
            'content_category': 'learning',
            'evolution_stage': 'basic',
            'evolution_params': agent.evolution_params.copy()
        }
        
        # Create proper quality scores dictionary
        quality_scores = {
            'relevance': quality + np.random.uniform(-0.1, 0.1),
            'helpfulness': quality + np.random.uniform(-0.1, 0.1),
            'clarity': quality + np.random.uniform(-0.1, 0.1),
            'completeness': quality + np.random.uniform(-0.1, 0.1),
            'engagement': quality + np.random.uniform(-0.1, 0.1)
        }
        
        evolution_system.add_experience(mock_response['evolution_params'], quality_scores)
    
    # Run evolution
    print("\nRunning evolution...")
    for generation in range(3):
        best_params = evolution_system.evolve_parameters()
        
        # Simulate more experiences with evolved parameters
        for j in range(2):
            quality = random.uniform(0.5, 0.9)  # Simulate improving quality
            
            # Create proper quality scores dictionary
            quality_scores = {
                'relevance': quality + np.random.uniform(-0.1, 0.1),
                'helpfulness': quality + np.random.uniform(-0.1, 0.1),
                'clarity': quality + np.random.uniform(-0.1, 0.1),
                'completeness': quality + np.random.uniform(-0.1, 0.1),
                'engagement': quality + np.random.uniform(-0.1, 0.1)
            }
            
            evolution_system.add_experience(best_params, quality_scores)
    
    # Print evolution summary
    print("\n=== Evolution Summary ===")
    summary = evolution_system.get_evolution_summary()
    print(f"Generations: {summary['generation']}")
    print(f"Best Fitness: {summary['best_fitness']:.4f}")
    print(f"Improvement Rate: {summary['improvement_rate']:.4f}")
    print(f"Convergence Status: {summary['convergence_status']}")
    print(f"Experience Count: {summary['experience_count']}")


if __name__ == "__main__":
    main()
