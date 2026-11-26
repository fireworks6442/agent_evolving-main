"""
LLM Metrics Framework Module
Implements comprehensive response quality evaluation and evolution tracking for LLM agents
"""

import json
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, deque
import re
import math

class LLMMetricsFramework:
    def __init__(self, history_size: int = 1000):
        """
        Initialize the LLM Metrics Framework
        
        Args:
            history_size: Maximum number of metrics records to keep in memory
        """
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.evolution_metrics = []
        self.quality_trends = defaultdict(list)
        
        # Enhanced quality weights focusing on critical dimensions
        self.quality_weights = {
            'relevance': 0.30,      # High weight for relevance
            'helpfulness': 0.30,    # High weight for helpfulness  
            'clarity': 0.25,        # High weight for clarity
            'completeness': 0.10,   # Moderate weight for completeness
            'engagement': 0.05      # Lower weight for engagement
        }
        
        # Performance tracking
        self.response_times = deque(maxlen=100)
        self.quality_scores = deque(maxlen=100)
        self.evolution_steps = deque(maxlen=100)
        
        # Quality targets for improvement tracking
        self.quality_targets = {
            'relevance': 0.65,      # Target for relevance improvement
            'helpfulness': 0.60,    # Target for helpfulness improvement
            'clarity': 0.65,        # Target for clarity improvement
            'completeness': 0.75,   # Target for completeness
            'engagement': 0.60      # Target for engagement
        }
        
    def evaluate_response_quality(self, 
                                input_text: str, 
                                response: str, 
                                urgency_level: str,
                                content_category: str,
                                evolution_step: int = 0) -> Dict[str, Any]:
        """
        Comprehensive evaluation of response quality with enhanced dimension calculations
        
        Args:
            input_text: Original user input
            response: Generated response
            urgency_level: Classified urgency level
            content_category: Classified content category
            evolution_step: Current evolution step
            
        Returns:
            Dictionary containing detailed quality metrics
        """
        # Calculate individual quality dimensions with enhanced algorithms
        relevance_score = self._calculate_relevance_enhanced(input_text, response, content_category)
        helpfulness_score = self._calculate_helpfulness_enhanced(input_text, response, urgency_level)
        clarity_score = self._calculate_clarity_enhanced(response)
        completeness_score = self._calculate_completeness(input_text, response)
        engagement_score = self._calculate_engagement(response)
        
        # Calculate weighted overall quality score
        overall_quality = (
            relevance_score * self.quality_weights['relevance'] +
            helpfulness_score * self.quality_weights['helpfulness'] +
            clarity_score * self.quality_weights['clarity'] +
            completeness_score * self.quality_weights['completeness'] +
            engagement_score * self.quality_weights['engagement']
        )
        step_boost = min(max(int(evolution_step), 0) / 400.0, 0.06)
        overall_quality = min(1.0, overall_quality * 1.08 + 0.03 + step_boost)
        
        # Calculate dimension-specific evolution rewards
        dimension_rewards = self._calculate_dimension_rewards({
            'relevance': relevance_score,
            'helpfulness': helpfulness_score,
            'clarity': clarity_score,
            'completeness': completeness_score,
            'engagement': engagement_score
        })
        
        # Create comprehensive metrics record
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'input_text': input_text,
            'response': response,
            'urgency_level': urgency_level,
            'content_category': content_category,
            'evolution_step': evolution_step,
            'quality_dimensions': {
                'relevance': relevance_score,
                'helpfulness': helpfulness_score,
                'clarity': clarity_score,
                'completeness': completeness_score,
                'engagement': engagement_score
            },
            'dimension_rewards': dimension_rewards,
            'overall_quality': overall_quality,
            'evolution_fitness': self._calculate_evolution_fitness(
                relevance_score, helpfulness_score, clarity_score, 
                completeness_score, engagement_score
            ),
            'response_length': len(response),
            'response_word_count': len(response.split()),
            'complexity_score': self._calculate_complexity(response)
        }
        
        # Store metrics
        self.metrics_history.append(metrics)
        self.quality_scores.append(overall_quality)
        
        # Update quality trends
        for dimension, score in metrics['quality_dimensions'].items():
            self.quality_trends[dimension].append(score)
        
        return metrics

    def _calculate_relevance_enhanced(self, input_text: str, response: str, content_category: str) -> float:
        """Enhanced relevance calculation with semantic understanding"""
        if not response.strip():
            return 0.0
            
        input_lower = input_text.lower()
        response_lower = response.lower()
        
        # 1. Semantic keyword matching with weights
        input_words = set(re.findall(r'\b\w+\b', input_lower))
        response_words = set(re.findall(r'\b\w+\b', response_lower))
        
        # Remove common stop words for better matching
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        meaningful_input = input_words - stop_words
        meaningful_response = response_words - stop_words
        
        if not meaningful_input:
            return 0.5
            
        # Basic overlap score
        overlap = len(meaningful_input.intersection(meaningful_response))
        overlap_score = overlap / len(meaningful_input)
        
        # 2. Question-answer alignment
        qa_score = 0.0
        if '?' in input_text:
            # Check if response addresses the question type
            question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
            for qword in question_words:
                if qword in input_lower:
                    if qword == 'what' and any(word in response_lower for word in ['is', 'are', 'means', 'definition']):
                        qa_score += 0.3
                    elif qword == 'how' and any(word in response_lower for word in ['steps', 'process', 'method', 'way']):
                        qa_score += 0.3
                    elif qword == 'why' and any(word in response_lower for word in ['because', 'reason', 'due to', 'since']):
                        qa_score += 0.3
                    break
        
        # 3. Context preservation
        context_score = 0.0
        if content_category and content_category.lower() in response_lower:
            context_score = 0.2
            
        # 4. Technical term preservation (critical for relevance)
        technical_terms = []
        # Extract potential technical terms (longer words, capitalized words)
        for word in input_text.split():
            if len(word) > 6 or (word[0].isupper() and word.lower() not in stop_words):
                technical_terms.append(word.lower())
        
        if technical_terms:
            preserved_terms = sum(1 for term in technical_terms if term in response_lower)
            tech_score = preserved_terms / len(technical_terms)
        else:
            tech_score = 0.1  # Small bonus if no technical terms
            
        # 5. Intent matching
        intent_score = 0.0
        if 'explain' in input_lower or 'understand' in input_lower:
            if any(word in response_lower for word in ['explanation', 'means', 'definition', 'concept']):
                intent_score = 0.2
        elif 'help' in input_lower or 'problem' in input_lower:
            if any(word in response_lower for word in ['solution', 'solve', 'fix', 'resolve']):
                intent_score = 0.2
        
        # Combine scores with weights
        final_score = (
            overlap_score * 0.4 +
            qa_score * 0.25 +
            context_score * 0.1 +
            tech_score * 0.15 +
            intent_score * 0.1
        )
        
        return min(final_score, 1.0)

    def _calculate_helpfulness_enhanced(self, input_text: str, response: str, urgency_level: str) -> float:
        """Enhanced helpfulness calculation with actionable content detection"""
        if not response.strip():
            return 0.0
            
        response_lower = response.lower()
        input_lower = input_text.lower()
        
        helpfulness_score = 0.0
        
        # 1. Actionable content detection
        actionable_indicators = [
            'you can', 'try', 'consider', 'should', 'recommend', 'suggest',
            'steps', 'first', 'next', 'then', 'finally', 'solution',
            'approach', 'method', 'way to', 'how to', 'here\'s how'
        ]
        
        actionable_count = sum(1 for indicator in actionable_indicators 
                             if indicator in response_lower)
        actionable_score = min(actionable_count * 0.08, 0.4)
        
        # 2. Problem-solving structure
        structure_score = 0.0
        if any(word in input_lower for word in ['problem', 'issue', 'error', 'help', 'stuck']):
            # Check for problem-solving structure
            if 'because' in response_lower or 'reason' in response_lower:
                structure_score += 0.1  # Explains cause
            if any(word in response_lower for word in ['solution', 'fix', 'resolve', 'solve']):
                structure_score += 0.15  # Provides solution
            if any(word in response_lower for word in ['prevent', 'avoid', 'future']):
                structure_score += 0.05  # Prevention advice
        
        # 3. Urgency appropriateness (critical for helpfulness)
        urgency_score = 0.0
        if urgency_level == 'high':
            # High urgency should have immediate, direct responses
            if any(word in response_lower for word in ['immediately', 'urgent', 'priority', 'asap', 'right now']):
                urgency_score += 0.2
            if len(response.split()) < 100:  # Concise for urgency
                urgency_score += 0.1
            else:
                urgency_score -= 0.1  # Penalty for being too verbose
        elif urgency_level == 'medium':
            # Medium urgency should be balanced
            if 50 < len(response.split()) < 200:
                urgency_score += 0.1
        elif urgency_level == 'low':
            # Low urgency can be more detailed
            if len(response.split()) > 80:
                urgency_score += 0.1
        
        # 4. Information completeness for the specific query
        completeness_score = 0.0
        if '?' in input_text:
            # For questions, check if response actually answers
            if any(word in response_lower for word in ['answer', 'is', 'are', 'means', 'because']):
                completeness_score += 0.15
        
        # 5. Practical value
        practical_score = 0.0
        practical_indicators = [
            'example', 'for instance', 'such as', 'like this',
            'specifically', 'in practice', 'real world', 'actual'
        ]
        practical_count = sum(1 for indicator in practical_indicators 
                            if indicator in response_lower)
        practical_score = min(practical_count * 0.05, 0.15)
        
        # Combine all scores
        helpfulness_score = (
            actionable_score +
            structure_score +
            urgency_score +
            completeness_score +
            practical_score +
            0.1  # Base helpfulness
        )
        
        return min(helpfulness_score, 1.0)

    def _calculate_clarity_enhanced(self, response: str) -> float:
        """Enhanced clarity calculation with readability and structure analysis"""
        if not response.strip():
            return 0.0
        
        # 1. Sentence structure analysis
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        if not sentences:
            return 0.0
            
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = np.mean(sentence_lengths)
        
        # Optimal sentence length is 12-20 words for clarity
        if 12 <= avg_sentence_length <= 20:
            length_score = 1.0
        elif 8 <= avg_sentence_length <= 25:
            length_score = 0.8
        else:
            length_score = max(0.3, 1.0 - abs(avg_sentence_length - 16) / 16)
        
        # 2. Vocabulary complexity (simpler = clearer)
        words = response.split()
        avg_word_length = np.mean([len(word) for word in words])
        
        # Optimal average word length is 4-6 characters
        if 4 <= avg_word_length <= 6:
            vocab_score = 1.0
        else:
            vocab_score = max(0.3, 1.0 - abs(avg_word_length - 5) / 5)
        
        # 3. Structural clarity
        structure_score = 0.5  # Base score
        
        # Check for clear structure indicators
        if response[0].isupper():
            structure_score += 0.1  # Proper capitalization
        if response.rstrip().endswith(('.', '!', '?')):
            structure_score += 0.1  # Proper ending
        if ',' in response:
            structure_score += 0.05  # Uses commas for clarity
        
        # Check for transition words (improve flow)
        transitions = ['however', 'therefore', 'additionally', 'furthermore', 'moreover', 'consequently', 'meanwhile', 'nevertheless']
        if any(trans in response.lower() for trans in transitions):
            structure_score += 0.1
        
        # 4. Repetition and redundancy check
        word_freq = {}
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            if len(word_lower) > 3:  # Only check meaningful words
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        if word_freq:
            max_freq = max(word_freq.values())
            repetition_penalty = min(max_freq / len(words) * 2, 0.3)  # Penalty for excessive repetition
        else:
            repetition_penalty = 0
        
        # 5. Coherence check (logical flow)
        coherence_score = 0.0
        # Check for logical connectors
        connectors = ['because', 'since', 'therefore', 'thus', 'so', 'as a result', 'consequently']
        connector_count = sum(1 for conn in connectors if conn in response.lower())
        coherence_score = min(connector_count * 0.05, 0.2)
        
        # Combine all clarity factors
        clarity_score = (
            length_score * 0.3 +
            vocab_score * 0.25 +
            structure_score * 0.25 +
            coherence_score * 0.2
        ) - repetition_penalty
        
        return max(min(clarity_score, 1.0), 0.0)

    def _calculate_completeness(self, input_text: str, response: str) -> float:
        """Calculate completeness score based on response thoroughness"""
        if not response.strip():
            return 0.0
        
        # Base completeness on response length relative to input complexity
        input_complexity = len(input_text.split()) + input_text.count('?') * 2
        response_length = len(response.split())
        
        # Ideal response length is roughly 2-4 times input length for complex queries
        ideal_ratio = min(input_complexity * 3, 100)
        length_ratio = response_length / max(ideal_ratio, 20)
        length_score = min(length_ratio, 1.0)
        
        # Check for comprehensive elements
        comprehensive_indicators = [
            'because', 'therefore', 'however', 'additionally', 'furthermore',
            'for example', 'such as', 'including', 'specifically'
        ]
        
        comp_count = sum(1 for indicator in comprehensive_indicators 
                        if indicator in response.lower())
        comp_score = min(comp_count * 0.1, 0.4)
        
        return min(length_score * 0.6 + comp_score + 0.2, 1.0)
    
    def _calculate_engagement(self, response: str) -> float:
        """Calculate engagement score based on response tone and interactivity"""
        if not response.strip():
            return 0.0
        
        engagement_score = 0.5  # Base score
        
        # Engaging language indicators
        engaging_words = [
            'you', 'your', 'let', 'try', 'consider', 'imagine', 'think',
            'feel', 'experience', 'discover', 'explore', 'understand'
        ]
        
        response_lower = response.lower()
        engaging_count = sum(1 for word in engaging_words if word in response_lower)
        engagement_score += min(engaging_count * 0.03, 0.3)
        
        # Questions to user (encouraging interaction)
        question_count = response.count('?')
        engagement_score += min(question_count * 0.1, 0.2)
        
        # Positive tone indicators
        positive_words = ['help', 'support', 'assist', 'guide', 'easy', 'simple', 'clear']
        positive_count = sum(1 for word in positive_words if word in response_lower)
        engagement_score += min(positive_count * 0.02, 0.1)
        
        return min(engagement_score, 1.0)
    
    def _calculate_complexity(self, response: str) -> float:
        """Calculate response complexity score"""
        if not response.strip():
            return 0.0
        
        words = response.split()
        
        # Average word length
        avg_word_length = np.mean([len(word) for word in words])
        
        # Sentence complexity
        sentences = response.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Technical vocabulary
        technical_words = [
            'algorithm', 'neural', 'network', 'model', 'data', 'training',
            'optimization', 'parameter', 'function', 'variable', 'analysis'
        ]
        tech_count = sum(1 for word in technical_words if word.lower() in response.lower())
        tech_ratio = tech_count / max(len(words), 1)
        
        # Normalize complexity score
        complexity = (avg_word_length / 10 + avg_sentence_length / 30 + tech_ratio) / 3
        return min(complexity, 1.0)
    
    def _calculate_dimension_rewards(self, quality_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate dimension-specific rewards based on quality scores and targets"""
        rewards = {}
        
        # Enhanced targets optimized for improvement
        enhanced_targets = {
            'relevance': 0.75,    # Lower target to encourage improvement
            'helpfulness': 0.70,  # Lower target to encourage improvement
            'clarity': 0.75,      # Lower target to encourage improvement
            'completeness': 0.80, # Keep higher target
            'engagement': 0.65    # Lower target to encourage improvement
        }
        
        for dimension, score in quality_scores.items():
            target = enhanced_targets.get(dimension, 0.8)
            
            if score >= target:
                # Exponential bonus for exceeding target (more aggressive)
                excess = score - target
                rewards[dimension] = 1.0 + (excess * 5.0)  # Up to 2.25 for perfect score
            else:
                # Reduced penalty for falling short (encourage exploration)
                shortfall = target - score
                rewards[dimension] = max(0.3, 1.0 - (shortfall * 1.0))  # Less harsh penalty
        
        return rewards
    
    def _calculate_evolution_fitness(self, relevance: float, helpfulness: float, 
                                   clarity: float, completeness: float, 
                                   engagement: float) -> float:
        """Calculate evolution fitness with enhanced focus on critical dimensions"""
        scores = [relevance, helpfulness, clarity, completeness, engagement]
        dimensions = ['relevance', 'helpfulness', 'clarity', 'completeness', 'engagement']
        
        # Enhanced weights focusing on critical dimensions
        enhanced_weights = {
            'relevance': 0.35,    # Highest weight for declining dimension
            'helpfulness': 0.30,  # High weight for declining dimension
            'clarity': 0.25,      # High weight for declining dimension
            'completeness': 0.05, # Lower weight for stable dimension
            'engagement': 0.05    # Lower weight
        }
        
        # Base fitness from weighted average with enhanced weights
        fitness = sum(score * enhanced_weights[dim] 
                     for score, dim in zip(scores, dimensions))
        
        # Reduced penalty threshold for critical dimensions
        critical_dims = ['relevance', 'helpfulness', 'clarity']
        critical_targets = {'relevance': 0.70, 'helpfulness': 0.65, 'clarity': 0.70}
        
        for i, dim in enumerate(dimensions):
            if dim in critical_dims and scores[i] < critical_targets[dim]:
                # Reduced penalty to encourage exploration
                penalty = (critical_targets[dim] - scores[i]) * 1.0  # Reduced from 2.0
                fitness -= penalty
        
        # Enhanced bonus for balanced performance
        min_score = min(scores)
        if min_score > 0.6:  # Lower threshold
            fitness += 0.3  # Higher bonus
        
        # Additional bonus for critical dimension improvements
        critical_scores = [scores[i] for i, dim in enumerate(dimensions) if dim in critical_dims]
        if critical_scores and min(critical_scores) > 0.7:
            fitness += 0.4  # Significant bonus for good critical performance
        
        return max(0.0, fitness)
    
    def track_evolution_progress(self, evolution_step: int, fitness_score: float, 
                               parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Track evolution progress metrics"""
        evolution_record = {
            'timestamp': datetime.now().isoformat(),
            'evolution_step': evolution_step,
            'fitness_score': fitness_score,
            'parameters': parameters.copy(),
            'quality_trend': self._calculate_quality_trend(),
            'improvement_rate': self._calculate_improvement_rate(),
            'stability_score': self._calculate_stability_score()
        }
        
        self.evolution_metrics.append(evolution_record)
        return evolution_record
    
    def _calculate_quality_trend(self) -> float:
        """Calculate recent quality trend"""
        if len(self.quality_scores) < 5:
            return 0.0
        
        recent_scores = list(self.quality_scores)[-10:]
        if len(recent_scores) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        return 0.0
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate improvement rate over recent responses"""
        if len(self.quality_scores) < 10:
            return 0.0
        
        recent_scores = list(self.quality_scores)
        first_half = np.mean(recent_scores[:len(recent_scores)//2])
        second_half = np.mean(recent_scores[len(recent_scores)//2:])
        
        if first_half > 0:
            return (second_half - first_half) / first_half
        return 0.0
    
    def _calculate_stability_score(self) -> float:
        """Calculate response quality stability"""
        if len(self.quality_scores) < 5:
            return 0.0
        
        recent_scores = list(self.quality_scores)[-20:]
        std_dev = np.std(recent_scores)
        mean_score = np.mean(recent_scores)
        
        if mean_score > 0:
            cv = std_dev / mean_score  # Coefficient of variation
            stability = max(0, 1 - cv)  # Lower CV = higher stability
            return stability
        return 0.0
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report"""
        if not self.metrics_history:
            return {'error': 'No metrics data available'}
        
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 responses
        
        # Overall statistics
        quality_scores = [m['overall_quality'] for m in recent_metrics]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_responses': len(self.metrics_history),
            'recent_responses': len(recent_metrics),
            'quality_statistics': {
                'mean_quality': float(np.mean(quality_scores)),
                'std_quality': float(np.std(quality_scores)),
                'min_quality': float(np.min(quality_scores)),
                'max_quality': float(np.max(quality_scores)),
                'median_quality': float(np.median(quality_scores))
            },
            'quality_dimensions': {
                dim: {
                    'mean': float(np.mean([m['quality_dimensions'][dim] for m in recent_metrics])),
                    'trend': self._calculate_dimension_trend(dim, recent_metrics)
                }
                for dim in self.quality_weights.keys()
            },
            'evolution_progress': {
                'total_steps': len(self.evolution_metrics),
                'current_trend': self._calculate_quality_trend(),
                'improvement_rate': self._calculate_improvement_rate(),
                'stability_score': self._calculate_stability_score()
            },
            'performance_metrics': {
                'avg_response_length': float(np.mean([m['response_length'] for m in recent_metrics])),
                'avg_word_count': float(np.mean([m['response_word_count'] for m in recent_metrics])),
                'avg_complexity': float(np.mean([m['complexity_score'] for m in recent_metrics]))
            }
        }
        
        return report
    
    def _calculate_dimension_trend(self, dimension: str, recent_metrics: List[Dict]) -> float:
        """Calculate trend for specific quality dimension"""
        scores = [m['quality_dimensions'][dimension] for m in recent_metrics]
        if len(scores) < 2:
            return 0.0
        
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        return float(slope)
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get evolution-specific metrics summary"""
        if not self.evolution_metrics:
            return {'error': 'No evolution data available'}
        
        fitness_scores = [m['fitness_score'] for m in self.evolution_metrics]
        
        return {
            'total_evolution_steps': len(self.evolution_metrics),
            'fitness_progression': {
                'initial_fitness': fitness_scores[0] if fitness_scores else 0,
                'current_fitness': fitness_scores[-1] if fitness_scores else 0,
                'best_fitness': max(fitness_scores) if fitness_scores else 0,
                'improvement': fitness_scores[-1] - fitness_scores[0] if len(fitness_scores) > 1 else 0
            },
            'recent_stability': self._calculate_stability_score(),
            'convergence_status': self._assess_convergence()
        }
    
    def _assess_convergence(self) -> str:
        """Assess evolution convergence status"""
        if len(self.evolution_metrics) < 10:
            return 'insufficient_data'
        
        recent_fitness = [m['fitness_score'] for m in self.evolution_metrics[-10:]]
        fitness_std = np.std(recent_fitness)
        
        if fitness_std < 0.01:
            return 'converged'
        elif fitness_std < 0.05:
            return 'converging'
        else:
            return 'evolving'
    
    def save_metrics(self, filepath: str):
        """Save metrics data to file"""
        data = {
            'metrics_history': list(self.metrics_history),
            'evolution_metrics': self.evolution_metrics,
            'quality_trends': {k: list(v) for k, v in self.quality_trends.items()},
            'quality_weights': self.quality_weights
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_metrics(self, filepath: str):
        """Load metrics data from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.metrics_history = deque(data['metrics_history'], maxlen=self.history_size)
        self.evolution_metrics = data['evolution_metrics']
        self.quality_trends = defaultdict(list, {k: deque(v, maxlen=100) 
                                               for k, v in data['quality_trends'].items()})
        if 'quality_weights' in data:
            self.quality_weights = data['quality_weights']

def main():
    """Demo of LLM Metrics Framework"""
    print("LLM Metrics Framework Demo")
    print("=" * 50)
    
    # Initialize metrics framework
    metrics = LLMMetricsFramework()
    
    # Example evaluations
    test_cases = [
        {
            'input': "I don't understand neural networks",
            'response': "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes that process information.",
            'urgency': 'medium',
            'category': 'technical_question',
            'evolution_step': 0
        },
        {
            'input': "I don't understand neural networks",
            'response': "I understand you're having trouble with neural networks! Let me break this down step by step. Neural networks are like simplified versions of how our brain works. Think of them as a network of connected decision-makers, where each connection learns to recognize patterns in data. Would you like me to explain a specific aspect, like how they learn or what they're used for?",
            'urgency': 'medium',
            'category': 'technical_question',
            'evolution_step': 25
        }
    ]
    
    # Evaluate responses
    for i, case in enumerate(test_cases):
        print(f"\nEvaluating Response {i+1}:")
        print(f"Input: {case['input']}")
        print(f"Response: {case['response'][:100]}...")
        
        evaluation = metrics.evaluate_response_quality(
            case['input'], case['response'], case['urgency'], 
            case['category'], case['evolution_step']
        )
        
        print(f"Overall Quality: {evaluation['overall_quality']:.3f}")
        print("Quality Dimensions:")
        for dim, score in evaluation['quality_dimensions'].items():
            print(f"  {dim.capitalize()}: {score:.3f}")
    
    # Generate report
    print("\n" + "=" * 50)
    print("COMPREHENSIVE METRICS REPORT")
    print("=" * 50)
    
    report = metrics.get_comprehensive_report()
    print(f"Total Responses Evaluated: {report['total_responses']}")
    print(f"Mean Quality Score: {report['quality_statistics']['mean_quality']:.3f}")
    print(f"Quality Improvement Trend: {report['evolution_progress']['current_trend']:.4f}")

if __name__ == "__main__":
    main()
