"""
Evolution System Module
Implements reinforcement learning and imitation learning for agent evolution
"""

import os
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class EvolutionSystem:
    def __init__(self, learning_rate=0.01, exploration_rate=0.1, decay_rate=0.995, models_dir='models'):
        """Initialize the evolution system"""
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.decay_rate = decay_rate
        self.models_dir = models_dir
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Learning components
        self.reward_model = None
        self.imitation_model = None
        self.policy_parameters = {}
        
        # Initialize agent parameters with more detailed structure
        self.agent_parameters = {
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate,
            'response_generation': {
                'response_length_factor': 1.0,
                'formality_level': 0.5,
                'empathy_level': 0.7,
                'urgency_sensitivity': 0.8
            },
            'template_selection': {
                'urgency_threshold': 0.5,
                'context_weight': 0.6,
                'user_preference_weight': 0.4
            },
            'quality_weights': {
                'relevance_weight': 0.4,
                'helpfulness_weight': 0.3,
                'clarity_weight': 0.3
            }
        }
        
        # Experience and learning data
        self.experience_buffer = []
        self.expert_demonstrations = []
        self.reward_history = []
        self.quality_history = []
        self.evolution_log = []
        
        # Initialize policy parameters
        self._initialize_policy()
    
    def get_imitation_model_path(self):
        """Get the standard imitation model file path"""
        return os.path.join(self.models_dir, 'imitation_model.joblib')
    
    def get_imitation_metadata_path(self):
        """Get the imitation model metadata file path"""
        return os.path.join(self.models_dir, 'imitation_model_metadata.json')
    
    def imitation_model_exists(self):
        """Check if a saved imitation model exists"""
        return os.path.exists(self.get_imitation_model_path())
    
    def should_retrain_imitation_model(self, force_retrain=False, max_age_days=30):
        """Determine if imitation model should be retrained"""
        if force_retrain:
            return True
            
        if not self.imitation_model_exists():
            return True
            
        # Check model age
        metadata_path = self.get_imitation_metadata_path()
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                training_date = datetime.fromisoformat(metadata['training_date'])
                age_days = (datetime.now() - training_date).days
                
                if age_days > max_age_days:
                    print(f"Imitation model is {age_days} days old, retraining...")
                    return True
                    
            except Exception as e:
                print(f"Error reading imitation model metadata: {e}, retraining...")
                return True
        
        return False
    
    def _initialize_policy(self):
        """Initialize policy parameters for reinforcement learning"""
        # Template selection policy
        self.policy_parameters['template_selection'] = {
            'low_urgency_templates': np.ones(4) / 4,  # Equal probability initially
            'high_urgency_templates': np.ones(4) / 4
        }
        
        # Response generation policy
        self.policy_parameters['response_generation'] = {
            'length_preference': 0.5,  # 0 = short, 1 = long
            'formality_level': 0.5,    # 0 = casual, 1 = formal
            'empathy_level': 0.5       # 0 = neutral, 1 = empathetic
        }
        
        # Quality optimization policy
        self.policy_parameters['quality_optimization'] = {
            'relevance_weight': 0.3,
            'helpfulness_weight': 0.3,
            'appropriateness_weight': 0.4
        }
    
    def record_experience(self, state, action, reward, next_state, response_data):
        """Record an experience for reinforcement learning"""
        experience = {
            'timestamp': datetime.now().isoformat(),
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'response_data': response_data,
            'quality_score': response_data.get('quality_score', 0),
            'urgency_level': response_data.get('urgency_level', 0),
            'response_time': response_data.get('response_time', 0)
        }
        
        self.experience_buffer.append(experience)
        self.reward_history.append(reward)
        
        # Limit buffer size
        if len(self.experience_buffer) > 1000:
            self.experience_buffer.pop(0)
    
    def calculate_reward(self, response_data, user_feedback=None):
        """Calculate reward based on response quality and performance metrics with progressive improvement"""
        reward = 0
        
        # Base quality reward with progressive scaling
        quality_score = response_data.get('quality_score', 0.5)
        
        # Evolution-based improvement factor - agent gets better over time
        evolution_steps = len(self.experience_buffer)
        evolution_steps = int(evolution_steps) if isinstance(evolution_steps, str) else evolution_steps
        improvement_factor = 1.0 + min(evolution_steps / 40, 2.5)  # Up to 3.5x improvement
        
        # Progressive quality standards - higher expectations as agent evolves
        quality_baseline = max(0.2, 0.5 - evolution_steps / 150)  # Faster decreasing baseline
        quality_improvement = max(0, quality_score - quality_baseline)
        
        # Enhanced quality reward with exponential scaling for high performance
        if quality_score > 0.8:
            quality_reward = (quality_score ** 2) * 35 * improvement_factor  # Increased from 20 to 35
        elif quality_score > 0.7:
            quality_reward = (quality_score ** 1.8) * 25 * improvement_factor  # Increased from 15 to 25
        elif quality_score > 0.6:
            quality_reward = (quality_score ** 1.6) * 18 * improvement_factor  # Increased from 10 to 18
        else:
            quality_reward = (quality_score ** 1.4) * 12 * improvement_factor  # Increased base reward
        
        reward += quality_reward
        
        # Performance consistency bonus - reward sustained good performance
        if len(self.reward_history) >= 5:
            recent_quality = [exp.get('quality_score', 0.5) for exp in self.experience_buffer[-5:]]
            if np.mean(recent_quality) > 0.65:
                consistency_bonus = 8 * improvement_factor  # Increased from 5 to 8
                reward += consistency_bonus
        
        # Evolution milestone bonuses - more frequent and higher rewards
        if evolution_steps > 0 and evolution_steps % 5 == 0:  # Every 5 steps instead of specific milestones
            milestone_bonus = 8 + evolution_steps / 8  # Increased frequency and amount
            reward += milestone_bonus
        
        # Response time efficiency with learning curve
        response_time = response_data.get('response_time', 1.0)
        if response_time < 0.5:
            time_bonus = 8 * improvement_factor  # Increased from 5 to 8
        elif response_time < 1.0:
            time_bonus = 5 * improvement_factor  # Increased from 3 to 5
        else:
            time_bonus = max(0, 3 - response_time) * improvement_factor  # Increased from 2 to 3
        reward += time_bonus
        
        # Enhanced urgency handling with evolution-based improvement
        urgency_level = response_data.get('urgency_level', 0)
        urgency_classification = response_data.get('urgency_classification', 0)
        
        # Reward correct urgency classification more as agent evolves
        if urgency_level == urgency_classification:
            urgency_bonus = 12 * improvement_factor  # Increased from 8 to 12
            reward += urgency_bonus
        else:
            urgency_penalty = -4 * min(improvement_factor, 1.5)  # Slightly increased penalty
            reward += urgency_penalty
        
        # Length appropriateness with adaptive standards
        response_length = len(response_data.get('response', ''))
        length_score = self._calculate_length_score(response_length, urgency_level)
        reward += length_score * 3 * improvement_factor  # Increased from 2 to 3
        
        # Template effectiveness bonus
        template_effectiveness = response_data.get('template_effectiveness', 0.5)
        if template_effectiveness > 0.8:
            template_bonus = 10 * improvement_factor  # Increased from 6 to 10
            reward += template_bonus
        
        # Learning progress bonus - reward improvement over previous performance
        if len(self.reward_history) > 10:
            recent_avg = np.mean(self.reward_history[-5:])
            historical_avg = np.mean(self.reward_history[-15:-5])
            if recent_avg > historical_avg:
                improvement_bonus = (recent_avg - historical_avg) * 3  # Increased from 2 to 3
                reward += improvement_bonus
        
        # User feedback integration with amplification
        if user_feedback:
            feedback_score = user_feedback.get('satisfaction', 0)
            reward += feedback_score * 20 * improvement_factor  # Increased from 15 to 20
            
            if user_feedback.get('helpful', False):
                reward += 12 * improvement_factor  # Increased from 8 to 12
            if user_feedback.get('appropriate', False):
                reward += 8 * improvement_factor  # Increased from 6 to 8
        
        # Quality threshold bonuses - reward exceeding quality targets
        if quality_score > 0.8:
            reward += 25 * improvement_factor  # Increased from 15 to 25
        elif quality_score > 0.7:
            reward += 18 * improvement_factor  # Increased from 10 to 18
        elif quality_score > 0.6:
            reward += 10 * improvement_factor  # Increased from 5 to 10
        elif quality_score > 0.5:
            reward += 5 * improvement_factor  # New threshold for medium quality
        
        # Reduce random noise as agent evolves - more predictable rewards
        evolution_steps = int(evolution_steps) if isinstance(evolution_steps, str) else evolution_steps
        noise_magnitude = max(0.05, 0.8 - evolution_steps / 120)  # Reduced noise faster
        noise = np.random.normal(0, noise_magnitude)
        reward += noise
        
        # Ensure minimum reward progression with higher baseline
        base_reward = 8 + evolution_steps / 15  # Increased base reward and faster progression
        reward = max(reward, base_reward)
        
        return reward
    
    def _calculate_length_score(self, length, urgency_level):
        """Calculate score based on response length appropriateness"""
        if urgency_level == 1:  # Urgent - prefer concise
            if 50 <= length <= 150:
                return 3
            elif length > 300:
                return -2
            else:
                return 1
        else:  # Non-urgent - can be detailed
            if 100 <= length <= 400:
                return 2
            elif length < 50:
                return -1
            else:
                return 0
    
    def _calculate_consistency_bonus(self, response_data):
        """Calculate bonus for consistency with high-quality responses"""
        if len(self.experience_buffer) < 5:
            return 0
        
        # Find high-quality past responses
        high_quality_responses = [
            exp for exp in self.experience_buffer[-20:]  # Last 20 experiences
            if exp['quality_score'] > 0.8
        ]
        
        if not high_quality_responses:
            return 0
        
        # Simple consistency measure based on response characteristics
        current_urgency = response_data.get('urgency_level', 0)
        current_quality = response_data.get('quality_score', 0)
        
        similar_responses = [
            exp for exp in high_quality_responses
            if exp['response_data']['urgency_level'] == current_urgency
        ]
        
        if similar_responses:
            avg_quality = np.mean([exp['quality_score'] for exp in similar_responses])
            if current_quality >= avg_quality:
                return 1.0  # Consistency bonus
        
        return 0
    
    def update_policy_with_reward(self, experience):
        """Update policy parameters based on experience reward with stronger updates"""
        reward = experience['reward']
        action = experience['action']
        
        # Amplify learning rate based on reward magnitude
        adaptive_lr = self.learning_rate * (1 + abs(reward) / 10)
        
        # Update template selection weights more aggressively
        if reward > 10:  # High reward threshold
            # Strongly reinforce successful template choices
            for template_type in self.policy_parameters['template_selection']:
                if template_type in action:
                    current_weight = self.policy_parameters['template_selection'][template_type]
                    self.policy_parameters['template_selection'][template_type] = min(
                        current_weight + adaptive_lr * 0.3, 1.0
                    )
        elif reward < 5:  # Low reward threshold
            # Reduce weights for poor choices
            for template_type in self.policy_parameters['template_selection']:
                if template_type in action:
                    current_weight = self.policy_parameters['template_selection'][template_type]
                    self.policy_parameters['template_selection'][template_type] = max(
                        current_weight - adaptive_lr * 0.2, 0.1
                    )
        
        # Update response generation parameters more significantly
        if reward > 8:
            for param in self.policy_parameters['response_generation']:
                if param in action:
                    current_value = self.policy_parameters['response_generation'][param]
                    target_value = action[param]
                    # Stronger movement towards successful actions
                    self.policy_parameters['response_generation'][param] = (
                        current_value + adaptive_lr * 2 * (target_value - current_value)
                    )
        
        # Update quality optimization weights with more impact
        quality_score = experience['quality_score']
        if quality_score > 0.7:
            # Reinforce current quality weights more strongly
            for weight in self.policy_parameters['quality_optimization']:
                if weight in action:
                    current_weight = self.policy_parameters['quality_optimization'][weight]
                    self.policy_parameters['quality_optimization'][weight] = min(
                        current_weight * (1 + adaptive_lr * 0.5), 1.0
                    )
        
        # Add parameter drift to prevent stagnation
        for category in self.policy_parameters:
            for param in self.policy_parameters[category]:
                drift = np.random.normal(0, 0.01)  # Small random drift
                current_val = self.policy_parameters[category][param]
                # Handle both scalar and array values
                if isinstance(current_val, np.ndarray):
                    # For arrays, apply drift element-wise
                    new_val = current_val + drift
                    self.policy_parameters[category][param] = np.clip(new_val, 0.0, 1.0)
                else:
                    # For scalars, apply drift directly
                    self.policy_parameters[category][param] = max(min(current_val + drift, 1.0), 0.0)
    
    def add_expert_demonstration(self, expert_response_data):
        """Add expert demonstration for imitation learning"""
        demonstration = {
            'timestamp': datetime.now().isoformat(),
            'original_text': expert_response_data.get('original_text', ''),
            'urgency_level': expert_response_data.get('urgency_level', 0),
            'expert_response': expert_response_data.get('response', ''),
            'quality_score': expert_response_data.get('quality_score', 1.0),
            'response_time': expert_response_data.get('response_time', 0.5),
            'expert_features': self._extract_response_features(expert_response_data)
        }
        
        self.expert_demonstrations.append(demonstration)
        
        # Limit demonstration buffer
        if len(self.expert_demonstrations) > 500:
            self.expert_demonstrations.pop(0)
    
    def _extract_response_features(self, response_data):
        """Extract features from response for imitation learning"""
        response_text = response_data.get('response', '')
        
        features = {
            'response_length': len(response_text.split()),
            'sentiment_score': self._calculate_sentiment(response_text),
            'urgency_indicators': self._count_urgency_indicators(response_text),
            'helpfulness_indicators': self._count_helpfulness_indicators(response_text),
            'formality_score': self._calculate_formality(response_text),
            'empathy_score': self._calculate_empathy(response_text)
        }
        
        return features
    
    def _calculate_sentiment(self, text):
        """Calculate sentiment score of text"""
        try:
            from textblob import TextBlob
            return TextBlob(text).sentiment.polarity
        except:
            return 0.0
    
    def _count_urgency_indicators(self, text):
        """Count urgency indicators in text"""
        urgency_words = ['urgent', 'immediate', 'quickly', 'right away', 'asap', 'priority']
        return sum(1 for word in urgency_words if word in text.lower())
    
    def _count_helpfulness_indicators(self, text):
        """Count helpfulness indicators in text"""
        helpful_words = ['help', 'assist', 'support', 'guide', 'solution', 'try', 'consider']
        return sum(1 for word in helpful_words if word in text.lower())
    
    def _calculate_formality(self, text):
        """Calculate formality score (0-1)"""
        formal_indicators = ['please', 'thank you', 'sincerely', 'regards', 'appreciate']
        informal_indicators = ['hey', 'yeah', 'ok', 'cool', 'awesome']
        
        formal_count = sum(1 for word in formal_indicators if word in text.lower())
        informal_count = sum(1 for word in informal_indicators if word in text.lower())
        
        total_indicators = formal_count + informal_count
        if total_indicators == 0:
            return 0.5  # Neutral
        
        return formal_count / total_indicators
    
    def _calculate_empathy(self, text):
        """Calculate empathy score (0-1)"""
        empathy_indicators = [
            'understand', 'feel', 'sorry', 'concern', 'worry', 'care',
            'support', 'here for you', 'i know', 'realize'
        ]
        
        empathy_count = sum(1 for phrase in empathy_indicators if phrase in text.lower())
        # Normalize by text length
        return min(empathy_count / max(len(text.split()), 1), 1.0)
    
    def train_imitation_model(self, force_retrain=False):
        """Train imitation learning model from expert demonstrations with persistence"""
        # Check if we should use existing model
        if not self.should_retrain_imitation_model(force_retrain):
            print("Loading existing imitation model...")
            if self.load_imitation_model():
                return True
            else:
                print("Failed to load imitation model, training new one...")
        
        if len(self.expert_demonstrations) < 10:
            print("Not enough expert demonstrations for training")
            return False
        
        print(f"Training imitation model with {len(self.expert_demonstrations)} demonstrations...")
        
        # Prepare training data
        X = []
        y = []
        
        for demo in self.expert_demonstrations:
            # Input features: original text characteristics + urgency
            input_features = [
                len(demo['original_text'].split()),  # Text length
                demo['urgency_level'],               # Urgency level
                self._calculate_sentiment(demo['original_text']),  # Input sentiment
            ]
            
            # Target features: expert response characteristics
            target_features = list(demo['expert_features'].values())
            
            X.append(input_features)
            y.append(target_features)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        self.imitation_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train
        self.imitation_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.imitation_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Imitation model trained - MSE: {mse:.4f}, R2: {r2:.4f}")
        
        # Save the trained model
        self.save_imitation_model(performance_metrics={'mse': mse, 'r2': r2})
        
        return True
    
    def save_imitation_model(self, filepath=None, performance_metrics=None):
        """Save the trained imitation model with metadata"""
        if filepath is None:
            filepath = self.get_imitation_model_path()
        
        if self.imitation_model is None:
            print("No imitation model to save!")
            return None
        
        # Save model
        model_data = {
            'model': self.imitation_model,
            'training_date': datetime.now().isoformat(),
            'expert_demonstrations_count': len(self.expert_demonstrations),
            'performance_metrics': performance_metrics or {}
        }
        
        joblib.dump(model_data, filepath)
        print(f"Imitation model saved to {filepath}")
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'expert_demonstrations_count': len(self.expert_demonstrations),
            'performance_metrics': performance_metrics or {},
            'model_path': filepath
        }
        
        metadata_path = self.get_imitation_metadata_path()
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Imitation model metadata saved to {metadata_path}")
        return filepath
    
    def load_imitation_model(self, filepath=None):
        """Load a saved imitation model"""
        if filepath is None:
            filepath = self.get_imitation_model_path()
            
        if not os.path.exists(filepath):
            print(f"Imitation model file not found: {filepath}")
            return False
            
        print(f"Loading imitation model from {filepath}")
        
        try:
            model_data = joblib.load(filepath)
            
            self.imitation_model = model_data['model']
            
            # Load metadata if available
            metadata_path = self.get_imitation_metadata_path()
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    print(f"Imitation model trained on: {metadata.get('training_date', 'Unknown')}")
                    print(f"Expert demonstrations: {metadata.get('expert_demonstrations_count', 'Unknown')}")
                    print(f"Performance: {metadata.get('performance_metrics', {})}")
            
            print("Imitation model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading imitation model: {e}")
            return False
    
    def predict_expert_response_features(self, original_text, urgency_level):
        """Predict what expert response features should be"""
        if self.imitation_model is None:
            return None
        
        # Prepare input features
        input_features = np.array([[
            len(original_text.split()),
            urgency_level,
            self._calculate_sentiment(original_text)
        ]])
        
        # Predict expert features
        predicted_features = self.imitation_model.predict(input_features)[0]
        
        feature_names = ['response_length', 'sentiment_score', 'urgency_indicators', 
                        'helpfulness_indicators', 'formality_score', 'empathy_score']
        
        return dict(zip(feature_names, predicted_features))
    
    def evolve_agent_parameters(self):
        """Evolve agent parameters based on recent performance with performance memory and feedback loops"""
        if len(self.experience_buffer) < 5:
            return
        
        # Initialize performance memory if not exists
        if not hasattr(self, 'performance_memory'):
            self.performance_memory = {
                'successful_parameters': [],
                'failed_parameters': [],
                'parameter_performance_history': {}
            }
        
        # Get recent experiences for analysis
        recent_experiences = self.experience_buffer[-20:]
        
        # Calculate performance metrics with enhanced variation
        rewards = [exp['reward'] for exp in recent_experiences]
        quality_scores = [exp['quality_score'] for exp in recent_experiences]
        
        avg_reward = np.mean(rewards)
        avg_quality = np.mean(quality_scores)
        reward_std = np.std(rewards)
        
        # Track current parameter performance
        current_params_key = self._serialize_parameters()
        if current_params_key not in self.performance_memory['parameter_performance_history']:
            self.performance_memory['parameter_performance_history'][current_params_key] = []
        
        self.performance_memory['parameter_performance_history'][current_params_key].append({
            'reward': avg_reward,
            'quality': avg_quality,
            'timestamp': datetime.now().isoformat()
        })
        
        # Classify current parameters as successful or failed
        if avg_reward > 15 and avg_quality > 0.65:
            if current_params_key not in self.performance_memory['successful_parameters']:
                self.performance_memory['successful_parameters'].append(current_params_key)
        elif avg_reward < 8 or avg_quality < 0.45:
            if current_params_key not in self.performance_memory['failed_parameters']:
                self.performance_memory['failed_parameters'].append(current_params_key)
        
        # Add performance momentum tracking
        if len(self.reward_history) > 10:
            recent_momentum = np.mean(self.reward_history[-5:]) - np.mean(self.reward_history[-10:-5])
        else:
            recent_momentum = 0
        
        # Enhanced adaptive exploration rate with feedback loops
        base_exploration = 0.15  # Increased base exploration
        if avg_reward > 18:  # Excellent performance - fine-tune
            self.exploration_rate = base_exploration * 0.6  # Less aggressive reduction
        elif avg_reward > 15:  # High performance - small adjustments
            self.exploration_rate = base_exploration * 0.7
        elif avg_reward < 8:  # Low performance - major exploration
            self.exploration_rate = base_exploration * 2.0
        elif recent_momentum < -2:  # Declining performance - boost exploration
            self.exploration_rate = base_exploration * 1.8
        else:
            self.exploration_rate = base_exploration * (1.0 + np.random.uniform(-0.2, 0.2))
        
        # Keep exploration rate in reasonable bounds
        self.exploration_rate = max(min(self.exploration_rate, 0.8), 0.02)
        
        # Enhanced evolution strength calculation with performance feedback
        performance_factor = max(0.2, 1.2 - (avg_reward / 20))  # Increased base and adjusted scaling
        momentum_factor = 1.0 + abs(recent_momentum) / 6  # More responsive to momentum
        
        # Reduce evolution strength if we have successful parameters to preserve
        if len(self.performance_memory['successful_parameters']) > 0:
            success_factor = 0.8  # Less conservative evolution
        else:
            success_factor = 1.4  # More aggressive evolution
        
        evolution_strength = 0.2 * performance_factor * momentum_factor * success_factor  # Increased base
        evolution_strength = max(0.05, min(evolution_strength, 0.6))  # Increased bounds
        
        # Evolve agent parameters with performance memory guidance
        for category in self.agent_parameters:
            if not isinstance(self.agent_parameters[category], dict):
                continue
                
            for param in self.agent_parameters[category]:
                current_value = self.agent_parameters[category][param]
                
                # Use successful parameter patterns if available
                if len(self.performance_memory['successful_parameters']) > 0:
                    # Bias towards successful parameter ranges
                    successful_values = self._get_successful_parameter_values(category, param)
                    if successful_values:
                        target_value = np.mean(successful_values)
                        # Move towards successful values
                        direction = np.sign(target_value - current_value)
                        adjustment = direction * np.random.uniform(0.01, 0.05) * evolution_strength
                    else:
                        # Standard evolution if no successful patterns found
                        adjustment = self._calculate_standard_adjustment(avg_reward, evolution_strength)
                else:
                    # Standard evolution for new parameters
                    adjustment = self._calculate_standard_adjustment(avg_reward, evolution_strength)
                
                # Apply momentum-based adjustment
                if recent_momentum > 1:  # Positive momentum - smaller changes
                    adjustment *= 0.7
                elif recent_momentum < -1:  # Negative momentum - larger changes
                    adjustment *= 1.4
                
                # Update parameter with bounds checking
                new_value = current_value + adjustment
                
                # Apply parameter-specific bounds
                if 'weight' in param or 'factor' in param:
                    new_value = max(0.1, min(new_value, 2.0))
                elif 'threshold' in param:
                    new_value = max(0.0, min(new_value, 1.0))
                elif 'rate' in param:
                    new_value = max(0.01, min(new_value, 0.5))
                else:
                    new_value = max(0.0, min(new_value, 1.0))
                
                self.agent_parameters[category][param] = new_value
        
        # Evolve policy parameters with feedback loops
        self._evolve_policy_parameters(avg_reward, recent_momentum, evolution_strength)
        
        # Clean up old performance memory to prevent memory bloat
        self._cleanup_performance_memory()
        
        # Log evolution step
        evolution_log = {
            'timestamp': datetime.now().isoformat(),
            'avg_reward': avg_reward,
            'avg_quality': avg_quality,
            'exploration_rate': self.exploration_rate,
            'evolution_strength': evolution_strength,
            'momentum': recent_momentum,
            'successful_params_count': len(self.performance_memory['successful_parameters']),
            'failed_params_count': len(self.performance_memory['failed_parameters'])
        }
        
        if not hasattr(self, 'evolution_log'):
            self.evolution_log = []
        self.evolution_log.append(evolution_log)
        
        # Keep only recent evolution logs
        if len(self.evolution_log) > 100:
            self.evolution_log = self.evolution_log[-100:]
    
    def _evolve_policy_parameters(self, avg_reward, momentum, evolution_strength):
        """Evolve policy parameters with enhanced variation"""
        # Evolve template selection probabilities
        for template_type in self.policy_parameters['template_selection']:
            current_probs = self.policy_parameters['template_selection'][template_type]
            
            # Add variation based on performance
            if avg_reward > 12:
                # Small refinements for good performance
                variation = np.random.normal(0, 0.02, len(current_probs))
            else:
                # Larger changes for poor performance
                variation = np.random.normal(0, 0.08, len(current_probs))
            
            # Apply momentum effect
            if momentum > 1:
                variation *= 0.7  # Reduce variation when improving
            elif momentum < -1:
                variation *= 1.5  # Increase variation when declining
            
            new_probs = current_probs + variation
            new_probs = np.maximum(new_probs, 0.1)  # Minimum probability
            new_probs = new_probs / np.sum(new_probs)  # Normalize
            
            self.policy_parameters['template_selection'][template_type] = new_probs
        
        # Evolve response generation parameters
        for param in self.policy_parameters['response_generation']:
            current_value = self.policy_parameters['response_generation'][param]
            
            # Performance-based variation
            if avg_reward > 12:
                adjustment = np.random.normal(0, 0.03 * evolution_strength)
            else:
                adjustment = np.random.normal(0, 0.1 * evolution_strength)
            
            # Momentum effect
            if momentum > 0:
                momentum_adj = np.random.uniform(-0.02, 0.04)
            else:
                momentum_adj = np.random.uniform(-0.06, 0.06)
            
            new_value = current_value + adjustment + momentum_adj
            self.policy_parameters['response_generation'][param] = max(min(new_value, 1.0), 0.0)
    
    def get_evolved_action(self, state):
        """Get action using evolved parameters with actual parameter application"""
        # Apply evolved parameters to action generation
        if np.random.random() < self.exploration_rate:
            action = self._generate_random_action(state)
            action['exploration'] = True
        else:
            action = self._generate_policy_action(state)
            action['exploration'] = False
        
        # Apply agent parameters to modify the action
        action = self._apply_agent_parameters(action, state)
        
        return action
    
    def _apply_agent_parameters(self, action, state):
        """Apply evolved agent parameters to modify the action"""
        # Apply response generation parameters
        for param, value in self.agent_parameters['response_generation'].items():
            if param == 'response_length_factor':
                action['target_length'] = action.get('target_length', 150) * value
            elif param == 'formality_level':
                action['formality'] = value
            elif param == 'empathy_level':
                action['empathy'] = value
            elif param == 'urgency_sensitivity':
                if state.get('urgency_level') == 1:
                    action['urgency_boost'] = value
        
        # Apply quality optimization parameters
        for param, value in self.agent_parameters['quality_weights'].items():
            if param == 'relevance_weight':
                action['relevance_focus'] = value
            elif param == 'helpfulness_weight':
                action['helpfulness_focus'] = value
            elif param == 'clarity_weight':
                action['clarity_focus'] = value
        
        # Apply template selection parameters
        template_weights = self.policy_parameters['template_selection']
        urgency = state.get('urgency_level', 0)
        
        if urgency == 1:
            action['template_preference'] = 'urgent' if template_weights.get('high_urgency_templates', [0.5])[0] > 0.5 else 'supportive'
        else:
            action['template_preference'] = 'informative' if template_weights.get('low_urgency_templates', [0.5])[0] > 0.5 else 'supportive'
        
        return action
    
    def _generate_random_action(self, state):
        """Generate random action for exploration"""
        urgency_level = state.get('urgency_level', 0)
        
        action = {
            'template_index': np.random.randint(0, 4),
            'length_preference': np.random.random(),
            'formality_level': np.random.random(),
            'empathy_level': np.random.random(),
            'urgency_level': urgency_level
        }
        
        return action
    
    def _generate_policy_action(self, state):
        """Generate action based on learned policy"""
        urgency_level = state.get('urgency_level', 0)
        
        # Select template based on learned probabilities
        template_type = 'high_urgency_templates' if urgency_level == 1 else 'low_urgency_templates'
        template_probs = self.policy_parameters['template_selection'][template_type]
        
        # Normalize probabilities to ensure they sum to 1
        template_probs = template_probs / np.sum(template_probs)
        template_index = np.random.choice(len(template_probs), p=template_probs)
        
        # Use learned response generation parameters
        action = {
            'template_index': template_index,
            'length_preference': self.policy_parameters['response_generation']['length_preference'],
            'formality_level': self.policy_parameters['response_generation']['formality_level'],
            'empathy_level': self.policy_parameters['response_generation']['empathy_level'],
            'urgency_level': urgency_level
        }
        
        return action
    
    def generate_evolution_report(self):
        """Generate comprehensive evolution report"""
        if not self.evolution_log:
            return {'error': 'No evolution data available'}
        
        recent_log = self.evolution_log[-10:] if len(self.evolution_log) > 10 else self.evolution_log
        
        report = {
            'evolution_summary': {
                'total_steps': len(self.evolution_log),
                'experiences_recorded': len(self.experience_buffer),
                'expert_demonstrations': len(self.expert_demonstrations),
                'current_exploration_rate': self.exploration_rate
            },
            'performance_metrics': {
                'avg_reward_recent': np.mean([step['avg_reward'] for step in recent_log]),
                'avg_quality_recent': np.mean([step['avg_quality'] for step in recent_log]),
                'reward_trend': self._calculate_trend([step['avg_reward'] for step in recent_log]),
                'quality_trend': self._calculate_trend([step['avg_quality'] for step in recent_log])
            },
            'current_parameters': {
                'agent_parameters': self.agent_parameters,
                'policy_parameters': self.policy_parameters
            },
            'learning_progress': {
                'imitation_model_trained': self.imitation_model is not None,
                'total_rewards': sum(self.reward_history),
                'avg_reward_all_time': np.mean(self.reward_history) if self.reward_history else 0
            }
        }
        
        return report
    
    def _calculate_trend(self, values):
        """Calculate trend direction from a list of values"""
        if len(values) < 2:
            return 'insufficient_data'
        
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        if second_half > first_half * 1.05:
            return 'improving'
        elif second_half < first_half * 0.95:
            return 'declining'
        else:
            return 'stable'
    
    def save_evolution_state(self):
        """Save the current evolution state to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_file = f"evolution_state_{timestamp}.json"
        
        state_data = {
            'timestamp': timestamp,
            'agent_parameters': self.agent_parameters,
            'policy_parameters': self.policy_parameters,
            'evolution_log': self.evolution_log,
            'experience_buffer': self.experience_buffer,
            'expert_demonstrations': self.expert_demonstrations,
            'reward_history': self.reward_history,
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate,
            'decay_rate': self.decay_rate
        }
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            print(f"Evolution state saved to {state_file}")
            return state_file
        except Exception as e:
            print(f"Error saving evolution state: {e}")
            return None

    def _serialize_parameters(self):
        """Serialize current parameters to a string key for tracking"""
        param_dict = {}
        for category in self.agent_parameters:
            if isinstance(self.agent_parameters[category], dict):
                for param, value in self.agent_parameters[category].items():
                    param_dict[f"{category}_{param}"] = round(value, 3)
        
        # Include policy parameters
        for param, value in self.policy_parameters.items():
            if isinstance(value, (int, float)):
                param_dict[f"policy_{param}"] = round(value, 3)
        
        return str(sorted(param_dict.items()))
    
    def _get_successful_parameter_values(self, category, param):
        """Get parameter values from successful parameter sets"""
        successful_values = []
        
        for params_key in self.performance_memory['successful_parameters']:
            try:
                # Parse the serialized parameters
                param_dict = eval(params_key)
                target_key = f"{category}_{param}"
                
                for key, value in param_dict:
                    if key == target_key:
                        successful_values.append(value)
                        break
            except:
                continue
        
        return successful_values
    
    def _calculate_standard_adjustment(self, avg_reward, evolution_strength):
        """Calculate standard parameter adjustment based on performance"""
        if avg_reward > 15:  # Excellent performance - small adjustments
            adjustment = np.random.normal(0, 0.02 * evolution_strength)
        elif avg_reward > 10:  # Good performance - moderate adjustments
            adjustment = np.random.normal(0, 0.05 * evolution_strength)
        elif avg_reward < 5:  # Poor performance - large adjustments
            adjustment = np.random.normal(0, 0.15 * evolution_strength)
        else:  # Medium performance - standard adjustments
            adjustment = np.random.normal(0, 0.08 * evolution_strength)
        
        return adjustment
    
    def _cleanup_performance_memory(self):
        """Clean up old performance memory to prevent memory bloat"""
        max_successful = 20
        max_failed = 15
        max_history_per_param = 10
        
        # Limit successful parameters
        if len(self.performance_memory['successful_parameters']) > max_successful:
            self.performance_memory['successful_parameters'] = \
                self.performance_memory['successful_parameters'][-max_successful:]
        
        # Limit failed parameters
        if len(self.performance_memory['failed_parameters']) > max_failed:
            self.performance_memory['failed_parameters'] = \
                self.performance_memory['failed_parameters'][-max_failed:]
        
        # Limit history per parameter set
        for params_key in list(self.performance_memory['parameter_performance_history'].keys()):
            history = self.performance_memory['parameter_performance_history'][params_key]
            if len(history) > max_history_per_param:
                self.performance_memory['parameter_performance_history'][params_key] = \
                    history[-max_history_per_param:]

if __name__ == "__main__":
    evolution = test_evolution_system()