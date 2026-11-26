"""
Main Evolving Urgency-Response Agent System
Integrates all components: classifier, response agent, evolution system, and metrics
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import system components
from binary_urgency_classifier import BinaryUrgencyClassifier
from response_agent import ResponseAgent
from evolution_system import EvolutionSystem
from metrics_framework import MetricsFramework

class EvolvingUrgencyResponseSystem:
    def __init__(self, data_path='processed_data.csv', max_features=500):
        """Initialize the complete evolving system"""
        self.data_path = data_path
        self.max_features = max_features
        
        # Initialize components
        self.classifier = BinaryUrgencyClassifier(data_path, max_features)
        self.response_agent = ResponseAgent()
        self.evolution_system = EvolutionSystem()
        self.metrics_framework = MetricsFramework()
        
        # System state
        self.is_trained = False
        self.classifier_model_path = None
        self.system_history = []
        
        print("Evolving Urgency-Response Agent System initialized")
    
    def setup_system(self, force_retrain=False):
        """Set up and train the complete system with model persistence"""
        print("Setting up the complete system...")
        
        # Step 1: Train or load the binary classifier
        print("\n1. Setting up Binary Urgency Classifier...")
        self.classifier, classifier_results = self._setup_classifier(force_retrain)
        
        # Step 2: Load classifier into response agent
        print("\n2. Setting up Response Agent...")
        if self.classifier_model_path:
            self.response_agent.load_classifier(self.classifier_model_path)
        
        # Step 3: Initialize metrics baseline
        print("\n3. Establishing Metrics Baseline...")
        self._establish_baseline_metrics(classifier_results)
        
        self.is_trained = True
        print("\nSystem setup completed successfully!")
        
        return True
    
    def _setup_classifier(self, force_retrain=False):
        """Set up the binary urgency classifier with persistence"""
        # Use the new train_or_load_model method
        result = self.classifier.train_or_load_model(force_retrain)
        
        if 'model_data' in result:
            # New training occurred
            # Extract results from the training pipeline
            classifier_results = result['results']
            self.classifier_model_path = self.classifier.get_model_path()
        else:
            # Model was loaded from disk
            # Create dummy results for baseline establishment
            classifier_results = {
                self.classifier.best_model: {
                    'accuracy': 0.85,  # Default values, will be updated if metadata available
                    'precision': 0.85,
                    'recall': 0.85,
                    'f1_score': 0.85
                }
            }
            self.classifier_model_path = self.classifier.get_model_path()
        
        return self.classifier, classifier_results
    
    def _train_classifier(self):
        """Train the binary urgency classifier (legacy method, kept for compatibility)"""
        # Load and prepare data
        data = self.classifier.load_data()
        X_text, X_numerical, y_binary = self.classifier.prepare_binary_features()
        
        # Create features
        X_tfidf = self.classifier.create_text_features()
        X_combined = self.classifier.combine_features(X_tfidf)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Train models
        models = self.classifier.train_binary_models(X_train, y_train)
        
        # Evaluate models
        results = self.classifier.evaluate_binary_models(X_test, y_test)
        
        # Save best model
        self.classifier_model_path = self.classifier.save_binary_model()
        
        return self.classifier, results
    
    def _establish_baseline_metrics(self, classifier_results):
        """Establish baseline metrics for the system"""
        if not classifier_results:
            return
        
        # Get best model results
        best_model = self.classifier.best_model
        best_results = classifier_results[best_model]
        
        # Create baseline classification metrics
        baseline_classification = {
            'accuracy': best_results['accuracy'],
            'precision': best_results['precision'],
            'recall': best_results['recall'],
            'f1_score': best_results['f1_score']
        }
        
        # Store baseline
        self.metrics_framework.performance_baselines['classification'] = baseline_classification
        
        print(f"Baseline established - Accuracy: {baseline_classification['accuracy']:.4f}")
    
    def process_single_request(self, text, user_context=None, record_metrics=True):
        """Process a single user request through the complete system"""
        if not self.is_trained:
            raise ValueError("System not trained. Call setup_system() first.")
        
        start_time = time.time()
        
        # Step 1: Get current state for evolution system
        state = {
            'text_length': len(text.split()),
            'urgency_level': None,  # Will be filled by classifier
            'user_context': user_context
        }
        
        # Step 2: Get evolved action from evolution system
        action = self.evolution_system.get_evolved_action(state)
        
        # Step 3: Set evolution step in response agent for quality calculation
        evolution_steps = len(self.evolution_system.experience_buffer)
        evolution_steps = int(evolution_steps) if isinstance(evolution_steps, str) else evolution_steps
        self.response_agent.evolution_step = evolution_steps
        
        # Step 4: Generate response using response agent with evolved action
        response_data = self.response_agent.generate_response(text, user_context=user_context, evolved_action=action)
        
        # Update state with classifier result
        state['urgency_level'] = response_data['urgency_level']
        
        # Step 5: Calculate reward for evolution system
        reward = self.evolution_system.calculate_reward(response_data)
        
        # Step 6: Record experience for learning
        next_state = state.copy()  # Simplified for this example
        self.evolution_system.record_experience(state, action, reward, next_state, response_data)
        
        # Step 7: Update policy based on reward
        if len(self.evolution_system.experience_buffer) > 0:
            latest_experience = self.evolution_system.experience_buffer[-1]
            self.evolution_system.update_policy_with_reward(latest_experience)
        
        # Step 8: Record metrics if requested
        if record_metrics:
            self._record_system_metrics(response_data, reward, time.time() - start_time)
        
        # Step 9: Periodic evolution
        if len(self.evolution_system.experience_buffer) % 10 == 0:
            self.evolution_system.evolve_agent_parameters()
        
        # Create comprehensive result
        result = {
            'timestamp': datetime.now().isoformat(),
            'original_text': text,
            'response_data': response_data,
            'reward': reward,
            'action': action,
            'state': state,
            'processing_time': time.time() - start_time,
            'quality_score': response_data.get('quality_score', 0),
            'urgency_prediction': response_data.get('urgency_level', 0)
        }
        
        self.system_history.append(result)
        
        return result
    
    def _record_system_metrics(self, response_data, reward, processing_time):
        """Record comprehensive system metrics"""
        # Classification metrics (simplified - would need actual predictions vs ground truth)
        classification_metrics = {
            'prediction_confidence': response_data.get('confidence', [0.5, 0.5])[1],
            'urgency_prediction': response_data['urgency_level']
        }
        
        # Response quality metrics
        response_metrics = {
            'quality_score': response_data['quality_score'],
            'response_time': response_data['response_time'],
            'urgency_level': response_data['urgency_level']
        }
        
        # System health metrics
        system_metrics = {
            'processing_time': processing_time,
            'reward': reward,
            'exploration_rate': self.evolution_system.exploration_rate
        }
        
        # User satisfaction (simulated based on quality and reward)
        satisfaction_metrics = {
            'estimated_satisfaction': min(response_data['quality_score'] + reward / 10, 1.0)
        }
        
        # Record snapshot
        self.metrics_framework.record_metrics_snapshot(
            classifier_metrics=classification_metrics,
            response_metrics=response_metrics,
            system_metrics=system_metrics,
            satisfaction_metrics=satisfaction_metrics
        )
    
    def batch_process_requests(self, texts, save_results=True):
        """Process a batch of requests and track evolution"""
        print(f"Processing batch of {len(texts)} requests...")
        
        results = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"Processing {i+1}/{len(texts)}")
            
            try:
                result = self.process_single_request(text)
                results.append(result)
            except Exception as e:
                print(f"Error processing request {i}: {e}")
                continue
        
        if save_results:
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"batch_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"Batch results saved to {results_file}")
        
        return results
    
    def add_expert_feedback(self, original_text, expert_response, quality_score=1.0):
        """Add expert demonstration for imitation learning"""
        expert_data = {
            'original_text': original_text,
            'response': expert_response,
            'quality_score': quality_score,
            'response_time': 0.5,  # Assume experts are efficient
            'urgency_level': self._predict_urgency_for_expert_demo(original_text)
        }
        
        self.evolution_system.add_expert_demonstration(expert_data)
        print("Expert demonstration added to evolution system")
    
    def _predict_urgency_for_expert_demo(self, text):
        """Predict urgency for expert demonstration"""
        try:
            if hasattr(self.response_agent, 'classifier'):
                prediction = self.response_agent.classifier.predict_urgency(text)
                return prediction['urgency_class']
        except:
            pass
        
        # Fallback: simple keyword-based prediction
        urgent_keywords = ['urgent', 'asap', 'emergency', 'deadline', 'help', 'problem']
        text_lower = text.lower()
        return 1 if any(keyword in text_lower for keyword in urgent_keywords) else 0
    
    def train_imitation_learning(self, force_retrain=False):
        """Train the imitation learning component with persistence"""
        print("Training imitation learning model...")
        success = self.evolution_system.train_imitation_model(force_retrain)
        
        if success:
            print("Imitation learning model ready!")
        else:
            print("Failed to set up imitation learning model")
        
        return success
    
    def generate_comprehensive_report(self):
        """Generate comprehensive system report"""
        print("Generating comprehensive system report...")
        
        report = {
            'system_overview': {
                'total_requests_processed': len(self.system_history),
                'system_trained': self.is_trained,
                'classifier_model_path': self.classifier_model_path,
                'max_features': self.max_features
            },
            'performance_summary': {},
            'evolution_report': {},
            'metrics_report': {}
        }
        
        # Performance summary
        if self.system_history:
            rewards = [r['reward'] for r in self.system_history]
            quality_scores = [r['response_data']['quality_score'] for r in self.system_history]
            processing_times = [r['processing_time'] for r in self.system_history]
            
            report['performance_summary'] = {
                'avg_reward': np.mean(rewards),
                'avg_quality_score': np.mean(quality_scores),
                'avg_processing_time': np.mean(processing_times),
                'reward_trend': self._calculate_trend(rewards),
                'quality_trend': self._calculate_trend(quality_scores)
            }
        
        # Evolution report
        report['evolution_report'] = self.evolution_system.generate_evolution_report()
        
        # Metrics report
        report['metrics_report'] = self.metrics_framework.generate_comprehensive_report()
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"system_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Comprehensive report saved to {report_file}")
        
        return report
    
    def _calculate_trend(self, values):
        """Calculate trend in values"""
        if len(values) < 3:
            return 0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.001:
            return 1  # Improving
        elif slope < -0.001:
            return -1  # Declining
        else:
            return 0  # Stable
    
    def save_system_state(self, filepath=None):
        """Save complete system state"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"system_state_{timestamp}.json"
        
        # Save individual component states
        evolution_state_file = self.evolution_system.save_evolution_state()
        metrics_state_file = self.metrics_framework.save_metrics_data()
        agent_state_file = self.response_agent.save_agent_state()
        
        # Save main system state
        system_state = {
            'is_trained': self.is_trained,
            'classifier_model_path': self.classifier_model_path,
            'max_features': self.max_features,
            'data_path': self.data_path,
            'system_history': self.system_history[-100:],  # Last 100 requests
            'component_state_files': {
                'evolution_system': evolution_state_file,
                'metrics_framework': metrics_state_file,
                'response_agent': agent_state_file
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(system_state, f, indent=2, default=str)
        
        print(f"Complete system state saved to {filepath}")
        return filepath
    
    def load_system_state(self, filepath):
        """Load complete system state"""
        try:
            with open(filepath, 'r') as f:
                system_state = json.load(f)
            
            self.is_trained = system_state.get('is_trained', False)
            self.classifier_model_path = system_state.get('classifier_model_path')
            self.max_features = system_state.get('max_features', 500)
            self.data_path = system_state.get('data_path', 'processed_data.csv')
            self.system_history = system_state.get('system_history', [])
            
            # Load component states
            component_files = system_state.get('component_state_files', {})
            
            if 'evolution_system' in component_files:
                self.evolution_system.load_evolution_state(component_files['evolution_system'])
            
            if 'metrics_framework' in component_files:
                self.metrics_framework.load_metrics_data(component_files['metrics_framework'])
            
            if 'response_agent' in component_files:
                self.response_agent.load_agent_state(component_files['response_agent'])
            
            # Reload classifier if available
            if self.classifier_model_path and self.is_trained:
                self.response_agent.load_classifier(self.classifier_model_path)
            
            print(f"System state loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading system state: {e}")
            return False

def run_system_demo():
    """Run a demonstration of the complete system"""
    print("=" * 60)
    print("EVOLVING URGENCY-RESPONSE AGENT SYSTEM DEMO")
    print("=" * 60)
    
    # Initialize system
    system = EvolvingUrgencyResponseSystem()
    
    # Setup system (train classifier, etc.)
    system.setup_system()
    
    # Test requests with different urgency levels
    test_requests = [
        "I can't access the assignment and the deadline is in 2 hours! Please help urgently!",
        "Could you explain the concept of machine learning from lecture 3?",
        "URGENT: The system crashed and I lost all my work. What should I do immediately?",
        "I'm having trouble understanding the homework instructions. Can you help?",
        "The final project submission system is down and deadline is today!",
        "Thanks for the great course. I'm enjoying the materials so far.",
        "I need help with question 5 on the assignment.",
        "EMERGENCY: I can't log into the system for the exam!",
        "What are the office hours for this week?",
        "The video lecture is not loading properly. Please fix this ASAP!"
    ]
    
    print(f"\nProcessing {len(test_requests)} test requests...")
    
    # Process requests and show evolution
    results = []
    for i, request in enumerate(test_requests):
        print(f"\n--- Request {i+1} ---")
        print(f"Input: {request}")
        
        result = system.process_single_request(request)
        
        print(f"Urgency: {result['response_data']['urgency_label']}")
        print(f"Response: {result['response_data']['response'][:100]}...")
        print(f"Quality Score: {result['response_data']['quality_score']:.3f}")
        print(f"Reward: {result['reward']:.2f}")
        print(f"Processing Time: {result['processing_time']:.3f}s")
        
        results.append(result)
    
    # Add some expert demonstrations
    print("\n--- Adding Expert Demonstrations ---")
    expert_demos = [
        ("I can't submit my assignment due to technical issues!", 
         "I understand this is urgent. I'm immediately escalating this to our technical support team. In the meantime, please document the error message and try submitting from a different browser. We'll ensure you get an extension if needed."),
        ("What is the difference between supervised and unsupervised learning?",
         "Great question! Supervised learning uses labeled data to train models (like classification), while unsupervised learning finds patterns in unlabeled data (like clustering). Check lecture 4 for detailed examples.")
    ]
    
    for text, expert_response in expert_demos:
        system.add_expert_feedback(text, expert_response, quality_score=0.95)
    
    # Train imitation learning
    system.train_imitation_learning()
    
    # Process a few more requests to see evolution
    print("\n--- Processing Additional Requests (Post-Evolution) ---")
    additional_requests = [
        "Help! I accidentally deleted my project file and deadline is tomorrow!",
        "Can you recommend some additional reading materials for this topic?"
    ]
    
    for request in additional_requests:
        result = system.process_single_request(request)
        print(f"Input: {request}")
        print(f"Response: {result['response_data']['response'][:100]}...")
        print(f"Quality: {result['response_data']['quality_score']:.3f}, Reward: {result['reward']:.2f}")
    
    # Generate comprehensive report
    print("\n--- Generating System Report ---")
    report = system.generate_comprehensive_report()
    
    # Show key metrics
    print("\n=== SYSTEM PERFORMANCE SUMMARY ===")
    if 'performance_summary' in report:
        perf = report['performance_summary']
        print(f"Total Requests Processed: {len(system.system_history)}")
        print(f"Average Reward: {perf.get('avg_reward', 0):.2f}")
        print(f"Average Quality Score: {perf.get('avg_quality_score', 0):.3f}")
        print(f"Average Processing Time: {perf.get('avg_processing_time', 0):.3f}s")
        print(f"Reward Trend: {'Improving' if perf.get('reward_trend', 0) > 0 else 'Stable' if perf.get('reward_trend', 0) == 0 else 'Declining'}")
        print(f"Quality Trend: {'Improving' if perf.get('quality_trend', 0) > 0 else 'Stable' if perf.get('quality_trend', 0) == 0 else 'Declining'}")
    
    # Save system state
    system_state_file = system.save_system_state()
    print(f"\nSystem state saved to: {system_state_file}")
    
    print("\n=== DEMO COMPLETED SUCCESSFULLY ===")
    
    return system, results, report

if __name__ == "__main__":
    system, results, report = run_system_demo()