"""
Integrated LLM System Module
Combines existing classifiers with LLM agent and orchestrates the complete workflow with evolution
"""

import json
import time
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Import existing components
from binary_urgency_classifier import BinaryUrgencyClassifier
from urgency_classifier import UrgencyClassifier
from llm_response_agent import LLMResponseAgent
from llm_evolution_system import LLMEvolutionSystem
from llm_metrics_framework import LLMMetricsFramework

class IntegratedLLMSystem:
    def __init__(self, 
                 models_dir: str = "models",
                 save_dir: str = ".",
                 evolution_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Integrated LLM System
        
        Args:
            models_dir: Directory containing trained classifier models
            save_dir: Directory for saving system state and metrics
            evolution_config: Configuration for evolution system
        """
        self.models_dir = models_dir
        self.save_dir = save_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.binary_classifier = None
        self.urgency_classifier = None
        self.llm_agent = None
        self.evolution_system = None
        self.metrics_framework = None
        
        # System state
        self.is_initialized = False
        self.total_requests = 0
        self.evolution_step = 0
        self.system_metrics = {
            'initialization_time': None,
            'total_processing_time': 0.0,
            'average_response_time': 0.0,
            'evolution_cycles': 0,
            'quality_improvement': 0.0
        }
        
        # Evolution configuration
        self.evolution_config = evolution_config or {
            'population_size': 8,
            'mutation_rate': 0.3,
            'crossover_rate': 0.7,
            'elite_ratio': 0.2,
            'evolution_frequency': 5  # Evolve every N requests
        }
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components"""
        print("Initializing Integrated LLM System...")
        start_time = time.time()
        
        try:
            # Initialize classifiers
            print("Loading classifiers...")
            self.binary_classifier = BinaryUrgencyClassifier()
            self.urgency_classifier = UrgencyClassifier()
            
            # Load trained models if available
            self._load_classifier_models()
            
            # Initialize LLM agent
            print("Initializing LLM Response Agent...")
            self.llm_agent = LLMResponseAgent()
            
            # Initialize metrics framework
            print("Setting up metrics framework...")
            self.metrics_framework = LLMMetricsFramework()
            
            # Initialize evolution system
            print("Setting up evolution system...")
            self.evolution_system = LLMEvolutionSystem(
                agent=self.llm_agent,
                population_size=self.evolution_config['population_size'],
                mutation_rate=self.evolution_config['mutation_rate'],
                crossover_rate=self.evolution_config['crossover_rate'],
                elite_ratio=self.evolution_config['elite_ratio']
            )
            
            # Set evolution step in agent
            self.llm_agent.set_evolution_step(self.evolution_step)
            
            initialization_time = time.time() - start_time
            self.system_metrics['initialization_time'] = initialization_time
            self.is_initialized = True
            
            print(f"System initialized successfully in {initialization_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error initializing system: {str(e)}")
            raise
    
    def _load_classifier_models(self):
        """Load pre-trained classifier models"""
        try:
            # Load binary urgency classifier
            binary_model_path = os.path.join(self.models_dir, "binary_urgency_classifier_gradient_boosting.joblib")
            if os.path.exists(binary_model_path):
                self.binary_classifier.load_model(binary_model_path)
                print("Binary urgency classifier loaded successfully")
            else:
                print("Binary urgency classifier model not found, will train if needed")
            
            # Load multi-class urgency classifier
            urgency_model_path = os.path.join(self.models_dir, "urgency_classifier_gradient_boosting.joblib")
            if os.path.exists(urgency_model_path):
                self.urgency_classifier.load_model(urgency_model_path)
                print("Multi-class urgency classifier loaded successfully")
            else:
                print("Multi-class urgency classifier model not found, will train if needed")
                
        except Exception as e:
            print(f"Warning: Error loading classifier models: {str(e)}")
    
    def process_request(self, 
                       input_text: str, 
                       user_context: Optional[Dict[str, Any]] = None,
                       enable_evolution: bool = True) -> Dict[str, Any]:
        """
        Process a complete request through the integrated system
        
        Args:
            input_text: User input text
            user_context: Optional user context information
            enable_evolution: Whether to enable evolution after processing
            
        Returns:
            Dictionary containing complete processing results
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized")
        
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Step 1: Classify urgency (binary and multi-class)
            print(f"\n{'='*60}")
            print(f"Processing Request #{self.total_requests}")
            print(f"Input: {input_text}")
            print(f"{'='*60}")
            
            print("\n1. Classifying urgency...")
            binary_urgency = self._classify_binary_urgency(input_text)
            detailed_urgency = self._classify_detailed_urgency(input_text)
            
            print(f"   Binary urgency: {binary_urgency}")
            print(f"   Detailed urgency: {detailed_urgency}")
            
            # Step 2: Classify content category
            print("\n2. Classifying content category...")
            content_category = self.llm_agent.classify_content_category(input_text)
            print(f"   Content category: {content_category}")
            
            # Step 3: Generate LLM response
            print(f"\n3. Generating LLM response (Evolution Step: {self.evolution_step})...")
            response_data = self.llm_agent.generate_response(
                text=input_text,
                context=user_context
            )
            
            print(f"   Response generated in {response_data['response_time']:.2f}s")
            print(f"   Response: {response_data['response'][:100]}...")
            
            # Step 4: Evaluate response quality
            print("\n4. Evaluating response quality...")
            quality_metrics = self.metrics_framework.evaluate_response_quality(
                input_text=input_text,
                response=response_data['response'],
                urgency_level=detailed_urgency,
                content_category=content_category,
                evolution_step=self.evolution_step
            )
            
            print(f"   Quality score: {quality_metrics['overall_quality']:.3f}")
            
            # Step 5: Update evolution system
            if enable_evolution:
                print("\n5. Updating evolution system...")
                try:
                    # Get current evolution parameters
                    current_params = self.llm_agent.evolution_params.copy()
                    
                    # Create quality scores dictionary from quality metrics
                    quality_scores = quality_metrics.get('quality_dimensions', {
                        'relevance': quality_metrics['overall_quality'],
                        'helpfulness': quality_metrics['overall_quality'],
                        'clarity': quality_metrics['overall_quality'],
                        'completeness': quality_metrics['overall_quality'],
                        'engagement': quality_metrics['overall_quality']
                    })
                    
                    self.evolution_system.add_experience(
                        parameters=current_params,
                        quality_scores=quality_scores,
                        response_time=response_data.get('response_time', 0.0)
                    )
                    
                    # Evolve parameters if needed
                    if self.total_requests % self.evolution_config['evolution_frequency'] == 0:
                        print("   Triggering evolution...")
                        evolved_params = self.evolution_system.evolve_parameters()
                        self.llm_agent.update_evolution_params(evolved_params)
                        self.evolution_step += 1
                        self.llm_agent.set_evolution_step(self.evolution_step)
                        self.system_metrics['evolution_cycles'] += 1
                        print(f"   Evolution completed. New step: {self.evolution_step}")
                except Exception as e:
                    import traceback
                    print(f"[FAIL] Error in evolution system: {str(e)}")
                    print("Full traceback:")
                    traceback.print_exc()
                    raise e
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.system_metrics['total_processing_time'] += processing_time
            self.system_metrics['average_response_time'] = (
                self.system_metrics['total_processing_time'] / self.total_requests
            )
            
            # Compile complete results
            results = {
                'request_id': self.total_requests,
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'input_text': input_text,
                'processing_time': processing_time,
                'evolution_step': self.evolution_step,
                'classification_results': {
                    'binary_urgency': binary_urgency,
                    'detailed_urgency': detailed_urgency,
                    'content_category': content_category
                },
                'llm_response': response_data,
                'quality_metrics': quality_metrics,
                'system_state': {
                    'total_requests': self.total_requests,
                    'evolution_cycles': self.system_metrics['evolution_cycles'],
                    'average_response_time': self.system_metrics['average_response_time']
                }
            }
            
            print(f"\n[OK] Request processed successfully in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            error_result = {
                'request_id': self.total_requests,
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'input_text': input_text,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
            print(f"[FAIL] Error processing request: {str(e)}")
            return error_result
    
    def _classify_binary_urgency(self, input_text: str) -> str:
        """Classify binary urgency using the binary classifier"""
        try:
            if hasattr(self.binary_classifier, 'predict') and self.binary_classifier.model is not None:
                prediction = self.binary_classifier.predict([input_text])
                return 'urgent' if prediction[0] == 1 else 'not_urgent'
            else:
                # Fallback to LLM agent urgency classification
                urgency_result = self.llm_agent.classify_urgency(input_text)
                return 'urgent' if urgency_result['final_urgency'] == 'high' else 'not_urgent'
        except Exception as e:
            print(f"Warning: Binary urgency classification failed: {str(e)}")
            urgency_result = self.llm_agent.classify_urgency(input_text)
            return 'urgent' if urgency_result['final_urgency'] == 'high' else 'not_urgent'
    
    def _classify_detailed_urgency(self, input_text: str) -> str:
        """Classify detailed urgency using the multi-class classifier"""
        try:
            if hasattr(self.urgency_classifier, 'predict') and self.urgency_classifier.model is not None:
                prediction = self.urgency_classifier.predict([input_text])
                urgency_map = {0: 'low', 1: 'medium', 2: 'high'}
                return urgency_map.get(prediction[0], 'medium')
            else:
                # Fallback to LLM agent urgency classification
                urgency_result = self.llm_agent.classify_urgency(input_text)
                return urgency_result['final_urgency']
        except Exception as e:
            print(f"Warning: Detailed urgency classification failed: {str(e)}")
            urgency_result = self.llm_agent.classify_urgency(input_text)
            return urgency_result['final_urgency']
    
    def batch_process(self, 
                     inputs: List[str], 
                     enable_evolution: bool = True,
                     show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple requests in batch
        
        Args:
            inputs: List of input texts to process
            enable_evolution: Whether to enable evolution during batch processing
            show_progress: Whether to show progress information
            
        Returns:
            List of processing results
        """
        results = []
        
        if show_progress:
            print(f"\nProcessing batch of {len(inputs)} requests...")
        
        for i, input_text in enumerate(inputs):
            if show_progress:
                print(f"\nProcessing {i+1}/{len(inputs)}: {input_text[:50]}...")
            
            result = self.process_request(
                input_text=input_text,
                enable_evolution=enable_evolution
            )
            results.append(result)
        
        if show_progress:
            print(f"\nBatch processing completed. {len(results)} requests processed.")
        
        return results
    
    def get_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report"""
        # Get metrics report
        metrics_report = self.metrics_framework.get_comprehensive_report()
        
        # Get evolution summary
        evolution_summary = self.evolution_system.get_evolution_summary() if self.evolution_system else {}
        
        # Calculate quality improvement
        if len(self.metrics_framework.quality_scores) > 10:
            initial_quality = float(np.mean(list(self.metrics_framework.quality_scores)[:5]))
            recent_quality = float(np.mean(list(self.metrics_framework.quality_scores)[-5:]))
            self.system_metrics['quality_improvement'] = recent_quality - initial_quality
        
        report = {
            'system_info': {
                'session_id': self.session_id,
                'initialization_time': self.system_metrics['initialization_time'],
                'total_requests': self.total_requests,
                'evolution_step': self.evolution_step,
                'is_initialized': self.is_initialized
            },
            'performance_metrics': self.system_metrics,
            'quality_metrics': metrics_report,
            'evolution_metrics': evolution_summary,
            'component_status': {
                'binary_classifier': self.binary_classifier is not None,
                'urgency_classifier': self.urgency_classifier is not None,
                'llm_agent': self.llm_agent is not None,
                'evolution_system': self.evolution_system is not None,
                'metrics_framework': self.metrics_framework is not None
            }
        }
        
        return report
    
    def save_system_state(self, filepath: Optional[str] = None):
        """Save complete system state"""
        if filepath is None:
            filepath = os.path.join(self.save_dir, f"integrated_system_state_{self.session_id}.json")
        
        # Get system report
        system_state = self.get_system_report()
        
        # Add agent state if available
        if self.llm_agent:
            agent_state = self.llm_agent.get_performance_summary()
            system_state['agent_state'] = agent_state
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(system_state, f, indent=2)
        
        print(f"System state saved to: {filepath}")
        
        # Also save individual component states
        if self.llm_agent:
            agent_filepath = os.path.join(self.save_dir, f"llm_agent_state_{self.session_id}.json")
            self.llm_agent.save_state(agent_filepath)
        
        if self.evolution_system:
            self.evolution_system.save_evolution_state()
        
        if self.metrics_framework:
            metrics_filepath = os.path.join(self.save_dir, f"metrics_data_{self.session_id}.json")
            self.metrics_framework.save_metrics(metrics_filepath)
    
    def load_system_state(self, filepath: str):
        """Load system state from file"""
        with open(filepath, 'r') as f:
            system_state = json.load(f)
        
        # Restore system metrics
        if 'system_info' in system_state:
            self.total_requests = system_state['system_info'].get('total_requests', 0)
            # Ensure evolution_step is always an integer
            evolution_step_value = system_state['system_info'].get('evolution_step', 0)
            try:
                self.evolution_step = int(evolution_step_value)
            except (ValueError, TypeError):
                print(f"Warning: Invalid evolution_step value '{evolution_step_value}', defaulting to 0")
                self.evolution_step = 0
        
        if 'performance_metrics' in system_state:
            self.system_metrics.update(system_state['performance_metrics'])
        
        # Update agent evolution step
        if self.llm_agent:
            self.llm_agent.set_evolution_step(self.evolution_step)
        
        print(f"System state loaded from: {filepath}")

def main():
    """Demo of Integrated LLM System"""
    print("Integrated LLM System Demo")
    print("=" * 60)
    
    # Initialize system
    system = IntegratedLLMSystem()
    
    # Test cases
    test_inputs = [
        "I don't understand neural networks",
        "URGENT: My model is not training properly!",
        "Can you explain machine learning algorithms?",
        "Help! My code is throwing errors and I have a deadline tomorrow!",
        "What's the difference between supervised and unsupervised learning?"
    ]
    
    # Process requests
    results = system.batch_process(test_inputs, enable_evolution=True)
    
    # Generate and display report
    print("\n" + "=" * 60)
    print("SYSTEM PERFORMANCE REPORT")
    print("=" * 60)
    
    report = system.get_system_report()
    
    print(f"Total Requests Processed: {report['system_info']['total_requests']}")
    print(f"Evolution Steps: {report['system_info']['evolution_step']}")
    print(f"Average Response Time: {report['performance_metrics']['average_response_time']:.2f}s")
    
    if 'quality_metrics' in report and 'quality_statistics' in report['quality_metrics']:
        quality_stats = report['quality_metrics']['quality_statistics']
        print(f"Mean Quality Score: {quality_stats['mean_quality']:.3f}")
        print(f"Quality Improvement: {report['performance_metrics']['quality_improvement']:.3f}")
    
    # Save system state
    system.save_system_state()
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()