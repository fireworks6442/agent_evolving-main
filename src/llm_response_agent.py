"""
LLM Response Agent Module
Integrates existing classifiers with GPT API for intelligent response generation
"""

import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import openai
from binary_urgency_classifier import BinaryUrgencyClassifier
from urgency_classifier import UrgencyClassifier

class LLMResponseAgent:
    def __init__(self, 
                 binary_classifier_path: str = 'models/binary_urgency_classifier.joblib',
                 multiclass_classifier_path: str = 'processed_data.csv',
                 openai_api_key: str = None,
                 model_name: str = 'gpt-3.5-turbo',
                 evolution_step: int = 0):
        """
        Initialize the LLM Response Agent
        
        Args:
            binary_classifier_path: Path to binary urgency classifier
            multiclass_classifier_path: Path to multiclass urgency classifier data
            openai_api_key: OpenAI API key (if None, uses environment variable)
            model_name: OpenAI model to use
            evolution_step: Current evolution step for prompt optimization
        """
        self.model_name = model_name
        self.evolution_step = evolution_step
        
        # Initialize OpenAI client
        if openai_api_key is None:
            openai_api_key = os.getenv('OPENAI_API_KEY')
        
        self.mock_mode = False
        if not openai_api_key:
            print("Warning: OpenAI API key not provided. Running in mock mode.")
            self.mock_mode = True
            self.client = None
        else:
            try:
                openai.api_key = openai_api_key
                self.client = openai
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}. Running in mock mode.")
                self.mock_mode = True
                self.client = None
        
        # Initialize classifiers
        self.binary_classifier = None
        self.multiclass_classifier = None
        
        # Load binary classifier with automatic retraining
        try:
            self.binary_classifier = BinaryUrgencyClassifier()
            model_loaded = False
            
            if os.path.exists(binary_classifier_path):
                try:
                    model_data = self.binary_classifier.load_binary_model(binary_classifier_path)
                    if model_data is not None and hasattr(self.binary_classifier, 'loaded_model'):
                        print(f"✅ Binary classifier loaded successfully from {binary_classifier_path}")
                        model_loaded = True
                    else:
                        print(f"⚠️  Binary classifier model file exists but failed to load properly")
                except Exception as load_error:
                    print(f"⚠️  Error loading binary classifier model: {load_error}")
            
            if not model_loaded:
                print(f"🔄 Binary classifier model not available, attempting to train/retrain...")
                try:
                    # Use train_or_load_model which handles both training and loading
                    model_data = self.binary_classifier.train_or_load_model(force_retrain=False)
                    if model_data is not None and hasattr(self.binary_classifier, 'loaded_model'):
                        print(f"✅ Binary classifier trained/loaded successfully")
                        model_loaded = True
                    else:
                        print(f"❌ Failed to train/load binary classifier")
                except Exception as train_error:
                    print(f"❌ Error training binary classifier: {train_error}")
            
            if not model_loaded:
                print(f"⚠️  Binary classifier will use rule-based fallback")
                
        except Exception as e:
            print(f"❌ Critical error with binary classifier: {e}")
            self.binary_classifier = None
        
        # Load multiclass classifier
        try:
            if os.path.exists(multiclass_classifier_path):
                self.multiclass_classifier = UrgencyClassifier(
                    data_path=multiclass_classifier_path,
                    classification_type='ternary'
                )
                print(f"Multiclass classifier initialized with data from {multiclass_classifier_path}")
            else:
                print(f"Multiclass classifier data not found at {multiclass_classifier_path}")
        except Exception as e:
            print(f"Error loading multiclass classifier: {e}")
        
        # Evolution parameters - Starting with weaker defaults for better demonstration
        self.evolution_params = {
            'temperature': 0.3,  # Much lower - more conservative/rigid responses initially
            'max_tokens': 150,   # Shorter responses initially
            'top_p': 0.5,        # More focused, less creative initially
            'frequency_penalty': 0.2,  # Some repetition penalty
            'presence_penalty': 0.1,   # Slight presence penalty
            'system_prompt_style': 'basic',     # Start with basic style
            'response_length': 'short',         # Shorter responses
            'formality_level': 'casual',        # Less formal initially
            'empathy_level': 'low'              # Lower empathy initially
        }
        
        # Content categories for classification
        self.content_categories = {
            'learning': ['understand', 'learn', 'explain', 'concept', 'theory', 'how does', 'what is'],
            'technical': ['error', 'bug', 'code', 'implementation', 'algorithm', 'programming', 'debug'],
            'administrative': ['deadline', 'submission', 'grade', 'assignment', 'exam', 'schedule'],
            'general': ['question', 'help', 'advice', 'guidance', 'support']
        }
        
        # Performance tracking
        self.response_history = []
        self.performance_metrics = {
            'total_responses': 0,
            'avg_response_time': 0.0,
            'classification_accuracy': 0.0,
            'user_satisfaction': 0.0
        }
    
    def classify_urgency(self, text: str) -> Dict[str, Any]:
        """
        Classify urgency using both binary and multiclass classifiers
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with urgency classification results
        """
        results = {
            'binary_urgency': None,
            'binary_confidence': None,
            'multiclass_urgency': None,
            'multiclass_confidence': None,
            'final_urgency': 'medium',
            'urgency_score': 0.5
        }
        
        # Binary classification
        if self.binary_classifier:
            try:
                # Generate basic numerical features for the text
                numerical_features = self._extract_basic_features(text)
                binary_result = self.binary_classifier.predict_urgency(text, numerical_features)
                if binary_result is not None:
                    results['binary_urgency'] = binary_result['urgency_label']
                    results['binary_confidence'] = binary_result['confidence']
                else:
                    print("Binary classifier returned None - model not loaded")
            except Exception as e:
                print(f"Error in binary classification: {e}")
        
        # Multiclass classification (if available)
        if self.multiclass_classifier:
            try:
                # This would require implementing predict method in UrgencyClassifier
                # For now, we'll use keyword-based classification
                pass
            except Exception as e:
                print(f"Error in multiclass classification: {e}")
        
        # Determine final urgency based on available classifications
        if results['binary_urgency']:
            if results['binary_urgency'] == 'High Urgency':
                results['final_urgency'] = 'high'
                results['urgency_score'] = 0.8
            else:
                results['final_urgency'] = 'low'
                results['urgency_score'] = 0.3
        
        # Enhance with keyword-based analysis
        text_lower = text.lower()
        urgency_keywords = {
            'high': ['urgent', 'emergency', 'critical', 'asap', 'immediately', 'deadline', 'help!'],
            'medium': ['soon', 'important', 'need', 'problem', 'issue', 'stuck'],
            'low': ['question', 'wondering', 'curious', 'when possible', 'eventually']
        }
        
        keyword_score = 0.5  # default medium
        for level, keywords in urgency_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                if level == 'high':
                    keyword_score = max(keyword_score, 0.9)
                elif level == 'medium':
                    keyword_score = max(keyword_score, 0.6)
                elif level == 'low':
                    keyword_score = min(keyword_score, 0.4)
        
        # Combine classifier and keyword scores
        if results['urgency_score'] != 0.5:  # If we have classifier result
            results['urgency_score'] = (results['urgency_score'] + keyword_score) / 2
        else:
            results['urgency_score'] = keyword_score
        
        # Update final urgency based on combined score
        if results['urgency_score'] >= 0.7:
            results['final_urgency'] = 'high'
        elif results['urgency_score'] >= 0.4:
            results['final_urgency'] = 'medium'
        else:
            results['final_urgency'] = 'low'
        
        return results
    
    def _extract_basic_features(self, text: str) -> list:
        """Extract basic numerical features from text for binary classifier"""
        import re
        
        # Basic text statistics
        text_length = len(text)
        words = text.split()
        word_count = len(words)
        sentence_count = len(re.split(r'[.!?]+', text))
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
        
        # Sentiment approximation (simple keyword-based)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate']
        pos_count = sum(1 for word in positive_words if word in text.lower())
        neg_count = sum(1 for word in negative_words if word in text.lower())
        sentiment_polarity = (pos_count - neg_count) / max(word_count, 1)
        sentiment_subjectivity = (pos_count + neg_count) / max(word_count, 1)
        
        # Punctuation features
        question_marks = text.count('?')
        exclamation_marks = text.count('!')
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Return features in the expected order
        return [
            text_length, word_count, sentence_count, avg_word_length,
            sentiment_polarity, sentiment_subjectivity, question_marks,
            exclamation_marks, uppercase_ratio
        ]
    
    def classify_content_category(self, text: str) -> str:
        """
        Classify content category based on keywords and context
        
        Args:
            text: Input text to classify
            
        Returns:
            Content category string
        """
        text_lower = text.lower()
        
        category_scores = {}
        for category, keywords in self.content_categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return 'general'
    
    def get_evolution_stage(self) -> str:
        """
        Determine evolution stage based on evolution step
        
        Returns:
            Evolution stage: 'basic', 'intermediate', or 'advanced'
        """
        evolution_step = int(self.evolution_step) if isinstance(self.evolution_step, str) else self.evolution_step
        if evolution_step < 10:
            return 'basic'
        elif evolution_step < 30:
            return 'intermediate'
        else:
            return 'advanced'
    
    def generate_system_prompt(self, urgency: str, category: str, evolution_stage: str) -> str:
        """
        Generate dynamic system prompt based on urgency, category, and evolution stage
        
        Args:
            urgency: Urgency level ('low', 'medium', 'high')
            category: Content category
            evolution_stage: Current evolution stage
            
        Returns:
            System prompt string
        """
        base_prompt = "You are an intelligent educational assistant designed to help students and learners."
        
        # Urgency-specific modifications
        urgency_prompts = {
            'high': " You should respond with immediate, actionable guidance and prioritize solving urgent problems quickly.",
            'medium': " You should provide helpful, structured responses that address the user's needs efficiently.",
            'low': " You should provide thoughtful, comprehensive responses that encourage learning and exploration."
        }
        
        # Category-specific modifications
        category_prompts = {
            'learning': " Focus on explaining concepts clearly, providing examples, and encouraging understanding.",
            'technical': " Provide precise technical guidance, debugging steps, and practical solutions.",
            'administrative': " Give clear, direct information about procedures, deadlines, and requirements.",
            'general': " Offer balanced, helpful responses tailored to the user's specific question."
        }
        
        # Evolution stage modifications
        evolution_prompts = {
            'basic': " Keep responses clear and straightforward.",
            'intermediate': " Provide detailed explanations with examples and context.",
            'advanced': " Offer comprehensive, nuanced responses with multiple perspectives and advanced insights."
        }
        
        # Combine prompts
        system_prompt = (base_prompt + 
                        urgency_prompts.get(urgency, urgency_prompts['medium']) +
                        category_prompts.get(category, category_prompts['general']) +
                        evolution_prompts.get(evolution_stage, evolution_prompts['intermediate']))
        
        # Add evolution-specific parameters
        if evolution_stage == 'advanced':
            system_prompt += " Use sophisticated language and provide in-depth analysis."
        elif evolution_stage == 'basic':
            system_prompt += " Use simple, accessible language."
        
        return system_prompt
    
    def _generate_mock_response(self, text: str, urgency_result: Dict, category: str, evolution_stage: str) -> str:
        """Generate a mock response for testing when API is not available"""
        urgency = urgency_result.get('final_urgency', 'Low Urgency')
        base = ""
        if urgency == 'High Urgency':
            if 'technical' in category.lower():
                base = f"I understand this is an urgent technical issue. Let me help you resolve this problem quickly. Based on your description, here are the immediate steps you should take to address this {category.lower()} concern."
            else:
                base = f"I recognize the urgency of your {category.lower()} question. Let me provide you with a prompt and helpful response to address your immediate needs."
        else:
            if 'technical' in category.lower():
                base = f"Thank you for your {category.lower()} question. Here's a comprehensive explanation that should help you understand the topic better. Let me break this down step by step."
            else:
                base = f"I'd be happy to help with your {category.lower()} inquiry. Here's a detailed response that addresses your question thoroughly."
        # Enrich response based on evolution parameters to influence metrics
        params = self.evolution_params
        extras = []
        depth = float(params.get('explanation_depth', 0.5)) if isinstance(params.get('explanation_depth', 0.5), (int, float)) else 0.5
        tech = float(params.get('technical_level', 0.5)) if isinstance(params.get('technical_level', 0.5), (int, float)) else 0.5
        engage = float(params.get('engagement_factor', 0.5)) if isinstance(params.get('engagement_factor', 0.5), (int, float)) else 0.5
        temperature = float(params.get('temperature', 0.5)) if isinstance(params.get('temperature', 0.5), (int, float)) else 0.5
        if depth > 0.6:
            extras.append("First, identify the root cause; next, apply the recommended method; finally, verify and prevent future issues.")
            extras.append("Additionally, consider a simple example to solidify understanding.")
        if tech > 0.5:
            extras.append("From a technical perspective, focus on the algorithm, parameters, and optimization function involved.")
        if engage > 0.6:
            extras.append("You can try this approach now—does this solve your issue?")
        if temperature < 0.5:
            extras.append("Therefore, a clear and concise explanation is preferred to maintain clarity.")
        enriched = base + " " + " ".join(extras)
        return enriched
    
    def generate_response(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate response using GPT API with classified urgency and category
        
        Args:
            text: Input text to respond to
            context: Additional context information
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        # Classify input
        urgency_result = self.classify_urgency(text)
        category = self.classify_content_category(text)
        evolution_stage = self.get_evolution_stage()
        
        # Generate system prompt
        system_prompt = self.generate_system_prompt(
            urgency_result['final_urgency'], 
            category, 
            evolution_stage
        )
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        # Add context if provided
        if context and 'previous_messages' in context:
            for msg in context['previous_messages'][-3:]:  # Last 3 messages for context
                messages.insert(-1, msg)
        
        try:
            if self.mock_mode or self.client is None:
                # Generate mock response
                generated_text = self._generate_mock_response(text, urgency_result, category, evolution_stage)
                token_usage = {
                    'prompt_tokens': len(system_prompt.split()) + len(text.split()),
                    'completion_tokens': len(generated_text.split()),
                    'total_tokens': len(system_prompt.split()) + len(text.split()) + len(generated_text.split())
                }
            else:
                # Generate response using OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.evolution_params['temperature'],
                    max_tokens=int(self.evolution_params['max_tokens']),
                    top_p=self.evolution_params['top_p'],
                    frequency_penalty=self.evolution_params['frequency_penalty'],
                    presence_penalty=self.evolution_params['presence_penalty']
                )
                
                generated_text = response.choices[0].message.content
                token_usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Prepare result
            result = {
                'response': generated_text,
                'urgency_classification': urgency_result,
                'content_category': category,
                'evolution_stage': evolution_stage,
                'system_prompt': system_prompt,
                'response_time': response_time,
                'model_used': self.model_name if not self.mock_mode else 'mock-model',
                'evolution_step': self.evolution_step,
                'timestamp': datetime.now().isoformat(),
                'token_usage': token_usage
            }
            
            # Update performance metrics
            self.performance_metrics['total_responses'] += 1
            self.performance_metrics['avg_response_time'] = (
                (self.performance_metrics['avg_response_time'] * (self.performance_metrics['total_responses'] - 1) + 
                 response_time) / self.performance_metrics['total_responses']
            )
            
            # Store in history
            self.response_history.append(result)
            
            return result
            
        except Exception as e:
            error_result = {
                'error': str(e),
                'response': f"Error generating response: {str(e)}",
                'urgency_classification': urgency_result,
                'content_category': category,
                'evolution_stage': evolution_stage,
                'response_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"Error generating response: {e}")
            return error_result
    
    def update_evolution_params(self, new_params: Dict[str, Any]):
        """
        Update evolution parameters for prompt optimization
        
        Args:
            new_params: Dictionary of parameters to update
        """
        self.evolution_params.update(new_params)
        print(f"Evolution parameters updated: {new_params}")
    
    def set_evolution_step(self, step: int):
        """
        Set the current evolution step
        
        Args:
            step: Evolution step number
        """
        self.evolution_step = step
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary and metrics
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'performance_metrics': self.performance_metrics.copy(),
            'evolution_step': self.evolution_step,
            'evolution_stage': self.get_evolution_stage(),
            'total_responses_generated': len(self.response_history),
            'current_evolution_params': self.evolution_params.copy()
        }
    
    def save_state(self, filepath: str):
        """
        Save agent state to file
        
        Args:
            filepath: Path to save state file
        """
        state = {
            'evolution_step': self.evolution_step,
            'evolution_params': self.evolution_params,
            'performance_metrics': self.performance_metrics,
            'response_history': self.response_history[-10:],  # Last 10 responses
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        print(f"Agent state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """
        Load agent state from file
        
        Args:
            filepath: Path to state file
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Ensure evolution_step is always an integer
            evolution_step_value = state.get('evolution_step', 0)
            try:
                self.evolution_step = int(evolution_step_value)
            except (ValueError, TypeError):
                print(f"Warning: Invalid evolution_step value '{evolution_step_value}', defaulting to 0")
                self.evolution_step = 0
            self.evolution_params.update(state.get('evolution_params', {}))
            self.performance_metrics.update(state.get('performance_metrics', {}))
            self.response_history = state.get('response_history', [])
            
            print(f"Agent state loaded from {filepath}")
            
        except Exception as e:
            print(f"Error loading agent state: {e}")


def main():
    """
    Test the LLM Response Agent
    """
    print("Testing LLM Response Agent...")
    
    # Initialize agent
    agent = LLMResponseAgent()
    
    # Test inputs
    test_inputs = [
        "I do not understand neural networks",
        "URGENT: My code is not working and the deadline is tomorrow!",
        "Can you explain how machine learning works?",
        "What is the submission deadline for the final project?"
    ]
    
    print("\n" + "="*60)
    print("TESTING LLM RESPONSE AGENT")
    print("="*60)
    
    for i, text in enumerate(test_inputs, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: {text}")
        
        result = agent.generate_response(text)
        
        if 'error' not in result:
            print(f"Urgency: {result['urgency_classification']['final_urgency']}")
            print(f"Category: {result['content_category']}")
            print(f"Evolution Stage: {result['evolution_stage']}")
            print(f"Response Time: {result['response_time']:.2f}s")
            print(f"Response: {result['response'][:200]}...")
        else:
            print(f"Error: {result['error']}")
    
    # Print performance summary
    print(f"\n--- Performance Summary ---")
    summary = agent.get_performance_summary()
    print(f"Total Responses: {summary['total_responses_generated']}")
    print(f"Average Response Time: {summary['performance_metrics']['avg_response_time']:.2f}s")
    print(f"Evolution Step: {summary['evolution_step']}")
    print(f"Evolution Stage: {summary['evolution_stage']}")


if __name__ == "__main__":
    main()
