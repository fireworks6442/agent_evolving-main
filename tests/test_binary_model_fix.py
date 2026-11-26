#!/usr/bin/env python3
"""
Test script to verify binary model loading fix
"""

import os
import sys
from llm_response_agent import LLMResponseAgent

def test_binary_model_loading():
    """Test that the binary model loads correctly"""
    print("🧪 Testing binary model loading fix...")
    print("=" * 50)
    
    # Initialize the LLM Response Agent
    agent = LLMResponseAgent(
        binary_classifier_path='models/binary_urgency_classifier.joblib',
        multiclass_classifier_path='processed_data.csv',
        openai_api_key=None,  # Use mock mode
        model_name='gpt-3.5-turbo'
    )
    
    print("\n📊 Agent initialization complete!")
    print(f"Binary classifier available: {agent.binary_classifier is not None}")
    
    if agent.binary_classifier:
        print(f"Model loaded: {hasattr(agent.binary_classifier, 'loaded_model')}")
        
        # Test prediction
        test_text = "This is an urgent message that needs immediate attention!"
        print(f"\n🔍 Testing prediction with: '{test_text}'")
        
        try:
            result = agent.classify_urgency(test_text)
            print(f"✅ Prediction successful!")
            print(f"   Result: {result}")
            
            # Check if it's using the model or fallback
            if hasattr(agent.binary_classifier, 'loaded_model'):
                print("   🎯 Using trained model (not fallback)")
            else:
                print("   ⚠️  Using rule-based fallback")
                
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Test complete!")

if __name__ == "__main__":
    test_binary_model_loading()