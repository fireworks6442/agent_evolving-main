#!/usr/bin/env python3
"""
Debug script to capture the exact traceback of the type error
"""

import traceback
import sys
from integrated_llm_system import IntegratedLLMSystem

def debug_type_error():
    """Run a simple test to capture the type error traceback"""
    try:
        print("Initializing IntegratedLLMSystem...")
        system = IntegratedLLMSystem()
        
        print("Processing request to trigger the error...")
        result = system.process_request("What is machine learning?", enable_evolution=True)
        
        print("Request processed successfully!")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("FULL TRACEBACK OF THE ERROR:")
        print(f"{'='*60}")
        
        # Print the full traceback
        traceback.print_exc()
        
        print(f"\n{'='*60}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(f"{'='*60}")
        
        # Also print to stderr for completeness
        print("\nFull traceback details:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

if __name__ == "__main__":
    debug_type_error()