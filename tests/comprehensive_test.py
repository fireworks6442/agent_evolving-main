"""
Comprehensive test suite for the enhanced Response Agent
Tests all improvements and measures metric improvements
"""

import time
import json
from response_agent import ResponseAgent

def run_comprehensive_tests():
    """Run comprehensive tests to validate all improvements"""
    print("Starting Comprehensive Response Agent Testing...")
    print("=" * 60)
    
    # Initialize agent
    agent = ResponseAgent()
    
    # Comprehensive test cases covering all scenarios
    test_cases = [
        # High urgency cases
        {
            "text": "URGENT: I can't access the assignment and the deadline is in 2 hours! Please help ASAP!",
            "expected_urgency": "high",
            "expected_topic": "assignment",
            "description": "High urgency assignment access issue"
        },
        {
            "text": "CRITICAL: The system crashed and I lost all my work. Final project due tomorrow!",
            "expected_urgency": "high", 
            "expected_topic": "technical",
            "description": "Critical technical issue with deadline"
        },
        {
            "text": "EMERGENCY: Can't submit my exam, getting error messages. Need immediate help!",
            "expected_urgency": "high",
            "expected_topic": "exam",
            "description": "Emergency exam submission issue"
        },
        
        # Medium urgency cases
        {
            "text": "I'm having trouble understanding the homework instructions. Due next week.",
            "expected_urgency": "medium",
            "expected_topic": "assignment", 
            "description": "Medium urgency homework help"
        },
        {
            "text": "The grade for my project seems incorrect. Could you please check?",
            "expected_urgency": "medium",
            "expected_topic": "grade",
            "description": "Grade inquiry"
        },
        {
            "text": "I need clarification on the exam format before studying.",
            "expected_urgency": "medium",
            "expected_topic": "exam",
            "description": "Exam format inquiry"
        },
        
        # Low urgency cases
        {
            "text": "Could you explain the concept of machine learning from lecture 3?",
            "expected_urgency": "low",
            "expected_topic": "academic",
            "description": "General academic question"
        },
        {
            "text": "Thanks for the great course. I'm enjoying the materials.",
            "expected_urgency": "low",
            "expected_topic": "general",
            "description": "Positive feedback"
        },
        {
            "text": "What are the office hours for next week?",
            "expected_urgency": "low",
            "expected_topic": "general",
            "description": "General information request"
        },
        
        # Edge cases
        {
            "text": "Help me please I don't know what to do",
            "expected_urgency": "medium",
            "expected_topic": "general",
            "description": "Vague request for help"
        },
        {
            "text": "The deadline is approaching and I'm stressed about the project requirements",
            "expected_urgency": "medium",
            "expected_topic": "assignment",
            "description": "Stress-related deadline concern"
        }
    ]
    
    # Track results
    results = {
        "total_tests": len(test_cases),
        "urgency_correct": 0,
        "topic_correct": 0,
        "quality_scores": [],
        "response_times": [],
        "template_effectiveness": [],
        "detailed_results": []
    }
    
    print(f"Running {len(test_cases)} test cases...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['description']}")
        print(f"Input: {test_case['text']}")
        
        start_time = time.time()
        response_data = agent.generate_response(test_case['text'])
        end_time = time.time()
        
        # Check urgency classification accuracy
        urgency_correct = response_data['urgency_label'] == test_case['expected_urgency']
        if urgency_correct:
            results["urgency_correct"] += 1
            
        # Check topic classification accuracy  
        topic_correct = response_data['topic'] == test_case['expected_topic']
        if topic_correct:
            results["topic_correct"] += 1
            
        # Collect metrics
        results["quality_scores"].append(response_data['quality_score'])
        results["response_times"].append(response_data['response_time'])
        
        # Calculate template effectiveness (non-zero quality indicates effective template)
        template_effective = response_data['quality_score'] > 0.3
        results["template_effectiveness"].append(1 if template_effective else 0)
        
        # Store detailed result
        detailed_result = {
            "test_id": i,
            "description": test_case['description'],
            "input": test_case['text'],
            "expected_urgency": test_case['expected_urgency'],
            "actual_urgency": response_data['urgency_label'],
            "urgency_correct": urgency_correct,
            "expected_topic": test_case['expected_topic'],
            "actual_topic": response_data['topic'],
            "topic_correct": topic_correct,
            "quality_score": response_data['quality_score'],
            "response_time": response_data['response_time'],
            "template_used": response_data['template_used'],
            "keywords_found": response_data['keywords_found'],
            "response": response_data['response'][:100] + "..." if len(response_data['response']) > 100 else response_data['response']
        }
        results["detailed_results"].append(detailed_result)
        
        # Print results
        print(f"Expected Urgency: {test_case['expected_urgency']} | Actual: {response_data['urgency_label']} | {'[OK]' if urgency_correct else '[FAIL]'}")
        print(f"Expected Topic: {test_case['expected_topic']} | Actual: {response_data['topic']} | {'[OK]' if topic_correct else '[FAIL]'}")
        print(f"Quality Score: {response_data['quality_score']:.3f}")
        print(f"Response Time: {response_data['response_time']:.3f}s")
        print(f"Template: {response_data['template_used']}")
        print(f"Keywords: {response_data['keywords_found']}")
        print(f"Response: {response_data['response'][:100]}{'...' if len(response_data['response']) > 100 else ''}")
        print("-" * 50)
    
    # Calculate final metrics
    urgency_accuracy = (results["urgency_correct"] / results["total_tests"]) * 100
    topic_accuracy = (results["topic_correct"] / results["total_tests"]) * 100
    avg_quality = sum(results["quality_scores"]) / len(results["quality_scores"])
    avg_response_time = sum(results["response_times"]) / len(results["response_times"])
    template_effectiveness = (sum(results["template_effectiveness"]) / len(results["template_effectiveness"])) * 100
    
    # Print comprehensive results
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    print(f"Total Tests: {results['total_tests']}")
    print(f"Urgency Classification Accuracy: {urgency_accuracy:.1f}% ({results['urgency_correct']}/{results['total_tests']})")
    print(f"Topic Classification Accuracy: {topic_accuracy:.1f}% ({results['topic_correct']}/{results['total_tests']})")
    print(f"Average Quality Score: {avg_quality:.3f} ({avg_quality*100:.1f}%)")
    print(f"Template Effectiveness: {template_effectiveness:.1f}%")
    print(f"Average Response Time: {avg_response_time:.3f}s")
    
    # Performance summary from agent
    print("\nAgent Performance Summary:")
    summary = agent.get_performance_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Save detailed results to file
    with open('test_results.json', 'w') as f:
        json.dump({
            "summary": {
                "total_tests": results["total_tests"],
                "urgency_accuracy": urgency_accuracy,
                "topic_accuracy": topic_accuracy,
                "average_quality_score": avg_quality,
                "template_effectiveness": template_effectiveness,
                "average_response_time": avg_response_time
            },
            "detailed_results": results["detailed_results"],
            "agent_summary": summary
        }, f, indent=2)
    
    print(f"\nDetailed results saved to 'test_results.json'")
    
    # Improvement recommendations
    print("\n" + "=" * 60)
    print("IMPROVEMENT RECOMMENDATIONS")
    print("=" * 60)
    
    if urgency_accuracy < 90:
        print(f"⚠️  Urgency classification needs improvement (current: {urgency_accuracy:.1f}%, target: >90%)")
        
    if avg_quality < 0.7:
        print(f"⚠️  Response quality needs improvement (current: {avg_quality*100:.1f}%, target: >70%)")
        
    if template_effectiveness < 85:
        print(f"⚠️  Template effectiveness needs improvement (current: {template_effectiveness:.1f}%, target: >85%)")
        
    if urgency_accuracy >= 90 and avg_quality >= 0.7 and template_effectiveness >= 85:
        print("✅ All metrics meet target requirements!")
    
    return results

if __name__ == "__main__":
    run_comprehensive_tests()