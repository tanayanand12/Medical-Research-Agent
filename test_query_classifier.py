"""
test_query_classifier.py
~~~~~~~~~~~~~~~~~~~~~~~

Test script for the medical query classification system.
Evaluates the accuracy of the classifier on various medical and non-medical queries.
"""

import logging
from pathlib import Path
from query_classifier import QueryClassifier
import json
from dotenv import load_dotenv

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/test_classifier.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    """Test the query classifier on a variety of example queries."""
    # Initialize the classifier
    classifier = QueryClassifier()
    
    # Test queries
    test_queries = [
        # Clear medical research queries
        {
            "query": "What is the radial artery occlusion rate for TR Band vs VasoStat?",
            "expected": True,
            "category": "Medical research"
        },
        {
            "query": "What are the latest research findings on autophagy in neurological conditions?",
            "expected": True,
            "category": "Medical research"
        },
        {
            "query": "Compare the efficacy of SGLT2 inhibitors versus GLP-1 receptor agonists in type 2 diabetes patients.",
            "expected": True,
            "category": "Medical research"
        },
        {
            "query": "What does PubMed say about the use of monoclonal antibodies for autoimmune diseases?",
            "expected": True,
            "category": "Medical research"
        },
        
        # Clear non-medical queries
        {
            "query": "What are the best stocks to invest in for 2025?",
            "expected": False,
            "category": "Finance"
        },
        {
            "query": "Can you recommend some good movies to watch this weekend?",
            "expected": False,
            "category": "Entertainment"
        },
        {
            "query": "What is the recipe for chocolate chip cookies?",
            "expected": False,
            "category": "Cooking"
        },
        
        # Ambiguous or edge cases
        {
            "query": "What is the effect of exercise on health?",
            "expected": None,  # Ambiguous - could be medical research or general health
            "category": "Health/Medical (ambiguous)"
        },
        {
            "query": "How does the healthcare system work in the United States?",
            "expected": None,  # Ambiguous - policy rather than research
            "category": "Healthcare policy (ambiguous)"
        },
        {
            "query": "Tell me about medical school requirements.",
            "expected": None,  # Ambiguous - education rather than research
            "category": "Medical education (ambiguous)"
        },
        {
            "query": "What are normal blood pressure ranges?",
            "expected": None,  # Ambiguous - general medical knowledge vs research
            "category": "General medical (ambiguous)"
        }
    ]
    
    # Run tests and collect results
    results = []
    correct = 0
    total_with_expected = 0
    
    for i, test in enumerate(test_queries):
        query = test["query"]
        expected = test["expected"]
        category = test["category"]
        
        logger.info(f"Testing query {i+1}: {query}")
        is_medical, classification = classifier.is_medical_research_query(query)
        
        # Check if result matches expectation (if provided)
        if expected is not None:
            total_with_expected += 1
            if is_medical == expected:
                correct += 1
                result = "CORRECT"
            else:
                result = "INCORRECT"
        else:
            result = "AMBIGUOUS (no expected value)"
        
        results.append({
            "query": query,
            "category": category,
            "expected": expected,
            "classified_as_medical": is_medical,
            "confidence": classification.get("confidence", 0),
            "domain": classification.get("domain", "unknown"),
            "reason": classification.get("reason", ""),
            "result": result
        })
        
        # Print results for this query
        print(f"\n--- Query {i+1} ---")
        print(f"Query: {query}")
        print(f"Category: {category}")
        print(f"Expected: {expected}")
        print(f"Classified as medical: {is_medical}")
        print(f"Confidence: {classification.get('confidence', 0)}")
        print(f"Domain: {classification.get('domain', 'unknown')}")
        print(f"Result: {result}")
    
    # Calculate accuracy for queries with expected values
    if total_with_expected > 0:
        accuracy = correct / total_with_expected * 100
    else:
        accuracy = "N/A"
    
    # Print summary
    print("\n" + "="*50)
    print(f"SUMMARY: {correct}/{total_with_expected} correct classifications ({accuracy}% accuracy)")
    print("="*50)
    
    # Save detailed results to file
    with open("classifier_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Classification testing complete. Accuracy: {accuracy}%")
    logger.info(f"Detailed results saved to classifier_test_results.json")

if __name__ == "__main__":
    logger.info("Starting query classifier test")
    main()
    logger.info("Testing completed")