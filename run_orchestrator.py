import os
import asyncio
from dotenv import load_dotenv
import openai
from orchestrator import Orchestrator
from aggregator import Aggregator
import logging
from pathlib import Path
from unicode_safe_logging import configure_all_loggers

# Configure all loggers to handle unicode safely
configure_all_loggers()



# ---- force UTF-8 for every process, every handler ----
import os, sys, io, functools

def _force_utf8_stream(stream):
    if not stream.encoding.lower().startswith("utf"):
        # Re-wrap original buffer with a UTF-8 TextIO
        return io.TextIOWrapper(stream.buffer, encoding="utf-8", newline='')
    return stream

sys.stdout = _force_utf8_stream(sys.stdout)
sys.stderr = _force_utf8_stream(sys.stderr)
os.environ["PYTHONIOENCODING"] = "utf-8"


# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenAI with environment variable
logger.info("Loading OpenAI API key from environment variables")
openai.api_key = os.getenv("OPENAI_API_KEY")

async def run_async():
    """Run the orchestrator with async processing to test both medical and non-medical queries."""
    # Initialize
    orchestrator = Orchestrator()
    aggregator = Aggregator()
    
    # Example queries - both medical and non-medical to test domain filtering
    medical_queries = [
        "What is the radial artery occlusion rate for TR Band vs VasoStat?",
        "How many radial artery compression devices are sold each year in the US and worldwide?",
        # "What are the latest research findings on the relationship between intermittent fasting and autophagy in neurological conditions like Parkinson's disease, Alzheimer's, and multiple sclerosis?",
        # "How many annual procedures are there that use the transradial access in the US and worldwide?",
        # "What was the average cost savings per patient when using the transradial (TRI) approach compared to transfemoral (TFI) in the multi-center study by Amin et al.?",
        # "What is the typical pain score for a patient that has the TR Band on a scale of 1-10?",
        # "What is the marketing rate of uptake since launch and what type of users are the best customers for the TR Band?",
        # "What is the average selling price for the VasoStat?What is the typical pain score for a patient that has the VasoStat on a scale of 1â€“10?",
        # "What is the marketing rate of uptake since launch and what type of users are the best customers for the VasoStat?",
        # "If the `Patent Hemostasis Technique` is employed, how difficult is the technique for the cath lab staff to carry out?",
        # "What is the difference in radial artery occlusion (RAO) rates at 24 hours and at 3 months between distal radial access (DRA) and traditional transradial access (TRA)?",
        # "What are the reported procedure times for transpedal access versus traditional retrograde femoral access in infrapopliteal interventions?",
        "How many transpedal arterial access studies have been completed or are ongoing?",
        "What are the most common inclusion and exclusion criteria for these studies?"


    ]
    
    non_medical_queries = [
        # "What are the best stocks to invest in for 2025?",
        # "Can you recommend some good movies to watch this weekend?",
        # "What is the recipe for chocolate chip cookies?",
        # "How do I improve my golf swing?",
        "What are the top tourist destinations in Europe?"
    ]
    
    # Run tests with both medical and non-medical queries
    # Select which queries to test
    test_medical = True
    test_non_medical = True
    
    context = {
        "model_id": "randy-data-testing",  # For local agent
        "db_name": "index",                # For PubMed agent
        "top_k": 5                         # For both agents
    }
    
    try:
        # Test medical queries
        if test_medical:
            logger.info("Testing medical research queries")
            for i, query in enumerate(medical_queries):
                logger.info(f"Processing medical query {i+1}: {query}")
                
                # Get responses from agents asynchronously
                orchestrator_result = await orchestrator.process_query_async(query, context)
                
                # Aggregate responses with fallback support
                final_result = aggregator.aggregate(orchestrator_result)
                
                # Print results
                print("\n" + "="*80)
                print(f"MEDICAL QUERY {i+1}: {query}")
                print("="*80)
                print("\nFinal Answer:")
                print(final_result["answer"])
                print("\nCitations:")
                for citation in final_result["citations"]:
                    print(f"- {citation}")
                
                # Print fallback status
                if final_result.get("fallback_used", False):
                    print("\nNote: This response was generated using our fallback mechanism.")
                    print(f"Reason: {final_result.get('fallback_reason', 'Primary agents failed to provide coherent responses')}")
                
                print("\n" + "="*80 + "\n")
        
        # Test non-medical queries
        if test_non_medical:
            logger.info("Testing non-medical queries")
            for i, query in enumerate(non_medical_queries):
                logger.info(f"Processing non-medical query {i+1}: {query}")
                
                # Get responses from agents asynchronously
                orchestrator_result = await orchestrator.process_query_async(query, context)
                
                # Aggregate responses with fallback support
                final_result = aggregator.aggregate(orchestrator_result)
                
                # Print results
                print("\n" + "="*80)
                print(f"NON-MEDICAL QUERY {i+1}: {query}")
                print("="*80)
                print("\nFinal Answer:")
                print(final_result["answer"])
                
                # Print fallback status
                if final_result.get("fallback_used", False):
                    print("\nNote: This response was generated using the domain filtering system.")
                    print(f"Reason: {final_result.get('fallback_reason', 'Query classified as non-medical research')}")
                
                print("\n" + "="*80 + "\n")
                
    except Exception as e:
        logger.error(f"Critical error in async process: {str(e)}", exc_info=True)
        print(f"A critical error occurred: {str(e)}")


def main():
    """Run the orchestrator with synchronous processing to test both medical and non-medical queries."""
    # Initialize
    orchestrator = Orchestrator()
    aggregator = Aggregator()
    
    # Example queries for testing
    medical_query = [
        "What is the relationship between intermittent fasting and autophagy in neurological conditions like Parkinson's disease?",
            ]

    non_medical_query = "What are the top investment strategies for cryptocurrency in 2025?"
    
    # context = {
    #     "model_id": "randy-data-testing",  # For local agent
    #     "db_name": "index",                # For PubMed agent
    #     "top_k": 5                         # For both agents
    # }
    
    context = {
        "model_id": "randy-data-testing",        # For local agent
        "db_name": "index",                      # For PubMed agent
        "top_k": 5,                             # For local and PubMed agents
        "clinical_trials_top_k": 10,            # Specific for clinical trials agent
        "max_trials": 25                        # For clinical trials agent
    }

    try:
        # Test a medical query
        logger.info("Testing a medical research query")
        logger.info(f"Query: {medical_query[-1]}")
        orchestrator_result = orchestrator.process_query(medical_query[-1], context)
        
        # Aggregate responses with fallback support
        logger.info("Aggregating responses")
        final_result = aggregator.aggregate(orchestrator_result)
        
        # Print results
        print("\n" + "="*80)
        print(f"MEDICAL QUERY: {medical_query[-1]}")
        print("="*80)
        print("\nFinal Answer:")
        print(final_result["answer"])
        print("\nCitations:")
        for citation in final_result["citations"]:
            print(f"- {citation}")
        
        # Print fallback status
        if final_result.get("fallback_used", False):
            print("\nNote: This response was generated using our fallback mechanism.")
            print(f"Reason: {final_result.get('fallback_reason', 'Primary agents failed to provide coherent responses')}")
        
        print("\n" + "="*80 + "\n")
        
        # Test a non-medical query
        logger.info("Testing a non-medical query")
        logger.info(f"Query: {non_medical_query}")
        orchestrator_result = orchestrator.process_query(non_medical_query, context)
        
        # Aggregate responses with fallback support
        logger.info("Aggregating responses")
        final_result = aggregator.aggregate(orchestrator_result)
        
        # Print results
        print("\n" + "="*80)
        print(f"NON-MEDICAL QUERY: {non_medical_query}")
        print("="*80)
        print("\nFinal Answer:")
        print(final_result["answer"])
        
        # Print fallback status
        if final_result.get("fallback_used", False):
            print("\nNote: This response was generated using the domain filtering system.")
            print(f"Reason: {final_result.get('fallback_reason', 'Query classified as non-medical research')}")
        
        print("\n" + "="*80 + "\n")
            
    except Exception as e:
        logger.error(f"Critical error in main process: {str(e)}", exc_info=True)
        print(f"A critical error occurred: {str(e)}")


# This section was missing from your file
if __name__ == "__main__":
    # Choose between sync and async execution
    use_async = True
    
    logger.info("Starting medical research agent system with domain filtering")
    
    if use_async:
        logger.info("Using asynchronous processing")
        asyncio.run(run_async())
    else:
        logger.info("Using synchronous processing")
        main()
        
    logger.info("Process completed")