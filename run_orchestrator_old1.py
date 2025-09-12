import os
import asyncio
from dotenv import load_dotenv
import openai
from orchestrator import Orchestrator
from aggregator import Aggregator
import logging
from pathlib import Path

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
openai.api_key = os.getenv("OPENAI_API_KEY")

async def run_async():
    """Run the orchestrator with async processing."""
    # Initialize
    orchestrator = Orchestrator()
    aggregator = Aggregator()
    
    
    query_list = [
        "What is the radial artery occlusion rate for TR Band vs VasoStat?",
        "How many radial artery compression devices are sold each year in the US and worldwide?",
        "What are the latest research findings on the relationship between intermittent fasting and autophagy in neurological conditions like Parkinson's disease, Alzheimer's, and multiple sclerosis?",
        "How many annual procedures are there that use the transradial access in the US and worldwide?",
        "What was the average cost savings per patient when using the transradial (TRI) approach compared to transfemoral (TFI) in the multi-center study by Amin et al.?",
        "What is the average selling price for the TR Band?",
        " What are the most common reported complications for the TR Band?",
        " When does the US TR Band patent expire?"
    ]
    
    # Example query
    # query = "What is the radial artery occlusion rate for TR Band vs VasoStat?"
    
    query = query_list[-1]
    logger.info(f"Query: {query}")
    context = {
        "model_id": "randy-data-testing",  # For local agent
        "db_name": "index",                # For PubMed agent
        "top_k": 5                         # For both agents
    }
    
    try:
        # Get responses from agents asynchronously
        logger.info("Starting async query processing")
        orchestrator_result = await orchestrator.process_query_async(query, context)
        
        # Aggregate responses with fallback support
        logger.info("Aggregating responses")
        final_result = aggregator.aggregate(orchestrator_result)
        
        # Print results
        print("\nFinal Answer:")
        print(final_result["answer"])
        print("\nCitations:")
        for citation in final_result["citations"]:
            print(f"- {citation}")
        
        # Print fallback status
        if final_result.get("fallback_used", False):
            print("\nNote: This response was generated using our fallback mechanism.")
            print(f"Reason: {final_result.get('fallback_reason', 'Primary agents failed to provide coherent responses')}")
            
    except Exception as e:
        logger.error(f"Critical error in main process: {str(e)}", exc_info=True)
        print(f"A critical error occurred: {str(e)}")

def main():
    """Run the orchestrator with synchronous processing."""
    # Initialize
    orchestrator = Orchestrator()
    aggregator = Aggregator()
    
    # Example query
    # query = "What is the radial artery occlusion rate for TR Band vs VasoStat?"
    
    # testing fallback mechanism
    query = "What is the relationship between intermittent fasting and autophagy in neurological conditions like Parkinson's disease, Alzheimer's, and multiple sclerosis, considering the latest contradictory research findings?"
    logger.info(f"Query: {query}")
    
    context = {
        "model_id": "randy-data-testing",  # For local agent
        "db_name": "index",                # For PubMed agent
        "top_k": 5                         # For both agents
    }
    
    try:
        # Get responses from agents
        logger.info("Starting synchronous query processing")
        logger.info(f"Query: {query}")
        orchestrator_result = orchestrator.process_query(query, context)
        
        # Aggregate responses with fallback support
        logger.info("Aggregating responses")
        final_result = aggregator.aggregate(orchestrator_result)
        
        # Print results
        print("\nFinal Answer:")
        print(final_result["answer"])
        print("\nCitations:")
        for citation in final_result["citations"]:
            print(f"- {citation}")
        
        # Print fallback status
        if final_result.get("fallback_used", False):
            print("\nNote: This response was generated using our fallback mechanism.")
            print(f"Reason: {final_result.get('fallback_reason', 'Primary agents failed to provide coherent responses')}")
            
    except Exception as e:
        logger.error(f"Critical error in main process: {str(e)}", exc_info=True)
        print(f"A critical error occurred: {str(e)}")

if __name__ == "__main__":
    # Choose between sync and async execution
    use_async = True
    
    logger.info("Starting medical research agent system")
    
    if use_async:
        logger.info("Using asynchronous processing")
        asyncio.run(run_async())
    else:
        logger.info("Using synchronous processing")
        main()
        
    logger.info("Process completed")