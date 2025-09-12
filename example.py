import os
from dotenv import load_dotenv
import openai
from orchestrator_old import Orchestrator
from aggregator import Aggregator

# Load environment variables
load_dotenv()

# Configure OpenAI with environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    # Initialize
    orchestrator = Orchestrator()
    aggregator = Aggregator()
    
    # Example query
    query = "What is the radial artery occlusion rate for TR Band vs VasoStat?"
# "Are post-device manipulations more common with VasoStat or TR Band?"
    context = {
        "model_id": "randy-data-testing",
        "top_k": 5
    }
    
    try:
        # Get responses from agents
        agent_responses = orchestrator.process_query(query, context)
        
        # Aggregate responses
        final_result = aggregator.aggregate(query, agent_responses)
        
        # Print results
        print("\nFinal Answer:")
        print(final_result["answer"])
        print("\nCitations:")
        for citation in final_result["citations"]:
            print(f"- {citation}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()