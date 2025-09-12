import os
import asyncio
import csv
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
        logging.FileHandler("logs/query_results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenAI with environment variable
logger.info("Loading OpenAI API key from environment variables")
openai.api_key = os.getenv("OPENAI_API_KEY")

# CSV file setup
CSV_FILE = "medical_query_results.csv"

def setup_csv():
    """Create CSV file with headers if it doesn't exist"""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Question', 'Answer', 'Citations'])
        logger.info(f"Created new CSV file: {CSV_FILE}")
    else:
        logger.info(f"CSV file already exists: {CSV_FILE}")

def append_result_to_csv(question, answer, citations):
    """Append a single result to the CSV file"""
    # Handle citations that could be dict objects or other types
    if citations and isinstance(citations, list):
        processed_citations = []
        for citation in citations:
            if isinstance(citation, dict):
                # If citation is a dict, convert relevant fields to string
                if 'title' in citation and 'authors' in citation:
                    processed_citations.append(f"{citation.get('title', '')} by {citation.get('authors', '')}")
                else:
                    # Use all available keys in the dict
                    processed_citations.append(str(citation))
            elif citation is not None:
                processed_citations.append(str(citation))
        citations_text = '; '.join(processed_citations)
    else:
        # If citations is not a list or is empty
        citations_text = str(citations) if citations else ""
    
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([question, answer, citations_text])
    logger.info(f"Added result for question: {question[:50]}...")

async def process_question(question, orchestrator, aggregator, context):
    """Process a single question and return the result"""
    logger.info(f"Processing question: {question}")
    try:
        # Get responses from agents asynchronously
        orchestrator_result = await orchestrator.process_query_async(question, context)
        
        # Aggregate responses with fallback support
        final_result = aggregator.aggregate(orchestrator_result)
        
        # Log result status
        if final_result.get("fallback_used", False):
            logger.warning(f"Fallback used for question: {question}")
            logger.warning(f"Reason: {final_result.get('fallback_reason', 'Unknown')}")
        
        # Debug log the citations structure
        citations = final_result.get("citations", [])
        if citations:
            logger.debug(f"Citations type: {type(citations)}")
            if isinstance(citations, list) and citations:
                logger.debug(f"First citation type: {type(citations[0])}")
        
        return final_result
    except Exception as e:
        logger.error(f"Error processing question: {question}")
        logger.error(f"Error details: {str(e)}", exc_info=True)
        return {
            "answer": f"Error processing query: {str(e)}",
            "citations": ["Error occurred"],
            "fallback_used": True,
            "fallback_reason": f"Exception: {str(e)}"
        }

async def run_questions():
    """Run all questions and save results to CSV"""
    # Initialize components
    orchestrator = Orchestrator()
    aggregator = Aggregator()
    
    # Set up context
    context = {
        "model_id": "randy-data-testing",  # For local agent
        "db_name": "index",                # For PubMed agent
        "top_k": 5                         # For both agents
    }
    
    # Define all questions
    questions = [
        # TR Band Questions
        "What is the typical pain score for a patient that has the TR Band on a scale of 1-10?",
        "What is the rate of subdermal bleed when using TR Band?",
        "What is the rate of radial artery occlusion associated with compression closure devices?",
        "Can a radial artery occlusion prevent the use of that artery for future interventional procedures?",
        "How many clinical studies have been conducted on the TR Band?",
        "What is the marketing rate of uptake since launch and what type of users are the best customers for the TR Band?",
        
        # Questions Based On Randy's Data
        "What was the average cost savings per patient when using the transradial (TRI) approach compared to transfemoral (TFI) in the multi-center study by Amin et al.?",
        "How did the addition of the StatSeal Advanced Disc impact time to hemostasis compared to TR Band alone in Condry et al.'s quality improvement study?",
        "According to the HANGAR Study, did long-term radial artery occlusion have any measurable impact on hand grip strength or thumb and forefinger function?",
        "In the DISCO RADIAL trial, what was the difference in hemostasis time between distal radial access (DRA) and traditional transradial access (TRA)?",
        "What was the reported incidence and severity distribution (Grade I-IV) of forearm hematoma after transradial intervention in the Indian single-center study by Dwivedi et al.?",
        "How did the AIR band perform compared to the TR Band in terms of radial artery occlusion and compression removal time?",
        "In the RIVAL study, how did procedural success and complication rates vary across high-volume vs. low-volume centers using radial access?",
        "What access-related complications were observed in the PCVI-CUBA trial when comparing transulnar vs. transradial approaches for coronary angioplasty?",
        "According to the 2022 trial comparing TR Band and VasoStat in transpedal access, what were the 30-day complication rates and hemostasis outcomes?",
        "From FDA MAUDE reports, what was the nature of the adverse events associated with TR Band malfunction and how were they clinically managed?",
        
        # VasoStat Questions
        "What is the average selling price for the VasoStat?",
        "What are the most common reported complications for the VasoStat?",
        "When does the US VasoStat patent expire?",
        "How many times does a clinician have to manipulate the VasoStat once it is placed on the patient's wrist?",
        "What is the typical pain score for a patient that has the VasoStat on a scale of 1-10?",
        "What is the rate of subdermal bleed when using VasoStat?",
        "How many clinical studies have been conducted on the VasoStat?",
        "What is the marketing rate of uptake since launch and what type of users are the best customers for the VasoStat?",
        "How often is the \"Patent Hemostasis Technique\" ignored/not employed in the cath lab setting?",
        "If the \"Patent Hemostasis Technique\" is employed, how difficult is the technique for the cath lab staff to carry out?"
    ]
    
    # Set up CSV file
    setup_csv()
    
    total_questions = len(questions)
    
    # Process each question and save results
    for i, question in enumerate(questions):
        try:
            # Log progress
            logger.info(f"Processing question {i+1}/{total_questions}")
            print(f"\n[{i+1}/{total_questions}] Processing: {question}")
            
            # Process the question
            result = await process_question(question, orchestrator, aggregator, context)
            
            # Get answer and citations
            answer = result["answer"]
            citations = result.get("citations", [])
            
            # Append to CSV
            append_result_to_csv(question, answer, citations)
            
            # Print progress update
            if result.get("fallback_used", False):
                status = "FALLBACK USED"
            else:
                status = "SUCCESS"
            
            print(f"✓ {status}: {question[:50]}...")
            
            # Optional: Add a small delay between requests to prevent rate limiting
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Critical error processing question {i+1}: {str(e)}", exc_info=True)
            print(f"✗ ERROR with question {i+1}: {question}")
            
            # Still add to CSV but with error message
            append_result_to_csv(
                question, 
                f"Error processing query: {str(e)}", 
                ["Error occurred"]
            )
    
    logger.info(f"All questions processed. Results saved to {CSV_FILE}")
    print(f"\nAll {total_questions} questions processed.")
    print(f"Results saved to {CSV_FILE}")

if __name__ == "__main__":
    logger.info("Starting batch processing of medical research queries")
    print("Starting batch processing of medical research queries...")
    print(f"Results will be saved to {CSV_FILE}")
    
    # Run all questions
    asyncio.run(run_questions())
    
    logger.info("Process completed")
    print("\nProcess completed!")

################################################################
# import os
# import asyncio
# import pandas as pd
# import numpy as np
# from dotenv import load_dotenv
# import logging
# from pathlib import Path
# from orchestrator import Orchestrator
# from aggregator import Aggregator

# # Ensure logs directory exists
# Path("logs").mkdir(exist_ok=True)

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("logs/process_questions.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# async def process_question(question, orchestrator, aggregator, context):
#     """Process a single question using the orchestrator and aggregator."""
#     try:
#         # Skip processing if question is NaN
#         if pd.isna(question):
#             logger.info("Skipping NaN question")
#             return {
#                 "new_clinical_response": "",
#                 "new_clinical_sources": ""
#             }
        
#         # Convert question to string if needed
#         question = str(question)
#         logger.info(f"Processing question: {question}")
        
#         # Get responses from agents asynchronously
#         orchestrator_result = await orchestrator.process_query_async(question, context)
        
#         # Aggregate responses with fallback support
#         final_result = aggregator.aggregate(orchestrator_result)
        
#         # Extract citations safely
#         citations_str = ""
#         if "citations" in final_result and final_result["citations"]:
#             citations = final_result["citations"]
#             if isinstance(citations, list):
#                 formatted_citations = []
#                 for citation in citations:
#                     if isinstance(citation, dict) and "citation" in citation:
#                         formatted_citations.append(citation["citation"])
#                     elif isinstance(citation, dict):
#                         formatted_citations.append(str(citation))
#                     else:
#                         formatted_citations.append(str(citation))
#                 citations_str = "; ".join(formatted_citations)
#             else:
#                 citations_str = str(citations)
        
#         return {
#             "new_clinical_response": final_result["answer"],
#             "new_clinical_sources": citations_str
#         }
#     except Exception as e:
#         logger.error(f"Error processing question '{question}': {str(e)}", exc_info=True)
#         return {
#             "new_clinical_response": f"Error: {str(e)}",
#             "new_clinical_sources": ""
#         }

# async def process_csv(csv_path, output_csv_path):
#     """Process all questions in the CSV file and save results to a new CSV."""
#     # Initialize
#     orchestrator = Orchestrator()
#     aggregator = Aggregator()
    
#     context = {
#         "model_id": "randy-data-testing",  # For local agent
#         "db_name": "index",                # For PubMed agent
#         "top_k": 5,                        # For both agents
#         "use_local_index": True            # Use local indexes only, don't download
#     }
    
#     try:
#         # Read the CSV file
#         logger.info(f"Reading CSV file: {csv_path}")
#         df = pd.read_csv(csv_path)
        
#         # Add new columns for the new responses
#         df["New Clinical Response"] = ""
#         df["New Clinical Sources"] = ""
        
#         # Process each question
#         for idx, row in df.iterrows():
#             question = row["Question"]
#             logger.info(f"Processing question {idx+1}/{len(df)}: {question}")
            
#             # Process the question
#             result = await process_question(question, orchestrator, aggregator, context)
            
#             # Update the DataFrame with the result
#             df.at[idx, "New Clinical Response"] = result["new_clinical_response"]
#             df.at[idx, "New Clinical Sources"] = result["new_clinical_sources"]
            
#             # Save progress after each question (in case of interruption)
#             df.to_csv(output_csv_path, index=False)
#             logger.info(f"Progress saved to {output_csv_path}")
            
#             # Add a small delay to prevent overwhelming the API
#             await asyncio.sleep(1)
        
#         # Final save
#         logger.info(f"All questions processed. Final results saved to {output_csv_path}")
#         df.to_csv(output_csv_path, index=False)
        
#         return df
#     except Exception as e:
#         logger.error(f"Critical error in processing CSV: {str(e)}", exc_info=True)
#         print(f"A critical error occurred: {str(e)}")
#         return None

# async def main():
#     """Main function to run the script."""
#     # Search for the CSV file with various possible name formats
#     possible_filenames = [
#         "ClinicalVsChatGPTComparison - Questions.csv",
#         "ClinicalVsChatGPTComparison Questions.csv", 
#         "ClinicalVsChatGPTComparison_Questions.csv",
#         "ClinicalVsChatGPTComparison-Questions.csv"
#     ]
    
#     input_csv = None
#     for filename in possible_filenames:
#         if os.path.exists(filename):
#             input_csv = filename
#             break
    
#     if input_csv is None:
#         logger.error("Could not find the CSV file. Please ensure it exists in the current directory.")
#         print("Could not find the CSV file. Please ensure it exists in the current directory.")
#         print("Tried the following filenames:")
#         for filename in possible_filenames:
#             print(f"- {filename}")
#         return
    
#     output_csv = "ClinicalVsChatGPTComparison_Updated.csv"
    
#     logger.info(f"Starting processing of {input_csv}")
    
#     result_df = await process_csv(input_csv, output_csv)
    
#     if result_df is not None:
#         logger.info(f"Processing completed successfully. Results saved to {output_csv}")
#         print(f"Processing completed successfully. Results saved to {output_csv}")
#     else:
#         logger.error("Processing failed")
#         print("Processing failed. Check logs for details.")

# if __name__ == "__main__":
#     asyncio.run(main())