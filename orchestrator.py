# """
# Orchestrator with the query classifier and fallback mechanism
# This orchestrator manages the execution of multiple agents, handles query classification,
# and implements a fallback mechanism for incoherent responses.
# """

# import asyncio
# import concurrent.futures
# import os
# from typing import Dict, Any, List, Optional
# import logging
# from pathlib import Path

# from agent_base import AgentBase
# from local_agent_wrapper import LocalAgent
# from pubmed_local_agent_wrapper import PubMedAgent
# from fallback import FallbackMechanism
# from query_classifier import QueryClassifier  # Import the new QueryClassifier

# # Ensure logs directory exists
# Path("logs").mkdir(exist_ok=True)

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("logs/orchestrator.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# class Orchestrator:
#     """
#     Orchestrator for running multiple research agents in parallel with fallback capabilities.
    
#     This class manages the execution of different agent types, handling:
#     - Query classification to filter non-medical research queries
#     - Parallel execution of agent queries for medical research questions
#     - Collection and formatting of responses
#     - Error handling across agents
#     - Fallback mechanism for incoherent responses
#     """
    
#     def __init__(self):
#         """Initialize the orchestrator with available agents, fallback mechanism, and query classifier."""
#         self.agents = {
#             "local": LocalAgent(),
#             "pubmed": PubMedAgent()
#         }
#         self.fallback = FallbackMechanism()
#         self.query_classifier = QueryClassifier()  # Initialize the query classifier
#         logger.info(f"Orchestrator initialized with {len(self.agents)} agents, fallback mechanism, and query classifier")
    
#     def register_agent(self, name: str, agent: AgentBase) -> None:
#         """
#         Register a new agent with the orchestrator.
        
#         Parameters
#         ----------
#         name : str
#             Unique identifier for the agent
#         agent : AgentBase
#             Agent instance implementing the AgentBase interface
#         """
#         self.agents[name] = agent
#         logger.info(f"Registered new agent: {name}")
    
#     def _execute_agent_query(self, agent_name: str, agent: AgentBase, 
#                            query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
#         """
#         Execute a query on a single agent with error handling.
        
#         Parameters
#         ----------
#         agent_name : str
#             Name of the agent
#         agent : AgentBase
#             Agent instance
#         query : str
#             User query
#         context : Dict[str, Any], optional
#             Additional context parameters
            
#         Returns
#         -------
#         Dict[str, Any]
#             Result dictionary containing agent name and response
#         """
#         try:
#             logger.info(f"Executing query on agent: {agent_name}")
#             response = agent.query(query, context)
#             logger.info(f"Query completed on agent: {agent_name}")
#             return {
#                 "agent": agent_name,
#                 "response": response,
#                 "status": "success"
#             }
#         except Exception as e:
#             logger.error(f"Error in agent {agent_name}: {str(e)}", exc_info=True)
#             return {
#                 "agent": agent_name,
#                 "response": {
#                     "answer": f"Error in {agent_name} agent: {str(e)}",
#                     "citations": [],
#                     "confidence": 0.0
#                 },
#                 "status": "error"
#             }
    
#     def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#         """
#         Process a query with domain filtering and, if appropriate, route through agents with fallback support.
        
#         Parameters
#         ----------
#         query : str
#             User query to process
#         context : Dict[str, Any], optional
#             Additional context parameters for agents
            
#         Returns
#         -------
#         Dict[str, Any]
#             Dictionary containing responses and processing details
#         """
#         # First, check if the query is related to medical research
#         is_medical, classification = self.query_classifier.is_medical_research_query(query)
        
#         # If not medical research, return generic response without invoking agents
#         if not is_medical:
#             logger.info(f"Non-medical research query detected, bypassing agent pipeline: {query}")
#             generic_response = self.query_classifier.get_non_medical_response(query, classification)
            
#             # Format response to match the expected output structure
#             return {
#                 "agent_responses": [],
#                 "fallback_result": {
#                     "answer": generic_response["answer"],
#                     "citations": [],
#                     "fallback_activated": True,
#                     "fallback_reason": "Non-medical research query"
#                 },
#                 "query": query,
#                 "query_classification": classification,
#                 "domain_filtered": True
#             }
        
#         # Continue with normal processing for medical research queries
#         logger.info(f"Medical research query confirmed, proceeding with agent pipeline: {query}")
        
#         # Create context dictionaries for each agent type if needed
#         context = context or {}
#         local_context = {
#             "model_id": context.get("model_id", "randy-data-testing"),
#             "top_k": context.get("top_k", 5)
#         }
        
#         pubmed_context = {
#             "db_name": context.get("db_name", "index"),
#             "top_k": context.get("top_k", 8)
#         }
        
#         # Use ThreadPoolExecutor for parallel execution
#         results = []
#         with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
#             future_to_agent = {
#                 executor.submit(
#                     self._execute_agent_query, 
#                     "local", 
#                     self.agents["local"], 
#                     query, 
#                     local_context
#                 ): "local",
#                 executor.submit(
#                     self._execute_agent_query,
#                     "pubmed",
#                     self.agents["pubmed"],
#                     query,
#                     pubmed_context
#                 ): "pubmed"
#             }
            
#             for future in concurrent.futures.as_completed(future_to_agent):
#                 agent_name = future_to_agent[future]
#                 try:
#                     result = future.result()
#                     results.append(result)
#                     logger.info(f"Collected result from {agent_name}")
#                 except Exception as e:
#                     logger.error(f"Exception while processing {agent_name}: {str(e)}", exc_info=True)
#                     results.append({
#                         "agent": agent_name,
#                         "response": {
#                             "answer": f"Critical error in {agent_name} agent: {str(e)}",
#                             "citations": [],
#                             "confidence": 0.0
#                         },
#                         "status": "error"
#                     })
        
#         # Check if fallback mechanism is needed
#         logger.info("Evaluating need for fallback mechanism")
#         fallback_result = self.fallback.process(query, results)
        
#         return {
#             "agent_responses": results,
#             "fallback_result": fallback_result,
#             "query": query,
#             "query_classification": classification,
#             "domain_filtered": False
#         }

#     async def process_query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#         """
#         Process a query across all registered agents asynchronously with domain filtering and fallback support.
        
#         Parameters
#         ----------
#         query : str
#             User query to process
#         context : Dict[str, Any], optional
#             Additional context parameters for agents
            
#         Returns
#         -------
#         Dict[str, Any]
#             Dictionary containing responses from each agent and fallback status
#         """
#         # First, check if the query is related to medical research
#         is_medical, classification = self.query_classifier.is_medical_research_query(query)
        
#         # If not medical research, return generic response without invoking agents
#         if not is_medical:
#             logger.info(f"Non-medical research query detected, bypassing agent pipeline: {query}")
#             generic_response = self.query_classifier.get_non_medical_response(query, classification)
            
#             # Format response to match the expected output structure
#             return {
#                 "agent_responses": [],
#                 "fallback_result": {
#                     "answer": generic_response["answer"],
#                     "citations": [],
#                     "fallback_activated": True,
#                     "fallback_reason": "Non-medical research query"
#                 },
#                 "query": query,
#                 "query_classification": classification,
#                 "domain_filtered": True
#             }
        
#         # Continue with normal async processing for medical research queries
#         logger.info(f"Medical research query confirmed, proceeding with agent pipeline: {query}")
        
#         # Create context dictionaries for each agent type if needed
#         context = context or {}
#         local_context = {
#             "model_id": context.get("model_id", "randy-data-testing"),
#             "top_k": context.get("top_k", 5)
#         }
        
#         pubmed_context = {
#             "db_name": context.get("db_name", "index"),
#             "top_k": context.get("top_k", 8)
#         }
        
#         # Create tasks for each agent
#         loop = asyncio.get_running_loop()
#         tasks = [
#             loop.run_in_executor(
#                 None, 
#                 self._execute_agent_query,
#                 "local",
#                 self.agents["local"],
#                 query,
#                 local_context
#             ),
#             loop.run_in_executor(
#                 None,
#                 self._execute_agent_query,
#                 "pubmed",
#                 self.agents["pubmed"],
#                 query,
#                 pubmed_context
#             )
#         ]
        
#         # Wait for all tasks to complete
#         results = await asyncio.gather(*tasks, return_exceptions=True)
        
#         # Process results, handling any exceptions
#         processed_results = []
#         for i, (agent_name, result) in enumerate(zip(["local", "pubmed"], results)):
#             if isinstance(result, Exception):
#                 logger.error(f"Exception in {agent_name}: {str(result)}", exc_info=True)
#                 processed_results.append({
#                     "agent": agent_name,
#                     "response": {
#                         "answer": f"Critical error in {agent_name} agent: {str(result)}",
#                         "citations": [],
#                         "confidence": 0.0
#                     },
#                     "status": "error"
#                 })
#             else:
#                 processed_results.append(result)
        
#         # Check if fallback mechanism is needed
#         logger.info("Evaluating need for fallback mechanism")
#         fallback_result = self.fallback.process(query, processed_results)
        
#         return {
#             "agent_responses": processed_results,
#             "fallback_result": fallback_result,
#             "query": query,
#             "query_classification": classification,
#             "domain_filtered": False
#         }


#################################################################################

import asyncio
import concurrent.futures
import os
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from agent_base import AgentBase
from local_agent_wrapper import LocalAgent
from pubmed_local_agent_wrapper import PubMedAgent
    
from fallback import FallbackMechanism
from query_classifier import QueryClassifier

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/orchestrator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


try :
    from clinical_trials_agent_wrapper import ClinicalTrialsAgent  # New import
    logger.info("ClinicalTrialsAgent successfully imported")
except ImportError:
    logger.error("ClinicalTrialsAgent import failed, ensure it is implemented and available in the path", exc_info=True)
    # logger.warning("ClinicalTrialsAgent not found, ensure it is implemented and available in the path")
    ClinicalTrialsAgent = None


try:
    from fda_agent_wrapper import FDAAgent  # New import
    logger.info("FDAAgent successfully imported")
except ImportError:
    logger.error("FDAAgent import failed, ensure it is implemented and available in the path", exc_info=True)
    # logger.warning("FDAAgent not found, ensure it is implemented and available in the path")
    FDAAgent = None

class Orchestrator:
    """
    Orchestrator for running multiple research agents in parallel with fallback capabilities.
    
    This class manages the execution of different agent types, handling:
    - Query classification to filter non-medical research queries
    - Parallel execution of agent queries for medical research questions
    - Collection and formatting of responses
    - Error handling across agents
    - Fallback mechanism for incoherent responses
    """
    
    def __init__(self):
        """Initialize the orchestrator with available agents, fallback mechanism, and query classifier."""
        self.agents = {
            "local": LocalAgent(),
            "pubmed": PubMedAgent(),
            "clinical_trials": ClinicalTrialsAgent(),  # Add the new agent
            "fda": FDAAgent()  # Add the new agent
        }
        self.fallback = FallbackMechanism()
        self.query_classifier = QueryClassifier()
        logger.info(f"Orchestrator initialized with {len(self.agents)} agents, fallback mechanism, and query classifier")
    
    def register_agent(self, name: str, agent: AgentBase) -> None:
        """
        Register a new agent with the orchestrator.
        
        Parameters
        ----------
        name : str
            Unique identifier for the agent
        agent : AgentBase
            Agent instance implementing the AgentBase interface
        """
        self.agents[name] = agent
        logger.info(f"Registered new agent: {name}")
    
    def _execute_agent_query(self, agent_name: str, agent: AgentBase, 
                           query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a query on a single agent with error handling.
        
        Parameters
        ----------
        agent_name : str
            Name of the agent
        agent : AgentBase
            Agent instance
        query : str
            User query
        context : Dict[str, Any], optional
            Additional context parameters
            
        Returns
        -------
        Dict[str, Any]
            Result dictionary containing agent name and response
        """
        try:
            logger.info(f"Executing query on agent: {agent_name}")
            response = agent.query(query, context)
            logger.info(f"Query completed on agent: {agent_name}")
            return {
                "agent": agent_name,
                "response": response,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error in agent {agent_name}: {str(e)}", exc_info=True)
            return {
                "agent": agent_name,
                "response": {
                    "answer": f"Error in {agent_name} agent: {str(e)}",
                    "citations": [],
                    "confidence": 0.0
                },
                "status": "error"
            }
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query with domain filtering and, if appropriate, route through agents with fallback support.
        
        Parameters
        ----------
        query : str
            User query to process
        context : Dict[str, Any], optional
            Additional context parameters for agents
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing responses and processing details
        """
        # First, check if the query is related to medical research
        is_medical, classification = self.query_classifier.is_medical_research_query(query)
        
        # If not medical research, return generic response without invoking agents
        if not is_medical:
            logger.info(f"Non-medical research query detected, bypassing agent pipeline: {query}")
            generic_response = self.query_classifier.get_non_medical_response(query, classification)
            
            return {
                "agent_responses": [],
                "fallback_result": {
                    "answer": generic_response["answer"],
                    "citations": [],
                    "fallback_activated": True,
                    "fallback_reason": "Non-medical research query"
                },
                "query": query,
                "query_classification": classification,
                "domain_filtered": True
            }
        
        # Continue with normal processing for medical research queries
        logger.info(f"Medical research query confirmed, proceeding with agent pipeline: {query}")
        
        # Create context dictionaries for each agent type
        context = context or {}
        
        # Context for local agent
        local_context = {
            "model_id": context.get("model_id", "randy-data-testing"),
            "top_k": context.get("top_k", 5)
        }
        
        # Context for PubMed agent
        pubmed_context = {
            "db_name": context.get("db_name", "index"),
            "top_k": context.get("top_k", 8)
        }
        
        # Context for Clinical Trials agent
        clinical_trials_context = {
            "top_k": context.get("clinical_trials_top_k", context.get("top_k", 10)),
            "max_trials": context.get("max_trials", 25)
        }

        # Context for FDA agent
        fda_context = {
            "top_k": context.get("fda_top_k", context.get("top_k", 5)),
            "max_records": context.get("max_records", 300),
            "chunk_size": context.get("chunk_size", 10000),
            "chunk_overlap": context.get("chunk_overlap", 400)
        }
        
        # Use ThreadPoolExecutor for parallel execution
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            future_to_agent = {
                executor.submit(
                    self._execute_agent_query, 
                    "local", 
                    self.agents["local"], 
                    query, 
                    local_context
                ): "local",
                executor.submit(
                    self._execute_agent_query,
                    "pubmed",
                    self.agents["pubmed"],
                    query,
                    pubmed_context
                ): "pubmed",
                executor.submit(
                    self._execute_agent_query,
                    "clinical_trials",
                    self.agents["clinical_trials"],
                    query,
                    clinical_trials_context
                ): "clinical_trials",
                executor.submit(
                    self._execute_agent_query,
                    "fda",
                    self.agents["fda"],
                    query,
                    fda_context
                ): "fda"
            }
            
            for future in concurrent.futures.as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Collected result from {agent_name}")
                except Exception as e:
                    logger.error(f"Exception while processing {agent_name}: {str(e)}", exc_info=True)
                    results.append({
                        "agent": agent_name,
                        "response": {
                            "answer": f"Critical error in {agent_name} agent: {str(e)}",
                            "citations": [],
                            "confidence": 0.0
                        },
                        "status": "error"
                    })
        
        # Check if fallback mechanism is needed
        logger.info("Evaluating need for fallback mechanism")
        fallback_result = self.fallback.process(query, results)
        
        return {
            "agent_responses": results,
            "fallback_result": fallback_result,
            "query": query,
            "query_classification": classification,
            "domain_filtered": False
        }

    async def process_query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query across all registered agents asynchronously with domain filtering and fallback support.
        
        Parameters
        ----------
        query : str
            User query to process
        context : Dict[str, Any], optional
            Additional context parameters for agents
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing responses from each agent and fallback status
        """
        # First, check if the query is related to medical research
        is_medical, classification = self.query_classifier.is_medical_research_query(query)
        
        # If not medical research, return generic response without invoking agents
        if not is_medical:
            logger.info(f"Non-medical research query detected, bypassing agent pipeline: {query}")
            generic_response = self.query_classifier.get_non_medical_response(query, classification)
            
            return {
                "agent_responses": [],
                "fallback_result": {
                    "answer": generic_response["answer"],
                    "citations": [],
                    "fallback_activated": True,
                    "fallback_reason": "Non-medical research query"
                },
                "query": query,
                "query_classification": classification,
                "domain_filtered": True
            }
        
        # Continue with normal async processing for medical research queries
        logger.info(f"Medical research query confirmed, proceeding with agent pipeline: {query}")
        
        # Create context dictionaries for each agent type
        context = context or {}
        
        local_context = {
            "model_id": context.get("model_id", "randy-data-testing"),
            "top_k": context.get("top_k", 5)
        }
        
        pubmed_context = {
            "db_name": context.get("db_name", "index"),
            "top_k": context.get("top_k", 8)
        }
        
        clinical_trials_context = {
            "top_k": context.get("clinical_trials_top_k", context.get("top_k", 10)),
            "max_trials": context.get("max_trials", 25)
        }

        fda_context = {
            "top_k": context.get("fda_top_k", context.get("top_k", 5)),
            "max_records": context.get("max_records", 300),
            "chunk_size": context.get("chunk_size", 10000),
            "chunk_overlap": context.get("chunk_overlap", 400)
        }

        
        
        # Create tasks for each agent
        loop = asyncio.get_running_loop()
        enabled_agents = [
            name for name, enabled in context.get("enabled_agents", {}).items()
            if enabled
        ]
        agent_contexts = {
            "local": (self.agents["local"], local_context),
            "pubmed": (self.agents["pubmed"], pubmed_context),
            "clinical_trials": (self.agents["clinical_trials"], clinical_trials_context),
            "fda": (self.agents["fda"], fda_context),
        }

        tasks = []
        agent_names = []

        for name in enabled_agents:
            if name in self.agents:
                agent, agent_ctx = agent_contexts[name]
                tasks.append(loop.run_in_executor(None, self._execute_agent_query, name, agent, query, agent_ctx))
                agent_names.append(name)

        logger.info(f"Running queries with agents: {agent_names}")

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results, handling any exceptions
        processed_results = []
        
        for i, (agent_name, result) in enumerate(zip(agent_names, results)):
            if isinstance(result, Exception):
                logger.error(f"Exception in {agent_name}: {str(result)}", exc_info=True)
                processed_results.append({
                    "agent": agent_name,
                    "response": {
                        "answer": f"Critical error in {agent_name} agent: {str(result)}",
                        "citations": [],
                        "confidence": 0.0
                    },
                    "status": "error"
                })
            else:
                processed_results.append(result)
   
        # Check if fallback mechanism is needed
        release = context.get("release", True)
        if release:
            logger.info("Evaluating need for fallback mechanism (release=True)")
            fallback_result = self.fallback.process(query, processed_results)
        else:
            logger.info("Skipping fallback mechanism (release=False)")
            fallback_result = {
                "answer": "",
                "citations": [],
                "fallback_activated": False,
                "fallback_reason": "Fallback disabled in non-release mode"
            }
        
        return {
            "agent_responses": processed_results,
            "fallback_result": fallback_result,
            "query": query,
            "query_classification": classification,
            "domain_filtered": False
        }