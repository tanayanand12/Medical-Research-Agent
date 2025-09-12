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

class Orchestrator:
    """
    Orchestrator for running multiple research agents in parallel with fallback capabilities.
    
    This class manages the execution of different agent types, handling:
    - Parallel execution of agent queries
    - Collection and formatting of responses
    - Error handling across agents
    - Fallback mechanism for incoherent responses
    """
    
    def __init__(self):
        """Initialize the orchestrator with available agents and fallback mechanism."""
        self.agents = {
            "local": LocalAgent(),
            "pubmed": PubMedAgent()
        }
        self.fallback = FallbackMechanism()
        logger.info(f"Orchestrator initialized with {len(self.agents)} agents and fallback mechanism")
    
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
        Process a query across all registered agents in parallel with fallback support.
        
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
        # Create context dictionaries for each agent type if needed
        context = context or {}
        local_context = {
            "model_id": context.get("model_id", "randy-data-testing"),
            "top_k": context.get("top_k", 5)
        }
        
        pubmed_context = {
            "db_name": context.get("db_name", "index"),
            "top_k": context.get("top_k", 8)
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
                ): "pubmed"
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
            "query": query
        }

    async def process_query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query across all registered agents asynchronously with fallback support.
        
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
        # Create context dictionaries for each agent type if needed
        context = context or {}
        local_context = {
            "model_id": context.get("model_id", "randy-data-testing"),
            "top_k": context.get("top_k", 5)
        }
        
        pubmed_context = {
            "db_name": context.get("db_name", "index"),
            "top_k": context.get("top_k", 8)
        }
        
        # Create tasks for each agent
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                None, 
                self._execute_agent_query,
                "local",
                self.agents["local"],
                query,
                local_context
            ),
            loop.run_in_executor(
                None,
                self._execute_agent_query,
                "pubmed",
                self.agents["pubmed"],
                query,
                pubmed_context
            )
        ]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results, handling any exceptions
        processed_results = []
        for i, (agent_name, result) in enumerate(zip(["local", "pubmed"], results)):
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
        logger.info("Evaluating need for fallback mechanism")
        fallback_result = self.fallback.process(query, processed_results)
        
        return {
            "agent_responses": processed_results,
            "fallback_result": fallback_result,
            "query": query
        }