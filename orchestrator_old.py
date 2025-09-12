from typing import List, Dict, Any
import openai
from agent_base import AgentBase
from local_agent_wrapper import LocalAgent

class Orchestrator:
    def __init__(self):
        self.agents: Dict[str, AgentBase] = {
            "local_medical": LocalAgent()
        }
    
    def _analyze_query(self, query: str) -> List[str]:
        """Determine which agents should handle the query"""
        try:
            # Use OpenAI to analyze query and select agents
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing questions and determining which specialized agent should handle them. Available agents: local_medical (Medical research papers and clinical data)"},
                    {"role": "user", "content": f"Analyze this query and return only the agent name that should handle it: {query}"}
                ]
            )
            agent = response.choices[0].message.content.strip().lower()
            return [agent] if agent in self.agents else ["local_medical"]
        except:
            return ["local_medical"]
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        context = context or {}
        results = []
        
        selected_agents = self._analyze_query(query)
        
        for agent_name in selected_agents:
            agent = self.agents.get(agent_name)
            if agent:
                result = agent.query(query, context)
                results.append({
                    "agent": agent_name,
                    "response": result
                })
                
        # print(f"Resules from agents: {results}")
                
        return results