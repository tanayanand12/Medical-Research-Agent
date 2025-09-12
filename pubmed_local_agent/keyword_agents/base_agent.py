"""
keyword_agents.base_agent
~~~~~~~~~~~~~~~~~~~~~~~~~

Base class for keyword processing agents.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class KeywordProcessingAgent(ABC):
    """
    Abstract base class for all keyword processing agents.
    
    Defines the required interface for keyword processing agents.
    """
    
    @abstractmethod
    def process_keywords(self, keywords: List[str], **kwargs) -> List[str]:
        """
        Process a list of keywords and return processed PubMed URLs.
        
        Parameters
        ----------
        keywords : List[str]
            Raw list of keywords to process
        **kwargs : Dict[str, Any]
            Additional parameters for specific implementations
            
        Returns
        -------
        List[str]
            List of PubMed search URLs
        """
        pass
    
    @abstractmethod
    def cluster_keywords(self, keywords: List[str], **kwargs) -> Dict[str, List[str]]:
        """
        Cluster keywords into semantically related groups.
        
        Parameters
        ----------
        keywords : List[str]
            Raw list of keywords to cluster
        **kwargs : Dict[str, Any]
            Additional parameters for specific implementations
            
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping cluster names to lists of keywords
        """
        pass
    
    @abstractmethod
    def format_pubmed_url(self, keyword_clusters: Dict[str, List[str]], **kwargs) -> List[str]:
        """
        Format PubMed URL(s) based on keyword clusters.
        
        Parameters
        ----------
        keyword_clusters : Dict[str, List[str]]
            Dictionary of keyword clusters
        **kwargs : Dict[str, Any]
            Additional parameters for specific implementations
            
        Returns
        -------
        List[str]
            List of PubMed search URLs
        """
        pass