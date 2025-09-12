"""
keyword_agents.utils
~~~~~~~~~~~~~~~~~~~

Utility functions for keyword processing agents.
"""
import logging
from typing import List, Dict, Any
from urllib.parse import quote_plus
from datetime import datetime, timezone

# Create module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def format_pubmed_search_url(terms: List[str], 
                             boolean_operator: str = "OR", 
                             date_floor: str = "2024/01/01", 
                             retmax: int = 1000) -> str:
    """
    Build a PubMed ESearch URL with proper boolean logic.
    
    Parameters
    ----------
    terms : List[str]
        List of search terms
    boolean_operator : str, default "OR"
        Boolean operator to join terms (OR, AND)
    date_floor : str, default "2024/01/01"
        Minimum publication date
    retmax : int, default 1000
        Maximum number of results to return
        
    Returns
    -------
    str
        Fully encoded PubMed ESearch URL
    """
    if not terms:
        raise ValueError("Term list is empty; cannot build PubMed query.")
        
    # Build the boolean-joined term string
    encoded_terms = [quote_plus(term) for term in terms]
    join_str = f"%20{boolean_operator}%20"
    term_clause = join_str.join(encoded_terms)
    term_clause = f"({term_clause})"
    
    # Append mandatory ESearch parameters
    today = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    params = (
        f"?db=pubmed"
        f"&term={term_clause}"
        f"&retmode=json"
        f"&datetype=pdat"
        f"&mindate={date_floor}"
        f"&maxdate={today}"
        f"&retmax={retmax}"
    )
    
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi{params}"
    logger.info("Generated PubMed URL: %s", url)
    return url


def batch_keywords(clusters: Dict[str, List[str]], max_keywords_per_batch: int = 50) -> List[List[str]]:
    """
    Batch keywords to prevent overly long URLs.
    
    Parameters
    ----------
    clusters : Dict[str, List[str]]
        Keyword clusters
    max_keywords_per_batch : int, default 50
        Maximum number of keywords per batch
        
    Returns
    -------
    List[List[str]]
        Batched keywords
    """
    batches = []
    current_batch = []
    count = 0
    
    # Process each cluster
    for cluster_name, keywords in clusters.items():
        for keyword in keywords:
            if count >= max_keywords_per_batch:
                batches.append(current_batch)
                current_batch = []
                count = 0
            
            current_batch.append(keyword)
            count += 1
    
    # Add the last batch if not empty
    if current_batch:
        batches.append(current_batch)
    
    logger.info(f"Created {len(batches)} batches from {len(clusters)} clusters")
    return batches