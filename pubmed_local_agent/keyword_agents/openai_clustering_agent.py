"""
keyword_agents.openai_clustering_agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keyword processing agent using OpenAI for semantic clustering.
"""
import logging
import os
from typing import List, Dict, Any, Tuple
import json

from openai import OpenAI
from dotenv import load_dotenv

from .base_agent import KeywordProcessingAgent
from .utils import format_pubmed_search_url, batch_keywords

# Set up logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


class OpenAIClusteringAgent(KeywordProcessingAgent):
    """
    Keyword processing agent that uses OpenAI to cluster medical keywords semantically.
    """
    
    def __init__(self, model="gpt-4o", max_batch_size=50):
        """
        Initialize the OpenAI clustering agent.
        
        Parameters
        ----------
        model : str, default "gpt-4o"
            OpenAI model to use for clustering
        max_batch_size : int, default 50
            Maximum number of keywords per PubMed URL batch
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not found. "
                "Set it in your environment or .env file before continuing."
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_batch_size = max_batch_size
        logger.info(f"Initialized OpenAIClusteringAgent with model {model}")
    
    def process_keywords(self, keywords: List[str], **kwargs) -> List[str]:
        """
        Process keywords from raw list to optimized PubMed URLs.
        
        Parameters
        ----------
        keywords : List[str]
            Raw list of keywords to process
        **kwargs : Dict[str, Any]
            Additional parameters:
            - max_clusters: int, maximum number of clusters to create
            - optimize_boolean: bool, whether to optimize boolean logic
            
        Returns
        -------
        List[str]
            List of PubMed search URLs
        """
        logger.info(f"Processing {len(keywords)} keywords")
        
        # Extract optional parameters
        max_clusters = kwargs.get('max_clusters', 10)
        optimize_boolean = kwargs.get('optimize_boolean', True)
        
        # Step 1: Cluster keywords
        clusters = self.cluster_keywords(keywords, max_clusters=max_clusters)
        
        # Step 2: Create optimized URL format using boolean logic
        urls = self.format_pubmed_url(clusters, optimize_boolean=optimize_boolean)
        
        logger.info(f"Generated {len(urls)} PubMed URLs from {len(keywords)} keywords")
        return urls
    
    def cluster_keywords(self, keywords: List[str], **kwargs) -> Dict[str, List[str]]:
        """
        Use OpenAI to cluster keywords semantically.
        
        Parameters
        ----------
        keywords : List[str]
            List of keywords to cluster
        **kwargs : Dict[str, Any]
            Additional parameters:
            - max_clusters: int, maximum number of clusters to create
            
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping cluster names to lists of keywords
        """
        max_clusters = kwargs.get('max_clusters', 10)
        
        if not keywords:
            logger.warning("Empty keyword list provided for clustering")
            return {}
        
        logger.info(f"Clustering {len(keywords)} keywords into max {max_clusters} clusters")
        
        try:
            # Format the prompt
            logger.info("Building prompt for OpenAI clustering")
            prompt_text = self._build_clustering_prompt(keywords, max_clusters)
            logger.info("Prompt built successfully")
            
            # Call OpenAI API
            logger.info("Calling OpenAI API for clustering")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a medical research expert specializing in organizing medical terminology into meaningful clusters for PubMed searches."
                        ),
                    },
                    {"role": "user", "content": prompt_text},
                ]
            )
            logger.info("OpenAI API call completed")

            # Log the response
            logger.info(f"OpenAI response: {response.choices[0].message.content}")

            
            # Parse the response
            logger.info("Parsing OpenAI response")
            # Strip Markdown-style code fencing like ```json ... ```
            import re
            raw_content = response.choices[0].message.content
            # Remove the code block markers and any leading/trailing whitespace
            cleaned_response_json_str = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip(), flags=re.IGNORECASE)
            logger.info(f"Cleaned response JSON string: {cleaned_response_json_str}")
            result = json.loads(cleaned_response_json_str)
            logger.info(f"Response parsed successfully: {result}")
            clusters = result.get("clusters", {})
            
            # Log the results
            cluster_sizes = {name: len(terms) for name, terms in clusters.items()}
            logger.info(f"Created {len(clusters)} clusters with sizes: {cluster_sizes}")
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error during keyword clustering: {str(e)}")
            # Fallback: return a single cluster with all keywords
            return {"general": keywords}
    
    def format_pubmed_url(self, keyword_clusters: Dict[str, List[str]], **kwargs) -> List[str]:
        """
        Format PubMed URLs based on keyword clusters.
        
        Parameters
        ----------
        keyword_clusters : Dict[str, List[str]]
            Dictionary of keyword clusters
        **kwargs : Dict[str, Any]
            Additional parameters:
            - optimize_boolean: bool, whether to optimize boolean logic
            
        Returns
        -------
        List[str]
            List of PubMed search URLs
        """
        optimize_boolean = kwargs.get('optimize_boolean', True)
        urls = []
        
        if not keyword_clusters:
            logger.warning("Empty cluster dictionary provided")
            return urls
        
        try:
            if optimize_boolean:
                # Complex strategy: optimize clusters with specific boolean logic
                urls = self._create_optimized_urls(keyword_clusters)
            else:
                # Simple strategy: batch keywords and use OR logic
                all_keywords = []
                for cluster_keywords in keyword_clusters.values():
                    all_keywords.extend(cluster_keywords)
                
                keyword_batches = batch_keywords({"all": all_keywords}, self.max_batch_size)
                urls = [format_pubmed_search_url(batch) for batch in keyword_batches]
            
            logger.info(f"Created {len(urls)} PubMed URLs")
            return urls
            
        except Exception as e:
            logger.error(f"Error formatting PubMed URLs: {str(e)}")
            # Fallback: flatten all keywords and create a single URL with OR logic
            all_keywords = []
            for cluster in keyword_clusters.values():
                all_keywords.extend(cluster)
            
            if all_keywords:
                url = format_pubmed_search_url(all_keywords)
                return [url]
            return []
    
    def _build_clustering_prompt(self, keywords: List[str], max_clusters: int) -> str:
        """
        Build the prompt for OpenAI to cluster keywords.
        
        Parameters
        ----------
        keywords : List[str]
            List of keywords to cluster
        max_clusters : int
            Maximum number of clusters to create
            
        Returns
        -------
        str
            Formatted prompt for OpenAI
        """
        keywords_str = "\n".join([f"- {kw}" for kw in keywords])
        
        # prompt = f"""
        # I have a list of medical keywords that I need to cluster semantically for PubMed searches.
        # Please group these keywords into at most {max_clusters} coherent clusters based on medical relevance.
        
        # Keywords:
        # {keywords_str}
        
        # Instructions:
        # 1. Group the keywords into meaningful clusters based on medical relationships
        # 2. Give each cluster a descriptive name that reflects the medical concept it represents
        # 3. Each keyword should appear in exactly one cluster
        # 4. Create at most {max_clusters} clusters, but fewer if that makes more sense semantically
        # 5. Return the results as a JSON object with the following format:
        
        # ```json
        # {{
        #     "clusters": {{
        #         "cluster_name_1": ["keyword1", "keyword2", ...],
        #         "cluster_name_2": ["keyword3", "keyword4", ...],
        #         ...
        #     }}
        # }}
        # ```
        
        # Provide ONLY the JSON response with no additional text.
        # """
        
        # Enhanced Medical Keyword Clustering Prompt for PubMed Searches

        
        prompt = f"""
        I have a list of medical keywords that I need to cluster semantically for optimal PubMed literature searches.
        Please group these keywords into at most {max_clusters} coherent clusters based on medical relevance and semantic relationships.

        Keywords:
        {keywords_str}

        Instructions:
        1. Group the keywords into meaningful clusters based on medical relationships. Consider the following hierarchical relationships:
        - Disease categories (e.g., cardiovascular, neurological, oncological, infectious)
        - Anatomical systems (e.g., respiratory, digestive, endocrine, musculoskeletal)
        - Treatment modalities (e.g., surgical, pharmacological, therapeutic, preventive)
        - Diagnostic approaches (e.g., imaging, laboratory tests, clinical assessments, biomarkers)
        - Biological mechanisms (e.g., inflammatory, genetic, metabolic, immunological)

        2. Give each cluster a descriptive name that reflects the medical concept it represents:
        - Use standardized medical terminology when appropriate (e.g., "Cardiovascular Disorders" rather than "Heart Problems")
        - Ensure names are specific enough to distinguish between related clusters
        - The name should represent the common theme connecting all keywords in the cluster
        - Consider using established terminology from medical taxonomies like MeSH when applicable

        3. Each keyword should appear in exactly one cluster:
        - Even if a keyword could potentially fit multiple categories, assign it to the most relevant cluster
        - Ensure complete coverage - every keyword must be assigned to a cluster
        - Do not remove, modify or add any keywords from the original list
        - For ambiguous terms, consider their most likely context in medical literature

        4. Create at most {max_clusters} clusters, but fewer if that makes more sense semantically:
        - Prioritize coherence and medical relevance over creating the maximum number of clusters
        - Avoid creating single-keyword clusters unless absolutely necessary
        - Balance cluster sizes when possible (avoid having one very large cluster and many small ones)
        - Merge smaller clusters if they represent closely related concepts
        - Consider the practical utility of clusters for PubMed search construction

        5. Return the results as a JSON object with the following format:
        ```json
        {{
            "clusters": {{
                "cluster_name_1": ["keyword1", "keyword2", ...],
                "cluster_name_2": ["keyword3", "keyword4", ...],
                ...
            }}
        }}
        ```

        6. Optimize for PubMed search relevance:
        - Consider how medical researchers would typically group these terms in literature searches
        - Group synonyms or closely related terms that would typically be combined with OR operators
        - Separate distinct medical concepts that would typically be combined with AND operators
        - Consider term specificity and sensitivity in medical literature indexing
        - Account for how MeSH terms are structured and related in the PubMed database

        7. Maintain scientific accuracy:
        - Respect established medical taxonomies and classification systems
        - Consider MeSH (Medical Subject Headings) categorization principles
        - Ensure clusters reflect clinically meaningful distinctions
        - Group terms according to standard practice in medical research literature
        - Account for both clinical and research-oriented perspectives

        8. Handle edge cases appropriately:
        - For very broad terms, place them in the most specific relevant cluster
        - For highly specific terms, group them with their broader category when appropriate
        - For acronyms or abbreviations, interpret them according to medical convention
        - For terms with multiple potential meanings, choose the most likely medical interpretation
        
        
        VERY VERY IMPORTANT - Return the results as a JSON object with the following format:
        ```json
        {{
            "clusters": {{
                "cluster_name_1": ["keyword1", "keyword2", ...],
                "cluster_name_2": ["keyword3", "keyword4", ...],
                ...
            }}
        }}
        ```

        Provide ONLY the JSON response with no additional text. Ensure the JSON is properly formatted and valid without any explanations, comments, or wrapper text.
        """
        
        
        return prompt
    
    def _create_optimized_urls(self, clusters: Dict[str, List[str]]) -> List[str]:
        """
        Create optimized PubMed URLs using sophisticated boolean logic.
        
        Parameters
        ----------
        clusters : Dict[str, List[str]]
            Dictionary of keyword clusters
            
        Returns
        -------
        List[str]
            List of optimized PubMed URLs
        """
        urls = []
        cluster_names = list(clusters.keys())
        
        # If we have multiple clusters, we can create complex boolean queries
        if len(cluster_names) > 1:
            # Approach 1: One representative term from each cluster joined by AND
            try:
                representative_terms = [clusters[name][0] for name in cluster_names if clusters[name]]
                if representative_terms:
                    urls.append(format_pubmed_search_url(representative_terms, boolean_operator="AND"))
            except Exception as e:
                logger.error(f"Error creating representative terms URL: {str(e)}")
        
        # Process each cluster individually
        for cluster_name, keywords in clusters.items():
            if not keywords:
                continue
                
            # Batch keywords from this cluster to avoid overly long URLs
            batches = [keywords[i:i+self.max_batch_size] for i in range(0, len(keywords), self.max_batch_size)]
            
            for batch in batches:
                urls.append(format_pubmed_search_url(batch, boolean_operator="OR"))
        
        return urls
