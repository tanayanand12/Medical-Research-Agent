#!/usr/bin/env python
"""
process.py
~~~~~~~~~~

End‑to‑end ETL pipeline:
    1. Build PubMed ESearch URL(s) from keywords - optionally using clustering agents
    2. Retrieve recent papers (Jan 2024 → today)
    3. Chunk + embed with Vectorizer
    4. Upsert into FAISS and persist on disk

Run `python process.py --help` for CLI options.
"""

from __future__ import annotations

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Union

from .core.medical_search_agent import MedicalSearchAgent
from .core.pubmed_retriever import PubMedRetriever
from .core.vectorizer import Vectorizer, EMBED_DIM
from .core.faiss_db_manager import FaissVectorDB

# Import keyword agents
try:
    from keyword_agents import OpenAIClusteringAgent
    KEYWORD_AGENTS_AVAILABLE = True
except ImportError:
    KEYWORD_AGENTS_AVAILABLE = False

# ────────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# ────────────────────────────────────────────────────────────────────────────────
# Main ETL function
# ────────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    keywords: List[str],
    db_name: str,
    max_papers: int = 2500,
    include_fulltext: bool = False,
    use_keyword_agent: bool = False,
    agent_params: Dict[str, Any] = None,
) -> None:
    """
    Execute the full pipeline and write FAISS index to disk.
    
    Parameters
    ----------
    keywords : List[str]
        List of keywords for PubMed search
    db_name : str
        Base name for the FAISS index
    max_papers : int, default 2500
        Maximum number of papers to retrieve
    include_fulltext : bool, default False
        Whether to include full text when available
    use_keyword_agent : bool, default False
        Whether to use keyword processing agent
    agent_params : Dict[str, Any], optional
        Parameters for keyword agent
    """
    # Initialize corpus
    corpus = {}

    logger.info("Starting pipeline...")
    logger.info("Keywords: %s", keywords)
    
    # 1. Generate PubMed URL(s) - with or without keyword agent
    if use_keyword_agent and KEYWORD_AGENTS_AVAILABLE:
        logger.info("Using keyword processing agent for query generation")
        try:
            # Initialize agent with parameters
            logger.info("Initializing keyword agent with parameters: %s", agent_params)
            agent_params = agent_params or {}
            agent = OpenAIClusteringAgent(**agent_params.get("init", {}))
            logger.info("Keyword agent initialized successfully")
            
            # Process keywords and get URLs
            logger.info("Processing keywords with agent")
            urls = agent.process_keywords(keywords, **agent_params.get("process", {}))
            logger.info(f"Generated {len(urls)} PubMed URLs using keyword agent")

            logger.info("URLs: %s", urls)
            
            # Process each URL
            logger.info("Initializing PubMed retriever")
            retriever = PubMedRetriever()
            logger.info("Using PubMed API with %d seconds delay", retriever.delay)
            for i, url in enumerate(urls):
                logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")
                logger.info("Creating corpus from URL")
                batch_corpus = retriever.create_corpus(
                    url, max_papers=max_papers, include_fulltext=include_fulltext
                )
                # dump batch_corpus to a file
                with open(f"pubmed_corpus_{i+1}.json", "w") as f:
                    json.dump(batch_corpus, f, indent=4)
                logger.info(f"Retrieved {len(batch_corpus)} papers from URL {i+1}")

                corpus.extend(batch_corpus)  # Merge batch corpus into main corpus

                # dump corpus to a file
                with open("pubmed_corpus.json", "w") as f:
                    json.dump(corpus, f, indent=4)
            
            logger.info("Total papers retrieved: %d", len(corpus))
            logger.info("Completed corpus creation using keyword agent")

            # dump corpus to a file
            with open("new_pubmed_corpus.json", "w") as f:
                json.dump(corpus, f, indent=4)
                
        except Exception as e:
            logger.error(f"Error using keyword agent: {str(e)}")
            logger.info("Falling back to standard keyword processing...")

            # Fallback to standard processing
            search_agent = MedicalSearchAgent()
            url = search_agent.format_pubmed_search_url(keywords)
            logger.info("PubMed URL: %s", url)

            retriever = PubMedRetriever()
            logger.info("Using PubMed API with %d seconds delay", retriever.delay)
            corpus = retriever.create_corpus(
                url, max_papers=max_papers, include_fulltext=include_fulltext
            )
    else:
        # Standard keyword processing (OR-based)
        search_agent = MedicalSearchAgent()
        url = search_agent.format_pubmed_search_url(keywords)
        logger.info("PubMed URL: %s", url)
        
        retriever = PubMedRetriever()
        logger.info("Using PubMed API with %d seconds delay", retriever.delay)
        corpus = retriever.create_corpus(
            url, max_papers=max_papers, include_fulltext=include_fulltext
        )
    
    logger.info("Total papers retrieved: %d", len(corpus))
    logger.info("Completed corpus creation.")
    
    # dump corpus to a file
    with open("pubmed_corpus.json", "w") as f:
        json.dump(corpus, f, indent=4)

    if not corpus:
        logger.error("No papers retrieved – aborting.")
        return

    # 3. Vectorise
    logger.info("Vectorising papers...")
    vectoriser = Vectorizer()
    docs = vectoriser.embed_corpus(corpus)
    logger.info("Total embedded chunks: %d", len(docs))

    # 4. Upsert into FAISS
    db = FaissVectorDB(dimension=EMBED_DIM)
    db.add_documents(docs)
    db.save(db_name)
    logger.info("Pipeline finished – FAISS DB saved as '%s'", db_name)


# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Build / refresh a PubMed FAISS DB")
    parser.add_argument(
        "--db_name",
        "-d",
        default="index",
        help="Base name for index under pubmed_faiss_index/",
    )
    parser.add_argument(
        "--max_papers",
        "-p",
        type=int,
        default=2500,
        help="Maximum number of PubMed papers to retrieve",
    )
    parser.add_argument(
        "--fulltext",
        action="store_true",
        default=True,
        help="Also fetch PMC full text when available (slower)",
    )
    parser.add_argument(
        "--use_keyword_agent",
        action="store_true",
        help="Use keyword processing agent for PubMed query generation",
    )
    parser.add_argument(
        "--agent_config",
        default=None,
        help="Path to JSON config file for keyword agent parameters",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    Path("pubmed_faiss_index").mkdir(exist_ok=True)

    import pandas as pd
    df = pd.read_csv("pubmeds_keywords.csv")
    lst_keywords = df["Pubmed keywords"].tolist()
    
    # Parse agent parameters if provided
    agent_params = None
    if args.agent_config:
        try:
            with open(args.agent_config, 'r') as f:
                agent_params = json.load(f)
            logger.info(f"Loaded agent config from {args.agent_config}")
        except Exception as e:
            logger.error(f"Error loading agent config: {str(e)}")

    run_pipeline(
        keywords=lst_keywords,
        db_name=args.db_name,
        max_papers=args.max_papers,
        include_fulltext=args.fulltext,
        use_keyword_agent=args.use_keyword_agent,
        agent_params=agent_params,
    )


if __name__ == "__main__":
    main()




##############################

# #!/usr/bin/env python
# """
# process.py
# ~~~~~~~~~~

# End‑to‑end ETL pipeline:
#     1. Build PubMed ESearch URL from a *flat* keyword list
#     2. Retrieve recent papers (Jan 2024 → today)
#     3. Chunk + embed with Vectorizer
#     4. Upsert into FAISS and persist on disk

# Run `python process.py --help` for CLI options.
# """

# from __future__ import annotations

# import argparse
# import logging
# from pathlib import Path
# from typing import List

# from core.medical_search_agent import MedicalSearchAgent
# from core.pubmed_retriever import PubMedRetriever
# from core.vectorizer import Vectorizer, EMBED_DIM
# from core.faiss_db_manager import FaissVectorDB

# # ────────────────────────────────────────────────────────────────────────────────
# # Logging
# # ────────────────────────────────────────────────────────────────────────────────
# logger = logging.getLogger(__name__)
# if not logger.handlers:
#     h = logging.StreamHandler()
#     h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
#     logger.addHandler(h)
# logger.setLevel(logging.INFO)


# # ────────────────────────────────────────────────────────────────────────────────
# # Main ETL function
# # ────────────────────────────────────────────────────────────────────────────────
# def run_pipeline(
#     keywords: List[str],
#     db_name: str,
#     max_papers: int = 2500,
#     include_fulltext: bool = False,
# ) -> None:
#     """Execute the full pipeline and write FAISS index to disk."""
#     # 1. Generate PubMed URL
#     search_agent = MedicalSearchAgent()
#     url = search_agent.format_pubmed_search_url(keywords)
#     logger.info("PubMed URL: %s", url)

#     # 2. Retrieve papers
#     logger.info("Retrieving papers...")
#     retriever = PubMedRetriever()
#     logger.info("Using PubMed API with %d seconds delay", retriever.delay)
#     corpus = retriever.create_corpus(
#         url, max_papers=max_papers, include_fulltext=include_fulltext
#     )
#     logger.info("Total papers retrieved: %d", len(corpus))
#     logger.info("Completed corpus creation.")
    
#     # dump corpus to a file
#     with open("pubmed_corpus.json", "w") as f:
#         import json
#         json.dump(corpus, f, indent=4)

#     if not corpus:
#         logger.error("No papers retrieved – aborting.")
#         return

#     # 3. Vectorise
#     logger.info("Vectorising papers...")
#     vectoriser = Vectorizer()
#     docs = vectoriser.embed_corpus(corpus)
#     logger.info("Total embedded chunks: %d", len(docs))

#     # 4. Upsert into FAISS
#     db = FaissVectorDB(dimension=EMBED_DIM)
#     db.add_documents(docs)
#     db.save(db_name)
#     logger.info("Pipeline finished – FAISS DB saved as '%s'", db_name)


# # ────────────────────────────────────────────────────────────────────────────────
# # CLI
# # ────────────────────────────────────────────────────────────────────────────────
# def main() -> None:
#     parser = argparse.ArgumentParser(description="Build / refresh a PubMed FAISS DB")
#     parser.add_argument(
#         "--db_name",
#         "-d",
#         default="index",
#         help="Base name for index under pubmed_faiss_index/",
#     )
#     parser.add_argument(
#         "--max_papers",
#         "-p",
#         type=int,
#         default=2500,
#         help="Maximum number of PubMed papers to retrieve",
#     )
#     parser.add_argument(
#         "--fulltext",
#         action="store_true",
#         default=True,
#         help="Also fetch PMC full text when available (slower)",
#     )
#     args = parser.parse_args()

#     # Ensure output directory exists
#     Path("pubmed_faiss_index").mkdir(exist_ok=True)

#     import pandas as pd
#     df = pd.read_csv("pubmeds_keywords.csv")

#     lst_keywords = df["Pubmed keywords"].tolist()

#     run_pipeline(
#         keywords=lst_keywords,
#         db_name=args.db_name,
#         max_papers=args.max_papers,
#         include_fulltext=args.fulltext,
#     )


# if __name__ == "__main__":
#     main()
