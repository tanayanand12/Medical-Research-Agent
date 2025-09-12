import os
from local_agent.gcp_storage_adapter import GCPStorageAdapter

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gcp_storage = GCPStorageAdapter(
    bucket_name="intraintel-cloudrun-clinical-volume",
    credentials_path="local_agent/service_account_credentials.json"
)

index_path = os.path.join("pubmed_faiss_index", f"index.index")
meta_pkl_path = os.path.join("pubmed_faiss_index", f"index.meta.pkl")

logger.info("Checking if index path exists")

# check if index path does not exist, make a directory
if not os.path.exists("pubmed_faiss_index"):
    os.makedirs("pubmed_faiss_index")

if not os.path.exists(index_path):
    logger.info("Index path does not exist, downloading index files")
    index_blob = gcp_storage.bucket.blob(f"indexes/pubmed_embeddings/index.index")
    index_blob.download_to_filename(index_path)

    meta_pkl_blob = gcp_storage.bucket.blob(f"indexes/pubmed_embeddings/index.meta.pkl")
    meta_pkl_blob.download_to_filename(meta_pkl_path)

    nested_index_path = os.path.join("pubmed_local_agent", "pubmed_faiss_index", f"index.index")
    nested_meta_pkl_path = os.path.join("pubmed_local_agent", "pubmed_faiss_index", f"index.meta.pkl")

    logger.info("Creating nested directory for index files")

    if not os.path.exists("pubmed_local_agent/pubmed_faiss_index"):
        os.makedirs("pubmed_local_agent/pubmed_faiss_index")

    # copy files to nested directory
    os.system(f"cp {index_path} {nested_index_path}")
    os.system(f"cp {meta_pkl_path} {nested_meta_pkl_path}")

    logger.info("Index files downloaded")

logger.info("Initializing Orchestrator and Aggregator")

