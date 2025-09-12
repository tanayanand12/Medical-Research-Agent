from gcp_storage_adapter import GCPStorageAdapter
from google.cloud import storage

gcp_storage = GCPStorageAdapter(bucket_name="intraintel-cloudrun-clinical-volume", credentials_path="service_account_credentials.json")

# intraintel-cloudrun-clinical-volume/localDatasets/qEr63Mt9iardCOi8KUFh

model_id="qEr63Mt9iardCOi8KUFh"
print(gcp_storage.list_pdfs(f"localDatasets/{model_id}/"))

# print(gcp_storage.download_pdfs_to_temp_using_model_id(model_id=model_id))

print(gcp_storage.upload_index_to_model_id(model_id=model_id, index_path="gcp-indexes/qEr63Mt9iardCOi8KUFh"))

# print(gcp_storage.download_index_using_model_id(model_id=model_id, local_path="gcp-indexes/qEr63Mt9iardCOi8KUFh"))
