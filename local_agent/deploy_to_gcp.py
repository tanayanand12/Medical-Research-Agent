# deploy_to_gcp.py
import argparse
import os
import subprocess
import logging
from gcp_storage_adapter import GCPStorageAdapter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_and_push_container(project_id: str, image_name: str) -> bool:
    """
    Build and push Docker container to Google Container Registry.
    
    Args:
        project_id: GCP project ID
        image_name: Name for the container image
        
    Returns:
        Success status
    """
    try:
        image_uri = f"gcr.io/{project_id}/{image_name}"
        
        # Build the container
        logger.info(f"Building container image: {image_uri}")
        subprocess.run(
            ["docker", "build", "-t", image_uri, "."], 
            check=True
        )
        
        # Push to GCR
        logger.info(f"Pushing container image to GCR: {image_uri}")
        subprocess.run(
            ["docker", "push", image_uri], 
            check=True
        )
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error building or pushing container: {e}")
        return False

def deploy_to_cloud_run(
    project_id: str, 
    region: str, 
    service_name: str, 
    image_name: str,
    env_vars: dict
) -> bool:
    """
    Deploy container to Cloud Run.
    
    Args:
        project_id: GCP project ID
        region: GCP region
        service_name: Cloud Run service name
        image_name: Container image name
        env_vars: Environment variables
        
    Returns:
        Success status
    """
    try:
        image_uri = f"gcr.io/{project_id}/{image_name}"
        
        # Build command
        cmd = [
            "gcloud", "run", "deploy", service_name,
            "--image", image_uri,
            "--platform", "managed",
            "--region", region,
            "--allow-unauthenticated",
            "--memory", "2Gi",
            "--cpu", "1",
            "--project", project_id
        ]
        
        # Add environment variables
        for key, value in env_vars.items():
            cmd.extend(["--set-env-vars", f"{key}={value}"])
        
        # Deploy to Cloud Run
        logger.info(f"Deploying to Cloud Run: {service_name}")
        subprocess.run(cmd, check=True)
        
        # Get the service URL
        url_cmd = [
            "gcloud", "run", "services", "describe", service_name,
            "--platform", "managed",
            "--region", region,
            "--format", "value(status.url)",
            "--project", project_id
        ]
        result = subprocess.run(url_cmd, capture_output=True, text=True, check=True)
        service_url = result.stdout.strip()
        
        logger.info(f"Deployed successfully. Service URL: {service_url}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error deploying to Cloud Run: {e}")
        return False

def main():
    parser = argparse.Argument