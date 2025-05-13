from huggingface_hub import HfApi, create_repo, login
import os
from pathlib import Path

def upload_model_to_hf(
    model_path: str,
    repo_name: str,
    token: str,
    repo_type: str = "model",
    private: bool = False
):
    """
    Upload a model to HuggingFace Hub
    
    Args:
        model_path: Path to the model directory or file
        repo_name: Name for the HuggingFace repository (format: username/repo_name)
        token: HuggingFace API token
        repo_type: Type of repository ("model" or "dataset")
        private: Whether the repository should be private
    """
    # Initialize the HuggingFace API and login
    api = HfApi()
    
    # Login using the token
    login(token=token)
    
    try:
        # Create the repository
        create_repo(
            repo_id=repo_name,
            token=token,
            private=private,
            repo_type=repo_type,
            exist_ok=True
        )
        
        # Upload the model files
        model_path = Path("/home/user/Desktop/Test/Exodus/Fine-tune/merged_model")
        if model_path.is_file():
            # Single file upload
            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=model_path.name,
                repo_id=repo_name,
                repo_type=repo_type
            )
        else:
            # Directory upload
            api.upload_folder(
                folder_path=model_path,
                repo_id=repo_name,
                repo_type=repo_type
            )
        
        print(f"Successfully uploaded to: https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"Error uploading model: {str(e)}")

if __name__ == "__main__":
    # Get token from environment variable for security
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Please set your HuggingFace token as an environment variable named 'HF_TOKEN'")
        exit(1)
    
    # Your specific details are already set
    model_path = "Fine-tune"
    repo_name = "yazeed-mshayekh/Exodus-Arabic-Model"
    
    upload_model_to_hf(
        model_path=model_path,
        repo_name=repo_name,
        token=token,
        private=False
    ) 