import argparse
from huggingface_hub import HfApi
import os

def main():
    parser = argparse.ArgumentParser(description="Upload a folder to Hugging Face Hub.")
    parser.add_argument("--local_path", type=str, required=True, help="Path to the local directory to upload.")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face Hub repository ID.")
    parser.add_argument("--path_in_repo", type=str, required=True, help="Path within the repository to upload to.")
    
    args = parser.parse_args()

    if not os.path.isdir(args.local_path):
        print(f"Error: Local path '{args.local_path}' is not a valid directory.")
        return

    api = HfApi()
    
    print(f"Uploading folder '{args.local_path}' to '{args.repo_id}' in subfolder '{args.path_in_repo}'...")
    
    try:
        api.upload_folder(
            folder_path=args.local_path,
            repo_id=args.repo_id,
            path_in_repo=args.path_in_repo,
            repo_type="model" # Assuming it's a model repo
        )
        print("Upload successful!")
    except Exception as e:
        print(f"An error occurred during upload: {e}")

if __name__ == "__main__":
    main()
