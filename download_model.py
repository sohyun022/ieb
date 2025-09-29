from huggingface_hub import snapshot_download

local_dir = os.path.expanduser("~/paper_repo/hf_models/models")
snapshot_download(repo_id="MODEL_ID_OR_PATH", cache_dir=os.path.expanduser("~/paper_repo/hf_models"), local_dir=local_dir)
print("downloaded to", local_dir)
