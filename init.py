import os
from huggingface_hub import snapshot_download

# 네 로컬 경로로 바꿔라 (절대경로 사용 권장)
MODEL_BASE = os.path.expanduser("~/paper_repo/hf_models")

# 디렉토리 만들기 (존재하면 무시)
os.makedirs(MODEL_BASE, exist_ok=True)
os.makedirs(os.path.join(MODEL_BASE, "transformers_cache"), exist_ok=True)

# 환경변수로 지정 (현재 프로세스에만 적용)
os.environ["HF_HOME"] = MODEL_BASE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(MODEL_BASE, "transformers_cache")

print("HF_HOME =", os.environ["HF_HOME"])
print("TRANSFORMERS_CACHE =", os.environ["TRANSFORMERS_CACHE"])

local_dir = os.path.expanduser("~/paper_repo/hf_models/models")
snapshot_download(repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct", cache_dir=os.path.expanduser("~/paper_repo/hf_models"), local_dir=local_dir)
print("downloaded to", local_dir)
