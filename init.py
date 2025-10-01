import os
from huggingface_hub import snapshot_download

# 2. 모델 저장 경로 (Google Drive 내부에 캐시/모델 저장)
MODEL_BASE = "/content/drive/MyDrive/hf_models"
os.makedirs(MODEL_BASE, exist_ok=True)
os.makedirs(os.path.join(MODEL_BASE, "transformers_cache"), exist_ok=True)

# 3. 환경 변수 세팅 (Colab 런타임에서 Hugging Face 캐시 경로 고정)
os.environ["HF_HOME"] = MODEL_BASE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(MODEL_BASE, "transformers_cache")

print("HF_HOME =", os.environ["HF_HOME"])
print("TRANSFORMERS_CACHE =", os.environ["TRANSFORMERS_CACHE"])

