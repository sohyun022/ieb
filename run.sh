COMMON_ARGS='--gpus all \
    --env CUDA_VISIBLE_DEVICES=0,1 \
    -v ~/vllm/vllm:/root/.cache/vllm \
    -v ./outputs:/app/outputs \
    -v ./data:/app/dataset \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --env CUDA_VISIBLE_DEVICES=1 \
    —env "HUGGING_FACE_HUB_TOKEN=" \
    —env "VLLM_WORKER_MULTIPROC_METHOD=spawn" \
    -p 8002:8002 \
    —ipc=host'