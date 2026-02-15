#!/bin/bash
# Start NVIDIA Cosmos Reason locally on GTX GPU

echo "Starting Cosmos Reason 2-8B on local GPU..."
echo "This will take a few minutes to download and load the model."
echo ""

docker run --runtime=nvidia --gpus all \
  --shm-size=32g -p 8000:8000 \
  -e NGC_API_KEY=$NGC_API_KEY \
  nvcr.io/nim/nvidia/cosmos-reason2-8b:latest
