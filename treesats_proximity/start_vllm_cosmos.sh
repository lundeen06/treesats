#!/bin/bash

echo "Starting Cosmos Reason 2 vLLM Server..."
echo "========================================"

# Stop any existing container with the same name
docker stop cosmos-server 2>/dev/null
docker rm cosmos-server 2>/dev/null

# Start vLLM server
docker run -d \
  --name cosmos-server \
  --gpus all \
  -v ~/treesats/treesats_proximity/cosmos-reason2:/model \
  -p 8000:8000 \
  nvcr.io/nvidia/vllm:26.01-py3 \
  vllm serve /model --tensor-parallel-size 2

echo ""
echo "Server starting in background..."
echo "To check status: docker logs -f cosmos-server"
echo "To stop server: docker stop cosmos-server"
echo ""
echo "Waiting for server to be ready..."

# Wait for server to be ready
for i in {1..60}; do
  if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "✓ Server is ready!"
    echo "You can now run: python assessment.py data/sat1.jpg"
    exit 0
  fi
  echo -n "."
  sleep 2
done

echo ""
echo "Server is still loading. Check logs with: docker logs -f cosmos-server"
EOF 

chmod +x start_vllm_cosmos.sh
