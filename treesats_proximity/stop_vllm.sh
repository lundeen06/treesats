#!/bin/bash

echo "Stopping Cosmos Reason 2 vLLM Server..."
docker stop cosmos-server
docker rm cosmos-server
echo "✓ Server stopped"
