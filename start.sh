#!/bin/bash

echo "Starting Phytologic AI Server..."

uvicorn api:app \
  --host 0.0.0.0 \
  --port $PORT \
  --workers 1
