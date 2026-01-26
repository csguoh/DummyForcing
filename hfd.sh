#!/bin/bash

# Script to download the latest version of a Hugging Face model
# Usage: ./download_hf_model.sh <model_name> [output_dir]

set -e

# Check if model name is provided
if [ -z "$1" ]; then
    echo "Error: Model name is required"
    echo "Usage: $0 <model_name> [output_dir]"
    echo "Example: $0 meta-llama/Llama-2-7b-hf ./models"
    exit 1
fi

MODEL_NAME="$1"

# Extract model folder name (part after the last '/')
MODEL_FOLDER="${MODEL_NAME##*/}"

# If output directory is provided, use it; otherwise use current directory
if [ -n "$2" ]; then
    OUTPUT_DIR="$2/$MODEL_FOLDER"
else
    OUTPUT_DIR="./$MODEL_FOLDER"
fi

echo "================================================"
echo "Downloading model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "================================================"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli is not installed"
    echo "Please install it using: pip install huggingface_hub"
    exit 1
fi

# Download only the latest version (no git history)
# The --local-dir-use-symlinks False option ensures actual files are downloaded
# instead of symlinks, and prevents downloading git history
huggingface-cli download "$MODEL_NAME" \
    --local-dir "$OUTPUT_DIR" \
    --local-dir-use-symlinks False

echo "================================================"
echo "Download completed successfully!"
echo "Model saved to: $OUTPUT_DIR"
echo "================================================"