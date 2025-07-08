#!/usr/bin/env bash
set -e

# Define URLs for the required files
BEST_MODEL_URL="https://cdn.bridgelabs.tech/ghana-ai-hackathon/food-classifier/best_model.pth"
CLASS_MAPPING_URL="https://cdn.bridgelabs.tech/ghana-ai-hackathon/food-classifier/class_mapping.json"

# Define target directories
CHECKPOINTS_DIR="$(dirname "$0")/checkpoints"
CLASS_MAPPING_PATH="$(dirname "$0")/class_mapping.json"
BEST_MODEL_PATH="$CHECKPOINTS_DIR/best_model.pth"

# Create checkpoints directory if it doesn't exist
mkdir -p "$CHECKPOINTS_DIR"

# Backup old files if they exist
if [ -f "$BEST_MODEL_PATH" ]; then
    mv "$BEST_MODEL_PATH" "$BEST_MODEL_PATH.bak"
    echo "Backed up old best_model.pth to best_model.pth.bak"
fi
if [ -f "$CLASS_MAPPING_PATH" ]; then
    mv "$CLASS_MAPPING_PATH" "$CLASS_MAPPING_PATH.bak"
    echo "Backed up old class_mapping.json to class_mapping.json.bak"
fi

# Download best_model.pth
curl -L "$BEST_MODEL_URL" -o "$BEST_MODEL_PATH"

# Download class_mapping.json
curl -L "$CLASS_MAPPING_URL" -o "$CLASS_MAPPING_PATH"

echo "Download complete. Files are in place." 
