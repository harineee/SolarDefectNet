#!/bin/bash

# ========================
#  CONFIGURATION
# ========================

# Environment
CONDA_ENV="solar"

# Model Settings
MODEL_PATH="/home/debonaire/h-proj/models/best_ensemble_model.pth"
USE_GPU=true

# Image Processing
IMAGE_SIZE=224          # ResNet input size
THRESHOLD=0.2          # Inactive pixel threshold for physics model

# Input/Output
INPUT_FILE="/home/debonaire/h-proj/data/processed/image_9.png"    # Set your input image path here
OUTPUT_DIR="results"

# ========================
#  SCRIPT START
# ========================

# Show configuration
echo "
🔧 Configuration:
-------------------
Model Path    : $MODEL_PATH
Input File   : $INPUT_FILE
Output Dir   : $OUTPUT_DIR
GPU Enabled  : $USE_GPU
Image Size   : $IMAGE_SIZE
Threshold    : $THRESHOLD
"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
if ! conda activate $CONDA_ENV; then
    echo "Failed to activate conda environment: $CONDA_ENV"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Input file not found: $INPUT_FILE"
    exit 1
fi

# Process image
echo "Processing image: $INPUT_FILE"
python /home/debonaire/h-proj/scripts/inference.py \
    --image "$INPUT_FILE" \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_DIR" \
    --gpu $USE_GPU \
    --size $IMAGE_SIZE \
    --threshold $THRESHOLD

if [ $? -eq 0 ]; then
    echo "Processing complete!"
    echo "Results saved in: $OUTPUT_DIR"
    
    # Display results from the saved file
    result_file="$OUTPUT_DIR/$(basename "${INPUT_FILE%.*}")_results.txt"
    if [ -f "$result_file" ]; then
        echo -e "\nSummary of Results:"
        echo "-------------------"
        cat "$result_file"
    fi
else
    echo "Processing failed!"
    exit 1
fi 