#!/bin/bash

# API Config
export OPENAI_API_KEY="Your OPENAI_API_KEY"
export OPENAI_BASE_URL="http://104.243.40.194:3000/v1"

# Models list to be evaluated
MODELS=(
    "gpt-4o-2024-08-06"
    "claude-3-5-sonnet-20240620"
    "gemini-2.5-flash-preview-05-20"
    "gemini-2.5-pro-preview-05-06"
)


TEMPERATURE=0
TOP_K=50
TOP_P=0.9
MAX_WORKERS=32


DATA_DIR="./images"
MASK_DIR="./masks" 
ANNOTATIONS="./annotations_normxy.json"


OUTPUT_DIR="./results_closed_source_models"
mkdir -p $OUTPUT_DIR


evaluate_model() {
    local model_name=$1
    local model_name_clean=$(echo "$model_name" | sed 's/[^a-zA-Z0-9]/_/g')
    local output_file="$OUTPUT_DIR/${model_name_clean}.json"
    
    echo "========================================"
    echo "Start evaluating: $model_name"
    echo "Output path: $output_file"
    echo "========================================"
    
    export OPENAI_MODEL_NAME="$model_name"
    
    python inference.py \
        --model gpt \
        --data_dir "$DATA_DIR" \
        --mask_dir "$MASK_DIR" \
        --annotations "$ANNOTATIONS" \
        --output "$output_file" \
        --temperature $TEMPERATURE \
        --top_k $TOP_K \
        --top_p $TOP_P \
        --max_workers $MAX_WORKERS
    

    if [ -f "$output_file" ]; then
        echo "Model $model_name finished evalation"
    else
        echo "Error: Cannot find output file $output_file"
    fi
}


main() {
    echo "Starting multi-model evaluation..."
    echo "API URL: $OPENAI_BASE_URL"
    echo "Generation parameters: temperature=$TEMPERATURE, top_k=$TOP_K, top_p=$TOP_P, max_workers=$MAX_WORKERS"
    echo "Output directory: $OUTPUT_DIR"
    echo ""
    
    if [ $# -gt 0 ]; then
        echo "Evaluating specified models: $@"
        for model in "$@"; do
            evaluate_model "$model"
        done
    else
        echo "Evaluating all models: ${MODELS[*]}"
        for model in "${MODELS[@]}"; do
            evaluate_model "$model"
        done
    fi
    
    echo "========================================"
    echo "All model evaluations completed!"
    echo "Result files saved in: $OUTPUT_DIR"
    echo "========================================"
}

# Display usage instructions
show_usage() {
    echo "Usage:"
    echo "  $0                              # Evaluate all models"
    echo "  $0 model1 model2               # Evaluate specified models"
    echo ""
    echo "Available models:"
    for model in "${MODELS[@]}"; do
        echo "  $model"
    done
    echo ""
}

# Process command line arguments
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

# Execute main function
main "$@"