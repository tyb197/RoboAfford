# python inference.py --model molmo --model_path /share/project/tangyingbo/checkpoint/Molmo-7B-D-0924 --annotations "annotations_normxy.json" --output "./results_open_source_models/molmo.json"

# python inference.py --model qwen2vl --model_path /share/project/tangyingbo/checkpoint/Qwen2-VL-7B-Instruct --annotations "annotations_normxy.json" --output "./results_open_source_models/qwen2vl.json"

# python inference.py --model llava_next --model_path /share/project/tangyingbo/checkpoint/llama3-llava-next-8b --annotations "annotations_normxy.json" --output "./results_open_source_models/llava_next.json"

# python inference.py --model spatialvlm --model_path /share/project/tangyingbo/checkpoint/SpaceMantis --annotations "annotations_normxy.json" --output "./results_open_source_models/spacemantis.json"

# python inference.py --model robopoint --model_path /share/project/tangyingbo/checkpoint/robopoint-v1-vicuna-v1.5-13b --annotations "annotations_normxy.json" --output "./results_open_source_models/robopoint.json"

# python inference.py --model qwen25vl-3b --model_path /share/project/tangyingbo/checkpoint/Qwen2.5-VL-3B-Instruct --annotations "annotations_absxy.json" --output "./results_open_source_models/qwen25vl-3b.json"

# python inference.py --model qwen25vl-7b --model_path /share/project/tangyingbo/checkpoint/Qwen2.5-VL-7B-Instruct --annotations "annotations_absxy.json" --output "./results_open_source_models/qwen25vl-7b.json"

python inference.py --model qwen25vl-7b-sft --model_path /share/project/tangyingbo/LLaMA-Factory/saves/qwen25vl-7b-sft --annotations "annotations_absxy.json" --output "./results_open_source_models/qwen25vl-7b-sft.json"
