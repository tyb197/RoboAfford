# RoboAfford: A Dataset and Benchmark for Enhancing Object and Spatial Affordance Learning in Robot Manipulation

## Environment Setup
```
git clone https://github.com/tyb197/RoboAfford.git
cd RoboAfford
conda create -n roboafford python=3.10 -y
conda activate roboafford
pip install -r requirements.txt
```
The above environment supports evaluation of both closed-source models (GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Flash, and Gemini-1.5-Pro) and the following open-source models: Qwen2-VL and Qwen2.5-VL.

For other open-source models, we recommend creating separate conda environments for each model following their official GitHub repositories.

## Evaluation
### Closed-Source Models
For closed-source model evaluation, configure your OpenAI API key in `eval_closed_source_models.sh`:
```
export OPENAI_API_KEY="Your OPENAI_API_KEY"
```
Then run the following command:
```
bash eval_closed_source_models.sh
```

### Open-Source Models
For open-source model evaluation, we recommend downloading the checkpoints from Hugging Face first, then loading them from local paths. Supported models include:
- **Molmo**: allenai/Molmo-7B-D-0924
- **Qwen2-VL**: Qwen/Qwen2-VL-7B-Instruct
- **LLaVA-Next**: lmms-lab/llama3-llava-next-8b
- **SpaceMantis**: remyxai/SpaceMantis
- **RoboPoint**: wentao-yuan/robopoint-v1-vicuna-v1.5-13b
- **Qwen2.5-VL**: Qwen/Qwen2.5-VL-3B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct

Run the evaluation with:
```
python inference.py \
--model qwen25vl-7b \
--model_path /your_path_to_checkpoint/Qwen2.5-VL-7B-Instruct \
--annotations "annotations_absxy.json" \
--output "./results_open_source_models/qwen25vl-7b.json"
```
We provide annotation files in two formats to accommodate different model output formats:
- **annotations_normxy.json** (values between 0-1)
- **annotations_absxy.json** (pixel values)

You can specify your preferred format using the `--annotations` argument when running the evaluation. For implementation details, please refer to the `eval_open_source_models.sh` script.

### Visualization
After evaluation, visualize the results using:
```
python visualize.py --json_path "./results_open_source_models/qwen25vl-7b.json" --image_dir "./images" --output_dir "./visualize_results"
```
