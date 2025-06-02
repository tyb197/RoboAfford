import os
import sys
import json
import base64
import random
import numpy as np
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading


from evaluation import (
    parse_points_from_answer,
    evaluate_model_results,
    calculate_accuracy
)

def encode_image(image_path):
    """Base64-encode an image file (for GPT-based or generic usage)."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def import_model_modules(model_name):
    """
    Dynamically import only the required modules for the specified model.
    Returns the appropriate model loading and running functions.
    """
    if model_name == "llava_next":
        from models import load_llava_next_model, run_llava_next
        return load_llava_next_model, run_llava_next
    elif model_name.startswith("spatialvlm"):
        from models import load_spatialvlm_model, run_spatialvlm
        return load_spatialvlm_model, run_spatialvlm
    elif model_name.startswith("robopoint"):
        from models import load_robopoint_model, run_robopoint
        return load_robopoint_model, run_robopoint
    elif model_name.startswith("qwen2vl"):
        from models import load_qwen2vl_model, run_qwen2vl
        return load_qwen2vl_model, run_qwen2vl
    elif model_name.startswith("qwen25vl"):
        from models import load_qwen25vl_model, run_qwen25vl
        return load_qwen25vl_model, run_qwen25vl
    elif model_name.startswith("molmo"):
        from models import load_molmo_model, run_molmo
        return load_molmo_model, run_molmo
    elif model_name.startswith("gpt"):
        from models import load_gpt_model, send_question_to_openai
        return load_gpt_model, send_question_to_openai
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def load_model(model_name, model_path=None):
    """
    Load the chosen model once, returning any needed kwargs (tokenizer, model object, etc.).
    If model_path is provided, it overrides the default model checkpoint.
    """
    # For API models, we don't need torch
    if model_name.startswith("gpt"):
        load_func, _ = import_model_modules(model_name)
        return load_func(model_path)
    
    # Import torch only when we actually need to load a model
    import torch
    global device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    # Dynamically import only what we need
    load_func, _ = import_model_modules(model_name)
    return load_func(model_path)

def run_single_inference(item_data):
    item, model_name, model_kwargs, generation_params, data_dir, mask_dir = item_data
    
    try:
        image_path = os.path.join(data_dir, item["img"])
        
        mask_path = os.path.join(mask_dir, item["mask"])
        
        category = item["category"]
        question = item["question"]

        if category == "object affordance":
            category = "object affordance prediction"
        elif category == "object reference":
            category = "object affordance recognition"
        elif category == "spatial affordance":
            category = "spatial affordance localization"

        # Run model inference
        answer = run_model(question, image_path, model_name, model_kwargs, generation_params)
        
        # Parse points from model answer
        pred_points = parse_points_from_answer(answer)
        
        # Calculate accuracy
        accuracy = calculate_accuracy(pred_points, mask_path)
        
        # Store results
        result = {
            "image": item["img"],
            "question": question,
            "category": category,
            "answer": answer,
            "accuracy": accuracy,
            "pred_points": pred_points
        }
        
        return result, None  # (result, error)
        
    except Exception as e:
        error_result = {
            "image": item["img"],
            "question": item.get("question", ""),
            "category": category,
            "answer": f"Inference Failed: {str(e)}",
            "accuracy": 0.0,
            "pred_points": []
        }
        return error_result, str(e)

def run_model(question, image_path, model_name, model_kwargs, generation_params=None):
    """
    Generic helper: runs the specified model on a single (question, image).
    Returns the string answer from the model.
    """
    if not os.path.exists(image_path):
        print(f"[WARNING] Image path not found: {image_path}")

    if generation_params is None:
        generation_params = {}

    # Access the correct run function for this model
    _, run_func = import_model_modules(model_name)
    
    if model_name.startswith("gpt"):
        # For GPT-based models, pass base64 and api_config
        image_base64 = encode_image(image_path)
        answer = run_func(question, image_base64, model_kwargs, **generation_params)
    else:
        answer = run_func(question, image_path, model_kwargs, **generation_params)
        
    return answer

def check_file_exists(base_path, extensions=['.png', '.jpg', '.jpeg']):
    for ext in extensions:
        full_path = f"{base_path}{ext}"
        if os.path.exists(full_path):
            return full_path
    return f"{base_path}{extensions[0]}"

def main():
    parser = argparse.ArgumentParser(description="Evaluate vision-language models on spatial reasoning tasks with concurrent API calls")
    parser.add_argument("--model", required=True, 
                        choices=["llava_next", "spatialvlm", "robopoint", "qwen2vl", "qwen25vl-3b", "qwen25vl-7b", "qwen25vl-3b-sft", "qwen25vl-7b-sft", "molmo", "gpt"],
                        help="Name of the model to evaluate")
    parser.add_argument("--model_path", default=None, 
                        help="Optional path to model checkpoint (overrides default)")
    parser.add_argument("--data_dir", default="./images", 
                        help="Directory containing images")
    parser.add_argument("--mask_dir", default="./masks", 
                        help="Directory containing mask images")
    parser.add_argument("--annotations", default="annotations_normxy.json", 
                        help="Path to annotations file")
    parser.add_argument("--output", default="results.json", 
                        help="Output file for evaluation results")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for text generation (default: 0.1)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k value for text generation (default: 50)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p value for text generation (default: 0.9)")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Maximum number of concurrent API calls (default: 8)")
    args = parser.parse_args()

    # Load annotations
    with open(args.annotations, "r") as f:
        annotations = json.load(f)

    # Optionally sample a subset
    if args.num_samples is not None and args.num_samples < len(annotations):
        annotations = random.sample(annotations, args.num_samples)

    # Load model
    print(f"Loading {args.model} model...")
    model_kwargs = load_model(args.model, args.model_path)
    
    generation_params = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p
    }
    
    if not args.model.startswith("gpt"):
        print("Local models, using single-thread mode...")
        args.max_workers = 1
    else:
        print(f"API models, using concurrent mode with max workers: {args.max_workers}")
    
    results = []
    category_stats = {}
    error_count = 0
    
    task_data = [
        (item, args.model, model_kwargs, generation_params, args.data_dir, args.mask_dir)
        for item in annotations
    ]
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_item = {
            executor.submit(run_single_inference, data): data[0]
            for data in task_data
        }
        
        with tqdm(total=len(annotations), desc="Evaluating samples") as pbar:
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result, error = future.result()
                    results.append(result)
                    
                    if error:
                        error_count += 1
                        print(f"Error processing {item['img']}: {error}")
                    
                    # Update category statistics
                    category = result["category"]
                    if category not in category_stats:
                        category_stats[category] = {
                            "total": 0,
                            "correct": 0.0,
                            "samples": []
                        }
                    category_stats[category]["total"] += 1
                    category_stats[category]["correct"] += result["accuracy"]
                    category_stats[category]["samples"].append({
                        "image": result["image"],
                        "accuracy": result["accuracy"]
                    })
                    
                except Exception as exc:
                    error_count += 1
                    print(f"Task generated an exception for {item['img']}: {exc}")
                    
                    error_result = {
                        "image": item["img"],
                        "question": item.get("question", ""),
                        "category": item.get("category", "unknown"),
                        "answer": f"Failed: {str(exc)}",
                        "accuracy": 0.0,
                        "pred_points": []
                    }
                    results.append(error_result)
                
                pbar.update(1)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate and print summary statistics
    print(f"\nEvaluation Summary:")
    print("=" * 60)
    print(f"Total time: {total_time:.2f}秒")
    print(f"Time per sample: {total_time/len(annotations):.2f}秒")
    print(f"Workers: {args.max_workers}")
    print(f"Error counts: {error_count}/{len(annotations)}")
    print("=" * 60)
    
    total_samples = 0
    total_correct = 0.0
    
    for category, stats in category_stats.items():
        category_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"{category.capitalize():<20}: Accuracy = {category_acc:.4f} ({stats['total']} samples)")
        total_samples += stats["total"]
        total_correct += stats["correct"]
    
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    print("=" * 60)
    print(f"{'Overall':<20}: Accuracy = {overall_acc:.4f} ({total_samples} samples)")
    
    # Save detailed results
    output_data = {
        "model": args.model,
        "overall_accuracy": overall_acc,
        "category_accuracies": {k: v["correct"]/v["total"] if v["total"] > 0 else 0.0 for k, v in category_stats.items()},
        "detailed_results": results,
        "generation_params": generation_params,
        "evaluation_stats": {
            "total_time": total_time,
            "avg_time_per_sample": total_time/len(annotations),
            "max_workers": args.max_workers,
            "error_count": error_count,
            "total_samples": len(annotations)
        }
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to {args.output}")

if __name__ == "__main__":
    main()