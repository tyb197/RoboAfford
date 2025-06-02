import os
import json
import re
import numpy as np
from PIL import Image
from tqdm import tqdm

def parse_points_from_answer(answer_text):
    points = []
    
    patterns = [
        r'\((\d+\.?\d*)\s*,\s*(\d+\.?\d*)\)',  # (x,y) or (x, y)
        r'\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]',  # [x,y] or [x, y]
        r'(\d+\.?\d*)\s+(\d+\.?\d*)',          # x y
        r'(\d+\.?\d*),(\d+\.?\d*)'             # x,y
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, answer_text)
        if matches:
            for match in matches:
                try:
                    x = float(match[0])
                    y = float(match[1])
                    points.append((x, y))
                except (ValueError, IndexError):
                    continue
    
    return points

def calculate_accuracy(pred_points, mask_path):
    if not pred_points:
        return 0.0
    
    try:
        mask_img = Image.open(mask_path)
        mask = np.array(mask_img) / 255.0
        
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
            
        height, width = mask.shape
        correct_count = 0
        
        is_normalized = all(0 <= x <= 1 and 0 <= y <= 1 for x, y in pred_points)
        
        for x, y in pred_points:
            if is_normalized:
                x = int(round(x * width))
                y = int(round(y * height))
            else:
                x = int(round(x))
                y = int(round(y))
            
            if 0 <= x < width and 0 <= y < height:
                if mask[y, x] > 0.5:
                    correct_count += 1
        
        accuracy = correct_count / len(pred_points)
        return accuracy
        
    except Exception as e:
        print(f"Error calculating accuracy: {e}")
        return 0.0

def evaluate_model_results(gt_data, model_results, data_dir):
    evaluation_results = []
    total_accuracy = 0.0
    valid_questions = 0
    
    result_map = {}
    for result in model_results:
        key = (result.get("question", ""), result.get("img", ""))
        result_map[key] = result
    
    for gt_entry in tqdm(gt_data, desc="Evaluating"):
        question = gt_entry.get("question", "")
        mask_rel_path = gt_entry.get("mask", "")
        image_rel_path = gt_entry.get("img", "")
        category = gt_entry.get("category", "unknown")
        
        if not question or not mask_rel_path or not image_rel_path:
            continue
            
        mask_path = os.path.join(data_dir, mask_rel_path)
        image_path = os.path.join(data_dir, image_rel_path)
        
        key = (question, image_rel_path)
        model_result = result_map.get(key)
        
        if not model_result:
            continue
            
        generated_answer = model_result.get("answer", "")
        pred_points = parse_points_from_answer(generated_answer)
        
        accuracy = calculate_accuracy(pred_points, mask_path)
        
        evaluation_results.append({
            "question": question,
            "expected_answer": str(gt_entry.get("answer", "")),
            "generated_answer": generated_answer,
            "accuracy": accuracy,
            "category": category,
            "image": image_path
        })
        
        total_accuracy += accuracy
        valid_questions += 1
    
    avg_accuracy = total_accuracy / valid_questions if valid_questions > 0 else 0.0
    print(f"\nAverage Accuracy: {avg_accuracy:.4f}")
    
    return evaluation_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", required=True, default="RoboAffordance-Eval/roboaffordance/annotations.json", help="Path to ground truth JSON file")
    parser.add_argument("--model_results_file", required=True, help="Path to model results JSON file")
    parser.add_argument("--data_dir", required=True, default="RoboAffordance-Eval/roboaffordance/images", help="Root directory of dataset")
    parser.add_argument("--output_file", default="./results/evaluation_results.json", help="Output file for evaluation results")
    args = parser.parse_args()
    
    with open(args.gt_file, "r") as f:
        gt_data = json.load(f)
    
    with open(args.model_results_file, "r") as f:
        model_results = json.load(f)
    
    results = evaluate_model_results(gt_data, model_results, args.data_dir)
    
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {args.output_file}")