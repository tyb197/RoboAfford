import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from tqdm import tqdm


json_path = "./results_open_source_models/qwen25vl-7b-sft.json"
image_dir = "./images"
output_dir = "./visualize_results"


os.makedirs(output_dir, exist_ok=True)


categories = ["object affordance prediction", "object affordance recognition", "spatial affordance localization"]
for category in categories:
    os.makedirs(os.path.join(output_dir, category.replace(" ", "_")), exist_ok=True)

def is_normalized_coords(points, img_width, img_height):
    for x, y in points:
        if x > 1 or y > 1:
            return False
    return True

def visualize_points(image_path, points, output_path, question=None):
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        if is_normalized_coords(points, img_width, img_height):
            points = [(x * img_width, y * img_height) for (x, y) in points]
        
        fig, ax = plt.subplots(1, figsize=(10, 6))
        plt.imshow(img)
        
        color_in = 'cyan'
        color_e = 'white'
        linewidth = 2
        
        for (x, y) in points:
            circle = patches.Circle((x, y), radius=8, edgecolor=color_e, 
                                  facecolor=color_in, linewidth=linewidth)
            ax.add_patch(circle)
        
        plt.axis('off')
        if question:
            title = question[:100] + '...' if len(question) > 100 else question
            plt.title(title, fontsize=10, pad=10)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        return True
    except Exception as e:
        print(f"Error occurred when processing {image_path}: {str(e)}")
        return False

with open(json_path, 'r') as f:
    data = json.load(f)

success_count = 0
total_count = len(data['detailed_results'])

with tqdm(data['detailed_results'], desc="Visualization Process") as pbar:
    for result in pbar:
        image_name = result['image']
        category = result['category']
        question = result['question']
        
        pred_points = result['pred_points']
        if not pred_points:
            continue
        
        category_dir = category.replace(" ", "_")
        output_filename = f"{os.path.splitext(image_name)[0]}_{category_dir}.png"
        output_path = os.path.join(output_dir, category_dir, output_filename)
        
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            if visualize_points(image_path, pred_points, output_path, question):
                success_count += 1
            pbar.set_postfix({"Success": success_count, "Fail": pbar.n - success_count + 1})
        else:
            print(f"Image file does not exist: {image_path}")

print(f"\nVisualization Finished! Processed {success_count}/{total_count} results")
print(f"Results saved: {output_dir}")