"""
Dynamic Task-Object Relevance Matrix Generator
----------------------------------------------
This script processes an uploaded image, dynamically detects objects within it 
using YOLOv8, and then uses Sentence-BERT to compute the semantic relevance 
between predefined human tasks and the specifically detected objects.
"""

import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import YOLO, prompt the user if it's missing
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: missing required library for object detection.")
    print("Please run: pip install ultralytics")
    sys.exit(1)

def main(image_path, tasks=None):
    # If no specific tasks are provided, default to the 14 project tasks
    if tasks is None:
        tasks = [
            "drink", "cut food", "take photo", "read", "ride", "play sport", 
            "use computer", "eat", "groom", "sit", "cook", "repair", 
            "talk on phone", "watch TV"
        ]

    print(f"\n--- STEP 1: OBJECT FILTRATION FROM IMAGE ---")
    print(f"Loading YOLOv8 model to analyze image: {image_path}...")
    
    # We use YOLOv8n (nano). It will automatically download the small weight file (yolov8n.pt) on the first run.
    yolo_model = YOLO('yolov8n.pt') 
    
    try:
        # Run inference on the provided image
        results = yolo_model(image_path, verbose=False)
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)
        
    # Extract classes detected
    detected_classes = results[0].boxes.cls.cpu().numpy()
    
    # Map class integers to string labels
    detected_objects = [yolo_model.names[int(cls)] for cls in detected_classes]
    
    # Remove duplicates to get a unique list of objects present in the image
    objects = list(set(detected_objects))
    
    if not objects:
        print("\nNo objects were detected in the image. Exiting calculations.")
        return
        
    print(f"SUCCESS: Detected {len(objects)} unique objects: {', '.join(objects)}\n")

    print(f"--- STEP 2: SEMANTIC EMBEDDINGS ---")
    print("Loading pretrained Sentence-BERT model (all-MiniLM-L6-v2)...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Encoding tasks and the detected objects into vector spaces...")
    task_embeddings = sbert_model.encode(tasks)
    object_embeddings = sbert_model.encode(objects)
    
    print(f"  Task vector shape: {task_embeddings.shape}") 
    print(f"  Detected Objects vector shape: {object_embeddings.shape}\n")

    print(f"--- STEP 3: RELEVANCE CALCULATION ---")
    print("Computing cosine similarities between tasks and detected objects...")
    relevance_matrix = cosine_similarity(task_embeddings, object_embeddings)

    # Scale to 0-1 context. Negative values -> completely irrelevant (0)
    relevance_matrix_scaled = np.clip(relevance_matrix, 0.0, 1.0)

    print(f"Dynamic Matrix Shape Generated: {relevance_matrix_scaled.shape}")
    
    print("\n--- MATRIX INSIGHTS ---")
    # For each task, check which DETECTED object is the most relevant
    for t_idx, task in enumerate(tasks):
        similarities = relevance_matrix_scaled[t_idx]
        best_obj_idx = np.argmax(similarities)
        best_score = similarities[best_obj_idx]
        
        # Only print if there's a somewhat strong logical connection 
        # (e.g. if the image only has a 'car' and the task is 'eat', score will be low)
        if best_score > 0.25: 
            print(f"  ✓ Action '{task}' aligns best with detected object: '{objects[best_obj_idx]}' (Co-relation Score: {best_score:.4f})")
        else:
            print(f"  ✗ Action '{task}' has no highly relevant objects in this image (Best was '{objects[best_obj_idx]}' at {best_score:.2f})")

    # Final Step: Export as standard float32 bin block
    relevance_matrix_final = relevance_matrix_scaled.astype(np.float32)
    filename = "dynamic_relevance.bin"
    print(f"\n[Disk Export] Saving matrix to '{filename}'...")
    relevance_matrix_final.tofile(filename)
    print("Process Complete!")

if __name__ == "__main__":
    # Allows passing image from terminal e.g. `python main.py my_photo.jpg`
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Standard fallback terminal input prompt
        print("-" * 50)
        img_path = input("Enter the path/name of your input image: ").strip()
        print("-" * 50)
        
    if img_path:
        main(img_path)
    else:
        print("No image path provided.")
