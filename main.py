"""
Task-Object Relevance Matrix Generator using Sentence-BERT
----------------------------------------------------------
This script dynamically generates a matrix representing the semantic relevance 
between predefined human tasks and user-input objects (or COCO dataset objects).
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    # STEP 1: Load pretrained Sentence-BERT model
    print("Loading pretrained Sentence-BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # DATA DEFINITIONS
    tasks = [
        "step on something", "sit comfortably", "place flowers", 
        "get potatoes out of fire", "water plant", "get lemon out of tea", 
        "dig hole", "open bottle of beer", "open parcel", "serve wine", 
        "pour sugar", "smear butter", "extinguish fire", "pound carpet"
    ]

    coco_objects = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    # STEP 2: Get user input
    print("\n" + "="*60)
    print("Task-Object Relevance Matrix Generator")
    print("="*60)
    user_input = input("Enter an object name (or comma-separated list of objects).\nLeave blank to use the default 80 COCO objects: ").strip()

    if not user_input:
        objects = coco_objects
        print(f"\nUsing default {len(objects)} COCO objects.")
    else:
        objects = [obj.strip() for obj in user_input.split(',')]
        print(f"\nUsing custom object(s): {objects}")

    # STEP 3: Convert all strings into embeddings
    print("\nEncoding tasks and objects into spatial vectors (embeddings)...")
    task_embeddings = model.encode(tasks)
    object_embeddings = model.encode(objects)
    
    print(f"  Task embeddings shape: {task_embeddings.shape}")
    print(f"  Object embeddings shape: {object_embeddings.shape}")

    # STEP 4: Compute Cosine Similarity
    print("Computing cosine similarities...")
    relevance_matrix = cosine_similarity(task_embeddings, object_embeddings)

    # STEP 5: Scale values properly between 0 and 1
    relevance_matrix_scaled = np.clip(relevance_matrix, 0.0, 1.0)

    # STEP 6: Print Table
    print("\n" + "="*80)
    print("RELEVANCE MATRIX (Tasks vs Objects)")
    print("="*80)
    
    # Header
    header = f"{'Task \\ Object':<25} | " + " | ".join([f"{obj[:10]:>10}" for obj in objects])
    print(header)
    print("-" * len(header))
    
    # Rows
    for t_idx, task in enumerate(tasks):
        row_str = f"{task[:25]:<25} | "
        scores = relevance_matrix_scaled[t_idx]
        row_str += " | ".join([f"{score:>10.4f}" for score in scores])
        print(row_str)

    # STEP 7: Find Best Task for Each Object
    print("\n" + "="*80)
    print("BEST TASK FOR EACH OBJECT")
    print("="*80)
    
    for o_idx, obj in enumerate(objects):
        object_scores = relevance_matrix_scaled[:, o_idx]
        best_task_idx = np.argmax(object_scores)
        best_task = tasks[best_task_idx]
        best_score = object_scores[best_task_idx]
        print(f"Object: {obj:<15} -> Best Task: {best_task:<25} (Score: {best_score:.4f})")

    # STEP 8: Convert matrix precision and save
    relevance_matrix_final = relevance_matrix_scaled.astype(np.float32)

    filename = "relevance.bin"
    print(f"\nSaving binary file to '{filename}' using np.tofile()...")
    relevance_matrix_final.tofile(filename)

    # STEP 9: Reload file and verify
    print("Reloading saved file to verify data integrity...")
    loaded_matrix = np.fromfile(filename, dtype=np.float32).reshape(len(tasks), len(objects))
    
    is_identical = np.allclose(relevance_matrix_final, loaded_matrix)
    print(f"  Success: Loaded matrix matches original perfectly -> {is_identical}")

if __name__ == "__main__":
    main()
