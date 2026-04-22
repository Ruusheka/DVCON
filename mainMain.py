"""
Task-Object Relevance Matrix Generator using Sentence-BERT
----------------------------------------------------------
This script dynamically generates a matrix representing the semantic relevance 
between predefined human tasks and 80 COCO dataset objects.
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

    objects = [
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

    print("\n" + "="*60)
    print("Task-Object Relevance Matrix Generator (14x80)")
    print("="*60)

    # STEP 2: Convert all strings into embeddings
    print("\nEncoding tasks and objects into spatial vectors (embeddings)...")
    task_embeddings = model.encode(tasks)
    object_embeddings = model.encode(objects)
    
    print(f"  Task embeddings shape: {task_embeddings.shape}")
    print(f"  Object embeddings shape: {object_embeddings.shape}")

    # STEP 3: Compute Cosine Similarity
    print("Computing cosine similarities...")
    relevance_matrix = cosine_similarity(task_embeddings, object_embeddings)

    # STEP 4: Scale values properly between 0 and 1
    relevance_matrix_scaled = np.clip(relevance_matrix, 0.0, 1.0)

    # STEP 5: Print Table
    print("\n" + "="*80)
    print("RELEVANCE MATRIX (14 Tasks vs 80 Objects)")
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

    # STEP 6: Find Best Task for Each Object
    print("\n" + "="*80)
    print("BEST TASK FOR EACH OBJECT")
    print("="*80)
    
    for o_idx, obj in enumerate(objects):
        object_scores = relevance_matrix_scaled[:, o_idx]
        best_task_idx = np.argmax(object_scores)
        best_task = tasks[best_task_idx]
        best_score = object_scores[best_task_idx]
        print(f"Object: {obj:<15} -> Best Task: {best_task:<25} (Score: {best_score:.4f})")

    # STEP 7: Find the Absolute Highest Value in the 14x80 Matrix
    print("\n" + "="*80)
    print("OVERALL HIGHEST RELEVANT VALUE IN ENTIRE MATRIX")
    print("="*80)
    
    # Find the indices of the maximum value
    max_idx = np.unravel_index(np.argmax(relevance_matrix_scaled, axis=None), relevance_matrix_scaled.shape)
    best_overall_task = tasks[max_idx[0]]
    best_overall_obj = objects[max_idx[1]]
    best_overall_val = relevance_matrix_scaled[max_idx]
    
    print(f"The highest relevance score is {best_overall_val:.4f}")
    print(f"Task   : {best_overall_task}")
    print(f"Object : {best_overall_obj}")

    # STEP 8: Convert matrix precision and save
    relevance_matrix_final = relevance_matrix_scaled.astype(np.float32)

    filename = "relevance.txt"
    print(f"\nSaving text file to '{filename}'...")
    with open(filename, "w") as f:
        for t_idx, task in enumerate(tasks):
            scores_str = "\t".join([f"{score:.6f}" for score in relevance_matrix_final[t_idx]])
            f.write(f"{task}\t{scores_str}\n")

    # STEP 9: Reload file and verify
    print("Reloading saved file to verify data integrity...")
    loaded_matrix = []
    loaded_tasks = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip("\n").split("\t")
            if len(parts) > 1:
                loaded_tasks.append(parts[0])
                loaded_matrix.append([float(x) for x in parts[1:]])
    
    loaded_matrix = np.array(loaded_matrix, dtype=np.float32)
    
    is_identical = np.allclose(relevance_matrix_final, loaded_matrix, atol=1e-5)
    tasks_match = tasks == loaded_tasks
    print(f"  Success: Loaded data matches original perfectly -> {is_identical and tasks_match}")

if __name__ == "__main__":
    main()
