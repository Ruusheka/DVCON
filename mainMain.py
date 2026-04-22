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

    # STEP 6: Validation checks & Output
    print("\n--- OUTPUT & VALIDATION ---")
    print(f"Final Matrix Shape: {relevance_matrix_scaled.shape} -> must be ({len(tasks)}, {len(objects)})")
    
    if not user_input:
        # Default verification for 80 objects
        drink_index = tasks.index("drink")
        drink_similarities = relevance_matrix_scaled[drink_index]
        
        top_5_indices = np.argsort(drink_similarities)[::-1][:5]
        top_5_objects = [objects[idx] for idx in top_5_indices]
        
        print(f"\nTask examined: 'drink'")
        print(f"Top 5 most relevant objects:")
        for obj in top_5_objects:
            score = drink_similarities[objects.index(obj)]
            print(f"  ✓ {obj}: {score:.4f}")
    else:
        # Custom object verification - print relevance for all tasks
        print("\nRelevance Matrix Output (Scores 0 to 1):")
        for t_idx, task in enumerate(tasks):
            scores = relevance_matrix_scaled[t_idx]
            scores_str = " | ".join([f"{obj}: {score:.4f}" for obj, score in zip(objects, scores)])
            print(f"  Task '{task:15s}': {scores_str}")
        print("-------------------------\n")

    # STEP 7: Convert matrix precision
    relevance_matrix_final = relevance_matrix_scaled.astype(np.float32)

    # STEP 8: Save to text file
    filename = "relevance.txt"
    print(f"\nSaving text file to '{filename}' using np.savetxt()...")
    np.savetxt(filename, relevance_matrix_final, fmt='%.6f')

    # STEP 9: Reload file and verify
    print("Reloading saved file to verify data integrity...")
    loaded_matrix = np.loadtxt(filename, dtype=np.float32).reshape(len(tasks), len(objects))
    
    is_identical = np.allclose(relevance_matrix_final, loaded_matrix)
    
    print(f"  Matrix re-shape successful: {loaded_matrix.shape}")
    if not user_input:
        sample_val = loaded_matrix[tasks.index('drink')][objects.index('cup')]
        print(f"  Sample value matches (drink->cup): {sample_val:.4f}")
    else:
        sample_val = loaded_matrix[0][0]
        print(f"  Sample value matches (first task -> first object): {sample_val:.4f}")
        
    print(f"  Success: Loaded matrix matches original perfectly -> {is_identical}")

if __name__ == "__main__":
    main()
