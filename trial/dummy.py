"""
Task-Object Relevance Matrix Generator using Sentence-BERT
----------------------------------------------------------
This script generates a 14x80 matrix representing the semantic relevance 
between human tasks and COCO dataset objects.
"""

# STEP 1: Install and import required libraries
# (To run this, first install: pip install sentence-transformers scikit-learn numpy)
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    # STEP 2: Load pretrained Sentence-BERT model
    # 'all-MiniLM-L6-v2' is incredibly fast, small, and creates high-quality 384-dimensional embeddings
    print("Loading pretrained Sentence-BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # DATA DEFINITIONS
    tasks = [
        "drink", "cut food", "take photo", "read", "ride", "play sport", 
        "use computer", "eat", "groom", "sit", "cook", "repair", 
        "talk on phone", "watch TV"
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

    # STEP 3: Convert all strings into embeddings
    print("Encoding tasks and objects into spatial vectors (embeddings)...")
    task_embeddings = model.encode(tasks)
    object_embeddings = model.encode(objects)
    
    # Print shapes to verify sizes: (Number of items, Embedding dimension size)
    print(f"  Task embeddings shape: {task_embeddings.shape}")      # Expected: (14, 384)
    print(f"  Object embeddings shape: {object_embeddings.shape}")  # Expected: (80, 384)

    # STEP 4: Compute Cosine Similarity
    # This computes the dot product of normalized vectors, yielding a 14x80 matrix
    print("Computing cosine similarities...")
    relevance_matrix = cosine_similarity(task_embeddings, object_embeddings)

    # STEP 5: Scale values properly between 0 and 1
    # Cosine similarity yields values from -1 to 1. Negative values imply opposite linguistic context.
    # In relevance terms, anything below 0 is "completely irrelevant". 
    # Therefore, we safely clip negative values to 0.0.
    relevance_matrix_scaled = np.clip(relevance_matrix, 0.0, 1.0)

    # STEP 6: Validation checks (CRITICAL)
    print("\n--- VALIDATION CHECKS ---")
    print(f"Final Matrix Shape: {relevance_matrix_scaled.shape} -> must be (14, 80)")
    
    drink_index = tasks.index("drink")
    drink_similarities = relevance_matrix_scaled[drink_index]
    
    # Get indices of the top 5 highest scores in the "drink" row
    top_5_indices = np.argsort(drink_similarities)[::-1][:5]
    top_5_objects = [objects[idx] for idx in top_5_indices]
    
    print(f"\nTask examined: 'drink'")
    print(f"Top 5 most relevant objects:")
    for obj in top_5_objects:
        score = drink_similarities[objects.index(obj)]
        print(f"  ✓ {obj}: {score:.4f}")
        
    print("Ensuring logical sense (comparing vs unrelated objects):")
    print(f"  ✗ airplane: {drink_similarities[objects.index('airplane')]:.4f} (Expected low)")
    print(f"  ✗ chair: {drink_similarities[objects.index('chair')]:.4f} (Expected low)")
    print("-------------------------\n")

    # STEP 7: Convert matrix precision
    # Floating point 32 is widely standard for ML inference memory footprints.
    relevance_matrix_final = relevance_matrix_scaled.astype(np.float32)

    # STEP 8: Save to binary file
    filename = "relevance.bin"
    print(f"Saving binary file to '{filename}' using np.tofile()...")
    relevance_matrix_final.tofile(filename)

    # STEP 9: Reload file and verify
    print("Reloading saved file to verify data integrity...")
    # Because it is a raw binary file, we must tell NumPy the dtype and reshape it manually.
    loaded_matrix = np.fromfile(filename, dtype=np.float32).reshape(14, 80)
    
    is_identical = np.allclose(relevance_matrix_final, loaded_matrix)
    sample_val = loaded_matrix[drink_index][objects.index('cup')]
    
    print(f"  Matrix re-shape successful: {loaded_matrix.shape}")
    print(f"  Sample value matches (drink->cup): {sample_val:.4f}")
    print(f"  Success: Loaded matrix matches original perfectly -> {is_identical}")

if __name__ == "__main__":
    main()
