# Task-Object Semantic Relevance Matrix

This project dynamically generates a semantic relevance matrix between predefined human tasks (e.g., "drink", "use computer") and objects (like COCO dataset objects). It uses AI to build a logical map of "what objects are used for what tasks" without requiring manual hard-coding of every single rule.

## How It Works

The core logic relies on **Semantic Embedding** using AI:

1. **Translating Words to Math (Embeddings):** 
   We pass human tasks and objects into the `Sentence-BERT` model (`all-MiniLM-L6-v2`). This AI has learned human language by reading millions of articles, books, and internet pages. It converts each word into a **384-dimensional mathematical vector** (a coordinate on a giant 384D map of concepts). 

2. **Comparing Meanings (Cosine Similarity):** 
   Words that are frequently used together or share similar contexts end up physically closer to each other on this map. To see how related a task is to an object, we calculate the **Cosine Similarity**—the mathematical angle between the two vectors. 

3. **The Matrix:** 
   A score near `1.0` means the items are highly related (small angle), while a score near `0.0` means they are completely unrelated. The script generates a matrix (grid) of all these overlapping scores and saves it as a binary file (`relevance.bin`) for use in other applications.

---

## The "Apple" Anomaly: AI Language Bias

When running the matrix, you might notice an interesting anomaly: the object `"apple"` scores unusually high (e.g., ~0.41) in relevance to the task `"use computer"`, sometimes scoring even higher than `"eat"` or `"cut food"`.

**Why does this happen?**

This is a classic example of **AI Language Bias** and **Polysemy** (words having multiple meanings). 

Sentence-BERT doesn't have physical eyes; it only knows the world through text from the internet. Think about how the word "apple" is used online:
* While it is a fruit, a massive percentage of internet text uses the word **Apple** to refer to **Apple Inc.** (MacBooks, iPhones, Apple Watches, iPads).
* Because Sentence-BERT was trained on news articles, Wikipedia, and tech blogs, the mathematical vector for `"apple"` is heavily "pulled" toward the concept of **technology, electronics, and computing**.
* Therefore, when it compares the vector for `"apple"` with the vector for `"use computer"`, the math assumes *"These two concepts show up in the same context all the time!"* and awards a high similarity score.

### Output Screenshot
![Output](SentenceBert.png)

### How to Fix This
If you want the AI to strictly think of the physical fruit and ignore the tech company, you must provide more context so the AI plots the vector differently. Instead of passing just `"apple"` to the model, passing:
`"an apple fruit"` or `"a fresh apple for eating"`
...will instantly shift the vector away from "computers" and move it toward "food", completely correcting the scores.

## Installation & Usage

1. Install the required dependencies:
   ```bash
   pip install sentence-transformers scikit-learn numpy
   ```

2. Run the main generator script:
   ```bash
   python mainMain.py
   ```
   *You can either input a custom comma-separated list of objects to generate a dynamic matrix, or press Enter to fall back to the default 80 COCO objects.*
