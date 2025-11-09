import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
import clip
import faiss
from tqdm.auto import tqdm
from PIL import Image

# Set environment variable to prevent MKL-related hangs (often on macOS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 1. Configuration ---
print("Setting up configuration...")

# --- Devices ---
# Use MPS (Apple Silicon GPU) if available for image processing, else CPU
device_image = "mps" if torch.backends.mps.is_available() else "cpu"
# Text processing is fast, so CPU is fine
device_text = "cpu"

# --- Paths ---
ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "Data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
ARTICLES_CSV = os.path.join(DATA_DIR, "articles.csv")

# --- Embeddings Paths ---
EMBEDDINGS_DIR = os.path.join(ROOT, "Embeddings")
emb_path = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy")
ids_path = os.path.join(EMBEDDINGS_DIR, "image_ids.pkl")


# --- 2. Helper Functions ---

def image_path(article_id):
    """Helper function to get the full path of an article image."""
    pfx = str(article_id)[:3]
    return os.path.join(IMAGE_DIR, pfx, f"{article_id}.jpg")


def get_similar_items(article_id, faiss_index, embeddings, ids, top_k=10):
    """Find similar items based on image similarity to a given article_id."""
    try:
        # Find the index position of the query article_id
        idx = ids.index(article_id)
    except ValueError:
        print(f"Error: Article {article_id} not found in embeddings index.")
        return []

    query_embedding = embeddings[idx].reshape(1, -1).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_k + 1)

    # [0][1:] skips the first result, which is the query item itself
    similar_article_ids = [ids[i] for i in indices[0][1:]]
    return similar_article_ids


def find_items_by_text(query, faiss_index, ids, clip_model, clip_tokenizer, device, top_k=5):
    """Find similar items based on a text query."""
    start_total = time.time()
    text_input = clip_tokenizer([query]).to(device)

    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_input)
        # Normalize the text embedding to match image embeddings
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

    query_embedding = text_embedding.cpu().numpy().astype('float32')

    # Search the FAISS index
    distances, indices = faiss_index.search(query_embedding, top_k)
    matched_ids = [ids[i] for i in indices[0]]

    print(f"Text query completed in {time.time() - start_total:.4f} s")
    return matched_ids


def show_article_details(article_ids_list, articles_df):
    """
    Takes a list of article_ids and the main articles DataFrame.
    Returns a formatted DataFrame with key details, preserving the order.
    """
    if not article_ids_list:
        return pd.DataFrame()

    # Filter the DataFrame to only include the relevant IDs
    result_df = articles_df[articles_df["article_id"].isin(article_ids_list)]

    # Preserve the order from the search results
    id_map = {id: pos for pos, id in enumerate(article_ids_list)}
    result_df = result_df.sort_values(by="article_id", key=lambda x: x.map(id_map))

    return result_df[[
        "article_id", "prod_name", "product_type_name",
        "graphical_appearance_name", "colour_group_name"
    ]]


# --- 3. Main Execution ---
def main():
    """Main script execution."""
    print("Starting script...")
    start_time = time.time()

    # --- Check/Generate Embeddings ---
    print(f"Using device for image embeddings: {device_image}")

    if os.path.exists(emb_path) and os.path.exists(ids_path):
        print("Embeddings already exist. Skipping generation.")
    else:
        print("No saved embeddings found. Computing now...")

        # --- 3.1. Load Articles for generation ---
        articles_gen_df = pd.read_csv(ARTICLES_CSV)
        articles_gen_df["article_id"] = articles_gen_df["article_id"].astype(str).str.zfill(10)

        # --- 3.2. THE SLOW I/O CHECK ---
        print("Checking which articles have images on disk...")
        valid_ids = [aid for aid in tqdm(articles_gen_df["article_id"].tolist()) if os.path.exists(image_path(aid))]
        print(f"Images found for {len(valid_ids)} article IDs.")

        # --- 3.3. THE SLOW MODEL LOAD ---
        print(f"Loading CLIP model 'ViT-B/32' to {device_image}...")
        model_image, preprocess = clip.load("ViT-B/32", device=device_image)

        # --- 3.4. Generate Embeddings ---
        BATCH_SIZE = 32
        embedding_list = []
        image_ids_gen = []

        for i in tqdm(range(0, len(valid_ids), BATCH_SIZE), desc="Embedding images"):
            batch_ids = valid_ids[i:i + BATCH_SIZE]
            batch_imgs = []

            for aid in batch_ids:
                try:
                    img_path = image_path(aid)
                    img = Image.open(img_path).convert("RGB")
                    batch_imgs.append(preprocess(img))
                    image_ids_gen.append(aid)
                except Exception:
                    # Skip corrupted images or other read errors
                    continue

            if not batch_imgs:
                continue

            batch_tensor = torch.stack(batch_imgs).to(device_image)

            with torch.no_grad():
                feats = model_image.encode_image(batch_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)  # Normalized here
                embedding_list.append(feats.cpu())

        embeddings_gen = torch.cat(embedding_list, dim=0).numpy()

        # --- 3.5. Save ---
        # Ensure the directory exists before saving
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

        np.save(emb_path, embeddings_gen)
        with open(ids_path, "wb") as f:
            pickle.dump(image_ids_gen, f)

        print(f"\nSaved {embeddings_gen.shape[0]} embeddings to {emb_path}")

        # Clean up memory
        del model_image, preprocess, embeddings_gen, image_ids_gen, articles_gen_df
        if 'torch' in locals():
            torch.cuda.empty_cache()  # if using cuda
            if device_image == "mps":
                torch.mps.empty_cache()

    # --- 4. Load Data for Querying ---

    # --- 4.1. Load Articles CSV ---
    print("\nLoading articles.csv...")
    articles = pd.read_csv(ARTICLES_CSV)
    articles["article_id"] = articles["article_id"].astype(str).str.zfill(10)
    print(f"Loaded articles in {time.time() - start_time:.2f}s")

    # --- 4.2. Load Pre-computed Embeddings ---
    print("Loading embeddings and IDs...")
    embeddings = np.load(emb_path)
    with open(ids_path, "rb") as f:
        image_ids = pickle.load(f)
    print(f"Loaded embeddings in {time.time() - start_time:.2f}s")

    # --- 4.3. Build FAISS Index ---
    print("Preparing FAISS index...")
    # Ensure data is contiguous and in float32 format for FAISS
    embeddings_faiss = np.ascontiguousarray(embeddings, dtype='float32')

    # Using IndexFlatIP because embeddings are normalized (dot product = cosine similarity)
    index = faiss.IndexFlatIP(embeddings_faiss.shape[1])
    index.add(embeddings_faiss)
    print(f"Index built with {index.ntotal} vectors in {time.time() - start_time:.2f}s")

    # --- 4.4. Load TEXT Model (to CPU) ---
    print(f"Loading TEXT model to {device_text}...")
    model_text, _ = clip.load("ViT-B/32", device=device_text)
    model_text.eval()
    print(f"Setup complete in {time.time() - start_time:.2f}s. Ready to query.")

    # --- 5. RUN QUERY ---
    QUERY = "a red summer dress with flowers"
    print(f"\n--- Running query: '{QUERY}' ---")

    text_search_ids = find_items_by_text(
        QUERY,
        faiss_index=index,
        ids=image_ids,
        clip_model=model_text,
        clip_tokenizer=clip.tokenize,
        device=device_text,
        top_k=5
    )

    # --- 6. Show Results ---
    results_df = show_article_details(text_search_ids, articles)

    print("\n--- Query Results ---")
    # Use print() instead of display() for .py files
    print(results_df)

    print(f"\nScript finished in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()