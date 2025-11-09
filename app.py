import streamlit as st
import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
import clip
import faiss
from PIL import Image

# Set environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 1. Configuration (Paths) ---
ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "Data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
ARTICLES_CSV = os.path.join(DATA_DIR, "articles.csv")

# --- Embeddings Paths ---
EMBEDDINGS_DIR = os.path.join(ROOT, "Embeddings")
emb_path = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy")
ids_path = os.path.join(EMBEDDINGS_DIR, "image_ids.pkl")

# --- 2. Page Asset URLs ---
# We just need the category names now
CATEGORIES = ["Dresses", "T-shirts", "Jackets", "Pants", "Shoes", "Accessories"]


# --- 3. Helper Functions (Unchanged) ---

def image_path(article_id):
    """Helper function to get the full path of an article image."""
    pfx = str(article_id)[:3]
    return os.path.join(IMAGE_DIR, pfx, f"{article_id}.jpg")


def find_items_by_text(query, faiss_index, ids, clip_model, clip_tokenizer, device, top_k=5):
    """Find similar items based on a text query."""
    text_input = clip_tokenizer([query]).to(device)

    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_input)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

    query_embedding = text_embedding.cpu().numpy().astype('float32')

    distances, indices = faiss_index.search(query_embedding, top_k)
    matched_ids = [ids[i] for i in indices[0]]

    return matched_ids


def show_article_details(article_ids_list, articles_df):
    """
    Takes a list of article_ids and the main articles DataFrame.
    Returns a formatted DataFrame with key details, preserving the order.
    """
    if not article_ids_list:
        return pd.DataFrame()

    result_df = articles_df[articles_df["article_id"].isin(article_ids_list)]

    id_map = {id: pos for pos, id in enumerate(article_ids_list)}
    result_df = result_df.sort_values(by="article_id", key=lambda x: x.map(id_map))

    return result_df


# --- 4. Cached Asset Loading (Unchanged) ---

@st.cache_resource
def load_search_assets():
    """
    Loads all necessary assets for the search app.
    This function is cached by Streamlit, so it only runs once.
    """
    articles = pd.read_csv(ARTICLES_CSV)
    articles["article_id"] = articles["article_id"].astype(str).str.zfill(10)

    if not os.path.exists(emb_path) or not os.path.exists(ids_path):
        st.error(
            f"Embedding files not found! Please run your original script once to generate {emb_path} and {ids_path}.")
        st.stop()

    embeddings = np.load(emb_path)
    with open(ids_path, "rb") as f:
        image_ids = pickle.load(f)

    embeddings_faiss = np.ascontiguousarray(embeddings, dtype='float32')
    index = faiss.IndexFlatIP(embeddings_faiss.shape[1])
    index.add(embeddings_faiss)

    device_text = "cpu"
    model_text, _ = clip.load("ViT-B/32", device=device_text)
    model_text.eval()

    return articles, index, image_ids, model_text, clip.tokenize


# --- 5. UI Display Function (Unchanged) ---

def display_results_grid(df, num_columns=4):
    """Helper function to display items in a styled grid."""

    cols = st.columns(num_columns)

    for i, (idx, row) in enumerate(df.iterrows()):
        col = cols[i % num_columns]

        with col.container(border=True):
            article_id = row["article_id"]
            img_path = image_path(article_id)

            if os.path.exists(img_path):
                st.image(img_path, width=200)
            else:
                st.warning(f"Image not found:\n{article_id}")

            st.markdown(f"<p style='text-align: center; font-weight: bold;'>{row['prod_name']}</p>",
                        unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'><b>Article ID:</b> {article_id}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'><b>Type:</b> {row['product_type_name']}</p>",
                        unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'><b>Color:</b> {row['colour_group_name']}</p>",
                        unsafe_allow_html=True)


# --- 6. Streamlit App UI ---

st.set_page_config(layout="wide")

# --- NEW: Custom CSS for Fashion Retail Look (MODIFIED) ---
st.markdown("""
    <style>
    /* 1. Import Google Font (Added weight 900) */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700;900&display=swap');

    /* 2. Apply Font and Base Colors */
    body, html {
        font-family: 'Montserrat', sans-serif;
        background-color: #FFFFFF; /* Clean white background */
        color: #333333; /* Dark grey text */
    }

    /* 3. Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* 4. Style Titles (Will be targeted by markdown h1/h3) */
    h1 { font-weight: 700; color: #111111; }
    h3 { font-weight: 400; color: #555555; }

    /* 5. Style Product Cards (Unchanged) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px;       
        padding: 16px;            
        background-color: #FAFAFA; 
        box-shadow: 0 4px 12px 0 rgba(0,0,0,0.05); 
        transition: 0.3s;
        border: 1px solid #EEEEEE; 
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.1);
    }

    /* 6. Style Search Button (Unchanged) */
    [data-testid="stFormSubmitButton"] button {
        background-color: #222222;
        color: #FFFFFF;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        transition: 0.3s;
        width: 100%; 
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background-color: #555555;
        color: #FFFFFF;
    }

    /* 7. Style Subheaders (Unchanged) */
    [data-testid="stSubheader"] {
        font-weight: 700;
        color: #111111;
        padding-top: 20px;
    }

    /* 8. --- REMOVED .circular-image class --- */

    /* 9. --- MODIFIED: Style for Category Buttons (to BE the circles) --- */
    .stButton > button {
        /* Shape and Size */
        width: 150px;
        height: 150px;
        border-radius: 50%;

        /* Color and Border */
        background-color: #F5F5F5;
        border: 2px solid #EEEEEE;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);

        /* Text Styling */
        color: #333333;
        font-family: 'Montserrat', sans-serif;
        font-weight: 900 !important; /* <-- MODIFIED: Added !important to force bold */
        font-size: 18px;

        /* Centering Text */
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;

        /* Other */
        transition: 0.3s;
        padding: 0;
        margin: auto;
        line-height: 1.2;
    }
    .stButton > button:hover {
        background-color: #EEEEEE;
        color: #E1005C;
        border-color: #DDD;
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.1);
        transform: scale(1.05);
    }
    .stButton > button:focus {
        background-color: #F5F5F5;
        color: #333333;
        border-color: #EEEEEE;
    }

    /* 10. --- MODIFIED: Style for Clear Search Button (to override circle) --- */
    .clear-button button {
        /* --- Override circle properties --- */
        width: auto;
        height: auto;
        border-radius: 5px; /* Back to a rectangle */
        padding: 8px 15px;  /* Normal button padding */

        /* --- Normal button styling --- */
        background-color: #FAFAFA;
        color: #555;
        border: 1px solid #DDD;
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        font-size: 14px;
    }
    /* --- Override hover effects for clear button --- */
    .clear-button button:hover {
        transform: none;
        box-shadow: none;
        background-color: #EEEEEE;
        color: #111;
        border-color: #CCC;
    }

    /* Ensure category button container is centered */
    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div {
        display: flex;
        justify-content: center;
    }

    </style>
    """, unsafe_allow_html=True)

# --- 7. Initialize Session State ---
if 'last_search' not in st.session_state:
    st.session_state.last_search = None

# --- 8. App Header and Search Bar ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center;'>🛍️ StyleSense AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Where Style meets Elegance and Innovation</h3>",
                unsafe_allow_html=True)
    st.markdown("---")

    with st.form(key="search_form"):
        query = st.text_input(
            "**What are you looking for?**",
            placeholder="search here"
        )
        submit_button = st.form_submit_button(label="Search")

        if submit_button and query:
            st.session_state.last_search = query
        elif submit_button and not query:
            st.warning("Please enter a search query.")

# --- 9. Load Assets (Silently) ---
try:
    articles, index, image_ids, model_text, tokenizer = load_search_assets()
except Exception as e:
    st.error(f"An error occurred while loading assets: {e}")
    st.stop()

# --- Removed Static Banner as requested ---


# --- 10. NEW: Category Bubbles (MODIFIED) ---
st.subheader("Shop by Category")
cat_cols = st.columns(len(CATEGORIES))

for i, category in enumerate(CATEGORIES):
    with cat_cols[i]:
        # --- MODIFIED: Removed image markdown, the button is now the circle ---
        if st.button(category, key=f"cat_{category}"):
            st.session_state.last_search = category
            st.rerun()

# --- 11. Main Display Logic (Modified) ---
st.markdown("---")

if st.session_state.last_search:
    st.subheader(f"Showing results for: '{st.session_state.last_search}'")

    col1_clear, col2_clear, col3_clear = st.columns([1, 2, 1])
    with col1_clear:
        st.markdown('<div class="clear-button">', unsafe_allow_html=True)
        if st.button("Clear Search / Show Featured"):
            st.session_state.last_search = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    start_time = time.time()
    text_search_ids = find_items_by_text(
        st.session_state.last_search,
        faiss_index=index,
        ids=image_ids,
        clip_model=model_text,
        clip_tokenizer=tokenizer,
        device="cpu",
        top_k=12
    )
    print(f"Query completed in {time.time() - start_time:.4f} s")

    results_df = show_article_details(text_search_ids, articles)

    if results_df.empty:
        st.warning("No results found for your query.")
    else:
        display_results_grid(results_df)

else:
    st.subheader("Featured Items")
    featured_df = articles.tail(12)
    display_results_grid(featured_df)