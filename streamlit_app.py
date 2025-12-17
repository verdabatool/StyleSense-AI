import os
import pickle
import streamlit as st
import pandas as pd
from PIL import Image

from multimodal_retrieval import (
    MultiModalRetriever,
    default_paths,
    image_path_from_article_id,
)

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Multimodal Search",
    layout="wide",
)

st.title("🛒 StyleSense AI")
st.caption("Where Fashion meets Innovation")

# --------------------------------------------------
# Load retriever & data (cached)
# --------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_retriever():
    paths = default_paths()
    retriever = MultiModalRetriever(paths)
    retriever.setup(build_embeddings_if_missing=True)
    return retriever


@st.cache_resource(show_spinner=False)
def load_copurchase_maps():
    base = os.path.join(os.getcwd(), "Embeddings")
    article_path = os.path.join(base, "co_purchase_article.pkl")
    type_path = os.path.join(base, "co_purchase_type.pkl")

    article_map, type_map = {}, {}

    if os.path.exists(article_path):
        with open(article_path, "rb") as f:
            article_map = pickle.load(f)

    if os.path.exists(type_path):
        with open(type_path, "rb") as f:
            type_map = pickle.load(f)

    return article_map, type_map


retriever = load_retriever()
article_co, type_co = load_copurchase_maps()
articles_df = retriever.articles_df

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
mode = st.sidebar.radio(
    "Search mode",
    ["Text search", "Product similarity", "Image upload"],
)

top_k = st.sidebar.slider("Number of results", 3, 20, 6)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def render_results(ids, scores):
    cols = st.columns(3)
    for i, (aid, score) in enumerate(zip(ids, scores)):
        with cols[i % 3]:
            img_path = image_path_from_article_id(
                retriever.paths.image_dir, aid
            )
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)

            row = articles_df[articles_df["article_id"] == aid]
            if not row.empty:
                r = row.iloc[0]
                st.markdown(f"**{r.get('prod_name','')}**")
                st.caption(
                    f"{r.get('product_type_name','')} • score={score:.3f}"
                )


def render_copurchase_grid(anchor_id, max_items=6):
    st.subheader("🧠 Frequently bought together")

    related_ids = []

    if anchor_id in article_co:
        related_ids = article_co[anchor_id][:max_items]

    if not related_ids:
        row = articles_df[articles_df["article_id"] == anchor_id]
        if not row.empty:
            ptype = row.iloc[0].get("product_type_name")
            if ptype in type_co:
                related_ids = (
                    articles_df[
                        articles_df["product_type_name"].isin(type_co[ptype][:3])
                    ]
                    .sample(
                        min(max_items, len(articles_df)),
                        replace=False,
                        random_state=42,
                    )["article_id"]
                    .tolist()
                )

    if not related_ids:
        st.info("No co-purchase data available for this item.")
        return

    cols = st.columns(3)
    for i, aid in enumerate(related_ids):
        with cols[i % 3]:
            img_path = image_path_from_article_id(
                retriever.paths.image_dir, aid
            )
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)

            row = articles_df[articles_df["article_id"] == aid]
            if not row.empty:
                r = row.iloc[0]
                st.markdown(f"**{r.get('prod_name','')}**")
                st.caption(r.get("product_type_name",""))


# --------------------------------------------------
# Main UI
# --------------------------------------------------
if mode == "Text search":
    query = st.text_input(
        "Describe what you want",
        placeholder="black leather jacket",
    )

    if query:
        with st.spinner("Searching…"):
            out = retriever.search_by_text(query, top_k=top_k)

        if not out["ids"]:
            st.warning("No results found.")
        else:
            st.subheader("Search results")
            render_results(out["ids"], out["scores"])

            render_copurchase_grid(out["ids"][0])


elif mode == "Product similarity":
    aid = st.text_input(
        "Article ID",
        placeholder="0108775015",
    )

    if aid:
        try:
            with st.spinner("Finding similar products…"):
                out = retriever.search_by_article_id(aid, top_k=top_k)

            st.subheader("Selected product")
            img = image_path_from_article_id(
                retriever.paths.image_dir, aid
            )
            if os.path.exists(img):
                st.image(img, width=260)

            st.subheader("Visually similar products")
            render_results(out["ids"], out["scores"])

            render_copurchase_grid(aid)

        except Exception as e:
            st.error(str(e))


elif mode == "Image upload":
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "png", "jpeg"],
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=300)

        with st.spinner("Matching against catalog…"):
            out = retriever.search_by_image_path(
                image_path=uploaded,
                top_k=top_k,
            )

        if not out["ids"]:
            st.warning("No matches found.")
        else:
            st.subheader("Closest matches")
            render_results(out["ids"], out["scores"])

            render_copurchase_grid(out["ids"][0])

