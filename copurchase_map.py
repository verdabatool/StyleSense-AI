import os
import pickle
from collections import defaultdict, Counter

import pandas as pd
from tqdm.auto import tqdm


ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "Data")
EMBEDDINGS_DIR = os.path.join(ROOT, "Embeddings")

TX_PATH = os.path.join(DATA_DIR, "transactions_train.csv")
ARTICLES_PATH = os.path.join(DATA_DIR, "articles.csv")

OUT_ARTICLE = os.path.join(EMBEDDINGS_DIR, "co_purchase_article.pkl")
OUT_TYPE = os.path.join(EMBEDDINGS_DIR, "co_purchase_type.pkl")


def build_copurchase(
    min_pair_count=1,
    top_k=20,
):
    print("Loading data...")
    tx = pd.read_csv(TX_PATH, usecols=["customer_id", "t_dat", "article_id"])
    articles = pd.read_csv(ARTICLES_PATH)

    tx["article_id"] = tx["article_id"].astype(str).str.zfill(10)
    articles["article_id"] = articles["article_id"].astype(str).str.zfill(10)

    id_to_type = dict(
        zip(articles["article_id"], articles["product_type_name"])
    )

    print("Building baskets (customer_id, date)...")
    baskets = (
        tx.groupby(["customer_id", "t_dat"])["article_id"]
        .apply(lambda x: list(set(x)))
        .tolist()
    )

    item_counts = Counter(tx["article_id"])
    article_co = defaultdict(Counter)
    type_co = defaultdict(Counter)

    print("Counting co-occurrences...")
    for basket in tqdm(baskets):
        # --- Article level ---
        for a in basket:
            for b in basket:
                if a != b:
                    article_co[a][b] += 1

        # --- Product type level ---
        types = list(set(id_to_type.get(a) for a in basket if a in id_to_type))
        for t in types:
            for u in types:
                if t != u:
                    type_co[t][u] += 1

    print("Scoring & pruning...")

    article_map = {}
    for a, ctr in article_co.items():
        scored = [
            (b, c / item_counts[a])
            for b, c in ctr.items()
            if c >= min_pair_count
        ]
        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            article_map[a] = [b for b, _ in scored[:top_k]]

    type_map = {}
    for t, ctr in type_co.items():
        scored = [(u, c) for u, c in ctr.items() if c >= min_pair_count]
        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            type_map[t] = [u for u, _ in scored[:top_k]]

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    with open(OUT_ARTICLE, "wb") as f:
        pickle.dump(article_map, f)

    with open(OUT_TYPE, "wb") as f:
        pickle.dump(type_map, f)

    print(f"Saved article co-purchase map: {len(article_map):,} items")
    print(f"Saved type co-purchase map: {len(type_map):,} types")


if __name__ == "__main__":
    build_copurchase()
