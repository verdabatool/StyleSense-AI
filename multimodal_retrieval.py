import os
import time
import pickle
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import clip
import faiss
from tqdm.auto import tqdm
from PIL import Image

# Prevent MKL-related hangs (often on macOS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ----------------------------
# Configuration / Paths
# ----------------------------

@dataclass
class RepoPaths:
    root: str
    data_dir: str
    image_dir: str
    articles_csv: str
    embeddings_dir: str
    emb_path: str
    ids_path: str
    faiss_path: str  # optional cached index


def default_paths(root: Optional[str] = None) -> RepoPaths:
    root = root or os.getcwd()
    data_dir = os.path.join(root, "Data")
    image_dir = os.path.join(data_dir, "images")
    articles_csv = os.path.join(data_dir, "articles.csv")

    embeddings_dir = os.path.join(root, "Embeddings")
    emb_path = os.path.join(embeddings_dir, "image_embeddings.npy")
    ids_path = os.path.join(embeddings_dir, "image_ids.pkl")
    faiss_path = os.path.join(embeddings_dir, "faiss.index")

    return RepoPaths(
        root=root,
        data_dir=data_dir,
        image_dir=image_dir,
        articles_csv=articles_csv,
        embeddings_dir=embeddings_dir,
        emb_path=emb_path,
        ids_path=ids_path,
        faiss_path=faiss_path,
    )


def detect_device(prefer_mps: bool = True) -> str:
    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def image_path_from_article_id(image_dir: str, article_id: str) -> str:
    """
    Your dataset stores images as:
      Data/images/<first 3 digits>/<article_id>.jpg
    """
    aid = str(article_id).zfill(10)
    pfx = aid[:3]
    return os.path.join(image_dir, pfx, f"{aid}.jpg")


# ----------------------------
# Core Retriever
# ----------------------------

class MultiModalRetriever:
    """
    Multimodal retrieval over product images using CLIP + FAISS.

    Supports:
      - Text -> image retrieval
      - Catalog image (by article_id) -> image retrieval
      - External image file path -> image retrieval
    """

    def __init__(
        self,
        paths: RepoPaths,
        clip_model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        top_k_default: int = 5,
    ):
        self.paths = paths
        self.clip_model_name = clip_model_name
        self.device = device or detect_device(prefer_mps=True)
        self.top_k_default = top_k_default

        self.articles_df: Optional[pd.DataFrame] = None
        self.ids: Optional[List[str]] = None
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None

        self.model: Optional[torch.nn.Module] = None
        self.preprocess = None

    # -------- Loading / Building --------

    def load_articles(self) -> pd.DataFrame:
        df = pd.read_csv(self.paths.articles_csv)
        df["article_id"] = df["article_id"].astype(str).str.zfill(10)
        self.articles_df = df
        return df

    def ensure_embeddings(self, batch_size: int = 32) -> None:
        """
        If Embeddings/image_embeddings.npy and Embeddings/image_ids.pkl exist, reuse them.
        Otherwise compute CLIP image embeddings for all images on disk and save.
        """
        if os.path.exists(self.paths.emb_path) and os.path.exists(self.paths.ids_path):
            return

        os.makedirs(self.paths.embeddings_dir, exist_ok=True)

        if self.articles_df is None:
            self.load_articles()

        # 1) Identify which articles actually have images on disk
        article_ids = self.articles_df["article_id"].tolist()
        valid_ids = []
        for aid in tqdm(article_ids, desc="Checking images exist"):
            if os.path.exists(image_path_from_article_id(self.paths.image_dir, aid)):
                valid_ids.append(aid)

        if not valid_ids:
            raise RuntimeError("No valid images found on disk. Check Data/images structure.")

        # 2) Load CLIP model for image encoding
        model, preprocess = clip.load(self.clip_model_name, device=self.device)
        model.eval()

        # 3) Batch encode
        embedding_chunks = []
        kept_ids = []

        for i in tqdm(range(0, len(valid_ids), batch_size), desc="Embedding images"):
            batch_ids = valid_ids[i : i + batch_size]
            batch_tensors = []

            for aid in batch_ids:
                p = image_path_from_article_id(self.paths.image_dir, aid)
                try:
                    img = Image.open(p).convert("RGB")
                    batch_tensors.append(preprocess(img))
                    kept_ids.append(aid)
                except Exception:
                    # corrupted/missing -> skip
                    continue

            if not batch_tensors:
                continue

            batch = torch.stack(batch_tensors).to(self.device)
            with torch.no_grad():
                feats = model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)  # normalize

            embedding_chunks.append(feats.cpu())

        if not embedding_chunks:
            raise RuntimeError("Failed to create embeddings (no images successfully processed).")

        embeddings = torch.cat(embedding_chunks, dim=0).numpy().astype("float32")

        # 4) Save to disk
        np.save(self.paths.emb_path, embeddings)
        with open(self.paths.ids_path, "wb") as f:
            pickle.dump(kept_ids, f)

        # Cleanup
        del model
        if self.device == "mps":
            torch.mps.empty_cache()

    def load_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        self.embeddings = np.load(self.paths.emb_path).astype("float32", copy=False)
        with open(self.paths.ids_path, "rb") as f:
            self.ids = pickle.load(f)

        # Defensive: ensure ids are strings padded
        self.ids = [str(x).zfill(10) for x in self.ids]
        return self.embeddings, self.ids

    def build_or_load_faiss(self, persist: bool = True) -> faiss.Index:
        """
        Build a FAISS IndexFlatIP over normalized embeddings.
        Optionally cache it to Embeddings/faiss.index
        """
        if persist and os.path.exists(self.paths.faiss_path):
            self.index = faiss.read_index(self.paths.faiss_path)
            return self.index

        if self.embeddings is None or self.ids is None:
            self.load_embeddings()

        emb = np.ascontiguousarray(self.embeddings, dtype="float32")
        index = faiss.IndexFlatIP(emb.shape[1])  # cosine sim because vectors are normalized
        index.add(emb)

        if persist:
            faiss.write_index(index, self.paths.faiss_path)

        self.index = index
        return index

    def load_clip(self) -> None:
        if self.model is not None and self.preprocess is not None:
            return
        model, preprocess = clip.load(self.clip_model_name, device=self.device)
        model.eval()
        self.model = model
        self.preprocess = preprocess

    def setup(self, build_embeddings_if_missing: bool = True) -> None:
        """
        One-call setup: loads articles, ensures embeddings exist, loads embeddings, builds/loads FAISS, loads CLIP.
        """
        self.load_articles()
        if build_embeddings_if_missing:
            self.ensure_embeddings()
        self.load_embeddings()
        self.build_or_load_faiss(persist=True)
        self.load_clip()

    # -------- Retrieval helpers --------

    def _faiss_search(self, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("FAISS index not initialized. Call setup().")
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype("float32")
        return self.index.search(query_vec, top_k)

    def _details_df(self, ids_in_order: List[str]) -> pd.DataFrame:
        if self.articles_df is None:
            self.load_articles()

        if not ids_in_order:
            return pd.DataFrame()

        df = self.articles_df[self.articles_df["article_id"].isin(ids_in_order)].copy()
        pos = {aid: i for i, aid in enumerate(ids_in_order)}
        df = df.sort_values(by="article_id", key=lambda s: s.map(pos))

        cols = [
            "article_id",
            "prod_name",
            "product_type_name",
            "graphical_appearance_name",
            "colour_group_name",
        ]
        cols = [c for c in cols if c in df.columns]
        return df[cols]

    # -------- Public API: the 3 modes --------

    def search_by_text(self, query: str, top_k: Optional[int] = None) -> Dict:
        """
        Text -> image
        """
        top_k = top_k or self.top_k_default
        self.load_clip()

        tokens = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_emb = self.model.encode_text(tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        q = text_emb.cpu().numpy().astype("float32")
        distances, indices = self._faiss_search(q, top_k)

        hits = [self.ids[i] for i in indices[0]]
        return {
            "mode": "text->image",
            "query": query,
            "ids": hits,
            "scores": distances[0].tolist(),
            "details": self._details_df(hits),
        }

    def search_by_article_id(self, article_id: str, top_k: Optional[int] = None) -> Dict:
        """
        Catalog image -> image (by article_id)
        """
        top_k = top_k or self.top_k_default
        if self.embeddings is None or self.ids is None:
            raise RuntimeError("Embeddings/IDs not loaded. Call setup().")

        aid = str(article_id).zfill(10)
        try:
            idx = self.ids.index(aid)
        except ValueError:
            raise ValueError(f"article_id={aid} not found in embeddings IDs list.")

        q = self.embeddings[idx].reshape(1, -1).astype("float32")

        # +1 to include itself, then remove it
        distances, indices = self._faiss_search(q, top_k + 1)
        raw_hits = [self.ids[i] for i in indices[0]]

        hits = [x for x in raw_hits if x != aid][:top_k]
        # scores aligned with raw; re-build scores for filtered ids:
        score_map = {self.ids[i]: float(distances[0][j]) for j, i in enumerate(indices[0])}
        scores = [score_map[x] for x in hits]

        return {
            "mode": "catalog_image->image",
            "query": aid,
            "ids": hits,
            "scores": scores,
            "details": self._details_df(hits),
        }

    def search_by_image_path(self, image_path: str, top_k: Optional[int] = None) -> Dict:
        """
        External photo -> image
        """
        top_k = top_k or self.top_k_default
        self.load_clip()

        img = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        q = emb.cpu().numpy().astype("float32")
        distances, indices = self._faiss_search(q, top_k)

        hits = [self.ids[i] for i in indices[0]]
        return {
            "mode": "external_image->image",
            "query": image_path,
            "ids": hits,
            "scores": distances[0].tolist(),
            "details": self._details_df(hits),
        }


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Multimodal retrieval: text->image, image->image (catalog), image->image (external).")
    parser.add_argument("--root", type=str, default=None, help="Repo root (defaults to cwd).")
    parser.add_argument("--topk", type=int, default=5, help="Number of results.")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_text = sub.add_parser("text", help="Text -> image")
    p_text.add_argument("--query", type=str, required=True)

    p_aid = sub.add_parser("article", help="Catalog image -> image (by article_id)")
    p_aid.add_argument("--article_id", type=str, required=True)

    p_img = sub.add_parser("image", help="External image -> image (by file path)")
    p_img.add_argument("--path", type=str, required=True)

    args = parser.parse_args()

    paths = default_paths(args.root)
    r = MultiModalRetriever(paths=paths)
    r.setup(build_embeddings_if_missing=True)

    t0 = time.time()

    if args.cmd == "text":
        out = r.search_by_text(args.query, top_k=args.topk)
    elif args.cmd == "article":
        out = r.search_by_article_id(args.article_id, top_k=args.topk)
    else:
        out = r.search_by_image_path(args.path, top_k=args.topk)

    print(f"\nMode: {out['mode']}")
    print(f"Query: {out['query']}")
    print(f"Elapsed: {time.time() - t0:.3f}s\n")

    # Print IDs + scores
    for i, (aid, score) in enumerate(zip(out["ids"], out["scores"]), start=1):
        print(f"{i:2d}. {aid}  score={score:.4f}")

    # Print metadata table
    print("\nDetails:")
    print(out["details"].to_string(index=False))


if __name__ == "__main__":
    main()
