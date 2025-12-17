# 🛍️ StyleSense AI

### Where Style meets Elegance and Innovation

StyleSense AI is a comprehensive fashion recommendation system. The project is designed to provide a complete e-commerce search experience with two primary objectives:

1.  **Visual Search:** Allows users to find relevant articles using natural language text queries (e.g., "a red summer dress with flowers").
2.  **Frequently Bought Together:** Recommends items that are commonly purchased with a selected product.

This app is built using Streamlit, Python, CLIP for multimodal embeddings, and FAISS for high-speed vector search.

---

## 🚀 How to Run Locally

Follow these steps to set up and run the project on your local machine.

### Step 1: Clone the Repository

```
### Step 1: Clone the reporisotory
git clone [https://github.com/verdabatool/StyleSense-AI.git](https://github.com/verdabatool/StyleSense-AI.git)
cd StyleSense-AI

### Step 2: Set Up a Python Environment
It's highly recommended to use a virtual environment.

# Create a virtual environment
python -m venv .venv

# Activate it (on Mac/Linux)
source .venv/bin/activate

# Or activate it (on Windows)
.\.venv\Scripts\activate

# Install all required libraries
pip install -r requirements.txt
```

### Step 3: Download the Data  
This project requires the H&M dataset from Kaggle.

Download the data from this link:[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data).

Unzip the folder and place articles.csv and the images folder inside a new folder named Data in the root of the project.

```bash
StyleSense-AI/
|-- Data/
|   |-- articles.csv
|   |-- images/
|       |-- 010/
|       |-- 011/
|       |-- 012/
|       |-- ...
|-- Embeddings/
|-- streamlit_app.py
|-- multimodel_retrieval.py
|-- copurchase_map.py
|-- requirements.txt
|-- .gitignore
|-- README.md
```
### Step 4: Generate the Image Embeddings
This is a crucial one-time step that will index all your images. This script will create the Embeddings folder and the necessary model files.

From your terminal, run the `multimodel_retrieval.py` script:
```bash
python multimodel_retrieval.py.py
```
This will:

- Load the CLIP model.

- Scan your Data/images folder.

- Generate embeddings for all images.

- Create a new `Embeddings/` folder and save `image_embeddings.npy` and `image_ids.pkl` inside it.

Note: This may take 20-30 minutes, depending on your computer's hardware.

### Step 5: Build the "Frequently Bought Together" Index

Run the transaction analysis script first. This processes transactions_train.csv and saves the co-occurrence maps to the Embeddings/ folder.

```python
python copurchase_map.py
```
Output: Generates `co_purchase_article.pkl` and `co_purchase_type.pkl`.

### Step 6: Run the Streamlit App
Once the Embeddings folder exists, you are ready to start the web app.
```
streamlit run app.py
```
Streamlit will open a new tab in your browser. You can now start searching!

[Here is the demo video where the entire project is demonstrated](https://drive.google.com/file/d/11_iPtsTdkYNe8wFCLVNLEDPyXFUgGnEx/view?usp=sharing).






















