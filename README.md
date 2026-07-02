# Byte-Pair Encoding (BPE) Tokenizer Visualizer

This is a simple project built for educational purposes to understand and visualize Byte Pair Encoding (BPE) tokenization, which is widely used in modern Large Language Models (LLMs). 

It contains a basic BPE tokenizer implemented in Python (using GPT-2 style pre-tokenization regex) and a simple web interface to visualize how any input text gets broken down into subword tokens.

---

## 🌐 Live Web App
The hosted web app is available at: [bpe.ankandas.online](https://bpe.ankandas.online)

---

## 🛠️ How to Run Locally

### 1. Install Dependencies
You need Python 3.8+ installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Web Server
To launch the interactive visualizer locally, start the FastAPI backend:
```bash
uvicorn backend.app:app --reload
```
Now, open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

---

## 🧠 Training a Custom Tokenizer

To train a BPE tokenizer on your own text dataset (e.g., the provided [shakespeare.txt](./shakespeare.txt)), use the helper script [train_tokenizer.py](./scripts/train_tokenizer.py):

```bash
python scripts/train_tokenizer.py \
  --corpus shakespeare.txt \
  --vocab_size 5000 \
  --save models/tokenizer_shakespeare_5000.pkl
```

### Pre-trained Models
For convenience, some pre-trained models are already provided in the [models/](./models) folder:
- [tokenizer_shakespeare_300.pkl](./models/tokenizer_shakespeare_300.pkl) (small vocabulary, 300 tokens)
- [tokenizer_shakespeare_5000.pkl](./models/tokenizer_shakespeare_5000.pkl) (medium vocabulary, 5000 tokens)

These models are trained using the [shakespeare.txt](./shakespeare.txt) file.
You can load these files directly through the web UI under the "Load BPE Model" section.

---

## 📂 Project Structure

- [backend/](./backend): FastAPI server and tokenizer logic.
  - [tokenizer.py](./backend/tokenizer.py): Basic BPE tokenizer implementation.
  - [pretokenizer.py](./backend/pretokenizer.py): GPT-2 regex pre-tokenization and thread-pool accelerated training.
  - [app.py](./backend/app.py): Server API routes (`/tokenize`, `/load`).
- [frontend/](./frontend): HTML, CSS, and JS files for the interactive visualization dashboard.
- [scripts/](./scripts): Python training scripts.
- [models/](./models): Pre-trained model pickle files.
