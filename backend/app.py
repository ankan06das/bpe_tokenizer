import tempfile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .schemas import TokenizeRequest
from .pretokenizer import PreBPETokenizer

app = FastAPI()

bpe = PreBPETokenizer(num_workers=8)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.post("/tokenize")
def tokenize(req: TokenizeRequest):
    return bpe.tokenize(req.text)


@app.post("/load")
async def load(file: UploadFile = File(...)):
    if not file.filename.endswith(".pkl"):
        raise HTTPException(400, "File must be a .pkl file")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        f.write(await file.read())
        tmppath = f.name
    try:
        bpe.load(tmppath)
    except Exception as e:
        raise HTTPException(400, f"Failed to load model: {e}")
    finally:
        Path(tmppath).unlink(missing_ok=True)
    return {"status": "loaded", "vocab_size": len(bpe.vocab)}


@app.get("/models")
def list_models():
    models_dir = Path(__file__).resolve().parent.parent / "models"
    if not models_dir.exists():
        return []
    return [p.name for p in models_dir.glob("*.pkl")]


@app.post("/load-pretrained/{filename}")
def load_pretrained(filename: str):
    models_dir = Path(__file__).resolve().parent.parent / "models"
    model_path = models_dir / filename
    # Security check: resolve and verify it's inside models_dir
    try:
        model_path = model_path.resolve()
        if not model_path.is_file() or models_dir.resolve() not in model_path.parents:
            raise HTTPException(400, "Invalid model file")
    except Exception:
        raise HTTPException(400, "Invalid model file path")
    
    try:
        bpe.load(str(model_path))
    except Exception as e:
        raise HTTPException(400, f"Failed to load model: {e}")
    return {"status": "loaded", "vocab_size": len(bpe.vocab)}

