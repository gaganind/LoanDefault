# api.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import StringIO, BytesIO

from evaluate import evaluate_on_dataset

app = FastAPI(title="Loan Default Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/evaluate-file")
async def evaluate_file(file: UploadFile = File(...)):
    """
    Accept a CSV file, run model, return metrics + confusion matrix.
    File MUST contain a 'Default' column as ground truth.
    """
    content = await file.read()
    try:
        s = content.decode("utf-8")
        df = pd.read_csv(StringIO(s))
    except UnicodeDecodeError:
        df = pd.read_csv(BytesIO(content))

    metrics = evaluate_on_dataset(df)
    return {"metrics": metrics}

@app.get("/")
def home():
    return {"message": "Loan Default Model API is running"}