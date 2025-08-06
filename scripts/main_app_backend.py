from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import os

app = FastAPI()
templates = Jinja2Templates(directory="frontend")

# Load models
clf_path = os.path.join("models", "draft_classifier.pkl")
reg_path = os.path.join("models", "round_predictor.pkl")

draft_clf = joblib.load(clf_path)
round_reg = joblib.load(reg_path)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request,
            position: str = Form(...),
            height: float = Form(...),
            weight: float = Form(...),
            forty: float = Form(...),
            bench: float = Form(...),
            vertical: float = Form(...),
            broad: float = Form(...),
            shuttle: float = Form(...),
            cone: float = Form(...),
            passing_yards: float = Form(...),
            passing_tds: float = Form(...),
            rushing_yards: float = Form(...),
            rushing_tds: float = Form(...),
            receiving_yards: float = Form(...),
            receiving_tds: float = Form(...),
            tackles: float = Form(...),
            sacks: float = Form(...),
            interceptions: float = Form(...)):

    # Define positions as one-hot features (for simplicity, map few example positions)
    pos_features = [1 if pos == position else 0 for pos in ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'CB', 'S']]

    features = np.array(pos_features + [height, weight, forty, bench, vertical, broad, shuttle, cone,
                                        passing_yards, passing_tds, rushing_yards, rushing_tds,
                                        receiving_yards, receiving_tds, tackles, sacks, interceptions]).reshape(1, -1)

    draft_pred = draft_clf.predict(features)[0]
    round_pred = int(round(round_reg.predict(features)[0])) if draft_pred == 1 else "N/A"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": "Drafted" if draft_pred == 1 else "Not Drafted",
        "round_result": round_pred
    })
