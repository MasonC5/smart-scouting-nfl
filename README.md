# Smart Scouting NFL

Smart Scouting NFL is a machine learning-powered web application that predicts whether a college football player will be drafted into the NFL, and in which round, based on their combine and collegiate stats.

## Features

- Predict NFL draft status (Drafted/Not Drafted)
- Predict NFL draft round
- Web interface using FastAPI & HTML
- Trained models using player combine and NCAA performance stats (2010–2025)

## Machine Learning Models

- **Draft Classifier:** Classifies if a player will be drafted.
- **Round Predictor:** Predicts the round a drafted player will be selected.

Both models are trained using Random Forest and saved as `.pkl` files for use in the FastAPI backend.

## 🗂️ Project Structure

smart-scouting-nfl/
│
├── data/ # Raw & processed data (CSV)
│ ├── combine_with_stats.csv
│ └── non_null_counts_by_position.csv
│
├── models/ # Trained ML model files
│ ├── draft_classifier.pkl
│ ├── round_predictor.pkl
│ ├── position_encoder.pkl
│ └── position_encoder_round.pkl
│
├── scripts/ # Python scripts for data & models
│ ├── fetch_cfb_player_stats.py
│ ├── train_draft_classifier.py
│ ├── train_round_predictor.py
│ ├── evaluate_models.py
│ ├── main_app_backend.py
│ └── fastapi_backend.py
│
├── templates/ # HTML frontend (Jinja2 templates)
│ └── index.html
├── preprocessed_draft_data.pkl # Pre-cleaned dataset for training
├── requirements.txt # Python dependencies
└── README.md # Project overview (this file)


## Running the App

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```
2. **Train models (if not already trained)*:

```bash
python scripts/train_draft_classifier.py
python scripts/train_round_predictor.py
```
3. Run FastAPI app:
```bash
uvicorn scripts.fastapi_backend:app --reload
```
4. Visit in browser: http://127.0.0.1:800
