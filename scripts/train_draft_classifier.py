import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

# Load data
df = pd.read_csv("data/combine_with_stats.csv")

# Drop rows with missing required values
required_columns = [
    "Height_in_inches", "Weight", "40yd", "Bench", "Vertical", "Broad Jump"
]
df = df.dropna(subset=required_columns)

# Select features and target
features = df[[
    "position", "Height_in_inches", "Weight", "40yd", "Bench",
    "Vertical", "Broad Jump"
]]
target = df["drafted"]

# One-hot encode position
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_positions = encoder.fit_transform(features[["position"]])
encoded_df = pd.DataFrame(encoded_positions, columns=encoder.get_feature_names_out(["position"]))

X = pd.concat([encoded_df, features.drop(columns=["position"]).reset_index(drop=True)], axis=1)
y = target

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model and encoder
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/draft_classifier.pkl")
joblib.dump(encoder, "models/position_encoder.pkl")

print("Draft classifier model and encoder saved.")
