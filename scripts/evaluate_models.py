import pandas as pd
import joblib
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split

# Load cleaned data
df = pd.read_csv("data/combined_cleaned.csv")

# Draft classifier evaluation
features_clf = df.drop(columns=["drafted", "draft_round", "player"])
target_clf = df["drafted"]
X_train, X_test, y_train, y_test = train_test_split(features_clf, target_clf, test_size=0.2, random_state=42)

clf = joblib.load("models/draft_classifier.pkl")
y_pred_clf = clf.predict(X_test)
print("Draft Classifier Evaluation:")
print(classification_report(y_test, y_pred_clf))

# Round predictor evaluation
df_drafted = df[df["drafted"] == 1]
features_reg = df_drafted.drop(columns=["drafted", "draft_round", "player"])
target_reg = df_drafted["draft_round"]
X_train, X_test, y_train, y_test = train_test_split(features_reg, target_reg, test_size=0.2, random_state=42)

reg = joblib.load("models/round_predictor.pkl")
y_pred_reg = reg.predict(X_test)
print("Round Predictor Evaluation:")
print("MSE:", mean_squared_error(y_test, y_pred_reg))
