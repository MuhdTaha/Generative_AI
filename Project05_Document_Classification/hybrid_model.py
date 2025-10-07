import os
import json
import joblib
import pathlib
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from classify_ai import classify_document, DOCUMENT_TYPES, load_document

MODEL_PATH = "models/tfidf_logreg.pkl"
DATA_PATH = "data/labeled.csv"


# -----------------------------------------------------
# 1. Generate pseudo-labeled dataset using Gemini
# -----------------------------------------------------
def generate_pseudo_labels(data_dir: str = "data/docs") -> pd.DataFrame:
    results = []

    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.lower().endswith((".pdf", ".txt", ".docx")):
                continue

            fpath = os.path.join(root, fname)
            fpath = pathlib.Path(fpath).as_posix()  # ✅ Convert to POSIX path

            print(f"\n🔍 Classifying {fpath} with Gemini...")
            try:
                result = classify_document(fpath, DOCUMENT_TYPES)
                print("➡️ Result:", result)

                if "predicted_class" in result:
                    results.append({
                        "filename": os.path.relpath(fpath, data_dir),
                        "text": load_document(fpath),
                        "label": result["predicted_class"],
                        "confidence": result["confidence_score"]
                    })
                else:
                    print(f"⚠️ Skipping {fname}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"❌ Failed to classify {fname}: {e}")

    if not results:
        print("🚨 No valid classifications found. Check your files and API key.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"\n✅ Pseudo-labeled dataset saved to {DATA_PATH}")
    return df


# -----------------------------------------------------
# 2. Train ML classifier (TF-IDF + Logistic Regression)
# -----------------------------------------------------
def train_ml_classifier(labeled_df: pd.DataFrame) -> None:
    """
    Trains a TF-IDF + Logistic Regression model using pseudo-labels
    from the Gemini + classical ML hybrid pipeline.
    """

    print("Training ML model on pseudo-labels...")

    # --- 🧩 Step 1: Normalize column names ---
    labeled_df.columns = [col.strip().lower() for col in labeled_df.columns]

    if "content" in labeled_df.columns:
        labeled_df = labeled_df.rename(columns={"content": "text"})
    elif "document_text" in labeled_df.columns:
        labeled_df = labeled_df.rename(columns={"document_text": "text"})
    elif "text" not in labeled_df.columns:
        print("⚠️ No 'text' column found. Columns present:", labeled_df.columns.tolist())
        raise ValueError("Expected a column named 'text' or 'content' in the labeled dataset.")

    if "label" not in labeled_df.columns:
        raise ValueError("Missing required column 'label' in pseudo-labeled dataset.")

    # --- 🧠 Step 2: Split data ---
    X_train, X_test, y_train, y_test = train_test_split(
        labeled_df["text"], labeled_df["label"], test_size=0.2, random_state=42
    )

    # --- ⚙️ Step 3: Define ML pipeline ---
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("logreg", LogisticRegression(max_iter=300))
    ])

    # --- 📈 Step 4: Train model ---
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # --- 📊 Step 5: Evaluate ---
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # --- 💾 Step 6: Save model ---
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")


# -----------------------------------------------------
# 3. Hybrid prediction (Gemini + ML)
# -----------------------------------------------------
def hybrid_classify(file_path: str, alpha: float = 0.6) -> Dict[str, Any]:
    """
    Combines Gemini and ML model predictions using weighted confidence averaging.
    alpha = weight given to Gemini (0.0 - 1.0)
    """
    # Load ML model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("ML model not trained yet. Run train_ml_classifier() first.")

    model = joblib.load(MODEL_PATH)

    # Get document text
    text = load_document(file_path)

    # 1. AI prediction
    ai_result = classify_document(file_path, DOCUMENT_TYPES)

    # 2. ML prediction
    ml_pred = model.predict([text])[0]
    ml_probs = model.predict_proba([text])[0]
    ml_conf = float(np.max(ml_probs))

    # 3. Combine results
    if "confidence_score" in ai_result:
        ai_conf = ai_result["confidence_score"]
        ai_label = ai_result["predicted_class"]

        if ai_label == ml_pred:
            final_label = ai_label
            final_conf = (alpha * ai_conf) + ((1 - alpha) * ml_conf)
        else:
            # Weighted pick based on confidence
            final_label = ai_label if ai_conf * alpha >= ml_conf * (1 - alpha) else ml_pred
            final_conf = max(ai_conf * alpha, ml_conf * (1 - alpha))
    else:
        final_label = ml_pred
        final_conf = ml_conf

    return {
        "final_label": final_label,
        "final_confidence": round(final_conf, 3),
        "ai_prediction": ai_result,
        "ml_prediction": {"label": ml_pred, "confidence": ml_conf}
    }
