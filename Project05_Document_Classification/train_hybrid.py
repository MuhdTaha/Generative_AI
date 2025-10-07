from hybrid_model import generate_pseudo_labels, train_ml_classifier

# Step 1: Use Gemini to label your documents
df = generate_pseudo_labels("data/docs")

# Step 2: Train the classical model on those labels
train_ml_classifier(df)
