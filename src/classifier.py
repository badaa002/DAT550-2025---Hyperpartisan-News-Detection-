import joblib
import json
import os

def save_full_model(model, vectorizer, scaler, metadata, output_dir="models", name="best_model"):
    """Save model, vectorizer, scaler, and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(output_dir, f"{name}.pkl"))
    joblib.dump(vectorizer, os.path.join(output_dir, f"{name}_vectorizer.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, f"{name}_scaler.pkl"))
    
    with open(os.path.join(output_dir, f"{name}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f" Model, vectorizer, scaler, and metadata saved to {output_dir}/")


# model, vectorizer, scaler, metadata = load_full_model("models", "best_rf")

def load_full_model(output_dir="models", name="best_model"):
    """Load model, vectorizer, scaler, and metadata."""
    model = joblib.load(os.path.join(output_dir, f"{name}.pkl"))
    vectorizer = joblib.load(os.path.join(output_dir, f"{name}_vectorizer.pkl"))
    scaler = joblib.load(os.path.join(output_dir, f"{name}_scaler.pkl"))
    
    with open(os.path.join(output_dir, f"{name}_metadata.json"), "r") as f:
        metadata = json.load(f)
    
    print(f" Loaded model, vectorizer, scaler, and metadata from {output_dir}/")
    return model, vectorizer, scaler, metadata
