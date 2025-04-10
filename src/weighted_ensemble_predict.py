import torch

def weighted_ensemble_predict(inputs, fold_models, fold_metrics, device): 
    """Make prediction using weighted ensemble of models"""
    # Calculate weights from metrics
    weights = torch.tensor(fold_metrics, device=device)
    weights = weights / weights.sum()  # Normalize
    
    # Get predictions from all models
    all_probs = []
    for model in fold_models:
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            all_probs.append(probs)
    
    # Apply weighted average
    weighted_probs = torch.zeros_like(all_probs[0])
    for i, probs in enumerate(all_probs):
        weighted_probs += probs * weights[i]
    
    return weighted_probs