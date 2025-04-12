import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, balanced_accuracy_score, ConfusionMatrixDisplay, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from collections import defaultdict

def load_cleaned_data(file_to_load):
    return pd.read_csv(file_to_load, sep='\t')


def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def metrics(y_test, y_pred, y_pred_proba):
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy: {balanced_acc:.2f}")    

    precision = precision_score(y_test, y_pred, average='binary', zero_division=1)
    print(f"Precision: {precision:.2f}")

    recall = recall_score(y_test, y_pred, average='binary', zero_division=1)
    print(f"Recall: {recall:.2f}")

    f1 = f1_score(y_test, y_pred, zero_division=1)
    print(f"F1: {f1:.2f}")
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.2f}")
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=2, label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Hyperpartisan Classification')
    plt.legend()
    plt.show()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    balanced_acc = balanced_accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    return {
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_confusion_matrix(y_true, y_pred, labels=["Not Hyper", "Hyper"]):
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        cmap='Blues'
    )
    plt.title("Confusion Matrix")
    plt.show()

def store_metrics(results_list, model_name, setup_label, y_test, y_pred, y_pred_proba):
    results_list.append({
        "Model": model_name,
        "Setup": setup_label,
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Balanced Accuracy": round(balanced_accuracy_score(y_test, y_pred),3),
        "Precision": round(precision_score(y_test, y_pred),3),
        "Recall": round(recall_score(y_test, y_pred),3),
        "F1 Score": round(f1_score(y_test, y_pred),3),
        "AUC": round(roc_auc_score(y_test, y_pred_proba),3)
    })

def load_config(config_path="config.json"):
    """Load configuration from a JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_model_paths(config, fold=None):
    """Get model paths based on config"""
    if fold is not None:
        return os.path.join(config["training"]["output_dir"], f"xlm_roberta_fold_{fold}")
    else:
        return os.path.join(config["training"]["output_dir"], "xlm_roberta_final")


def get_fold_model_paths(config):
    """Get paths for all fold models"""
    return [get_model_paths(config, fold=i+1) for i in range(config["cross_validation"]["n_splits"])]


def apply_oversampling(X_train, y_train, random_state=42):
    ros = RandomOverSampler(random_state=random_state)
    train_indices = np.array(range(len(X_train))).reshape(-1, 1)
    train_indices_resampled, y_train_resampled = ros.fit_resample(train_indices, y_train.values)
    train_indices_resampled = train_indices_resampled.flatten()
    
    X_train_resampled = X_train.iloc[train_indices_resampled].reset_index(drop=True)
    y_train_resampled = pd.Series(y_train_resampled)
    
    print(f"Original training data distribution: {y_train.value_counts().to_dict()}")
    print(f"Resampled training data distribution: {y_train_resampled.value_counts().to_dict()}")
    
    return X_train_resampled, y_train_resampled


def aggregate_publishers_by_domain(df):
    """
    Aggregate publishers based on domain, filtering out domains with mixed labels.
    
    Args:
        df: DataFrame with columns 'domain' and 'label'
        
    Returns:
        DataFrame with consistent publishers and their labels
    """
    # Group by domain and check label consistency
    domain_stats = defaultdict(lambda: {'count': 0, 'labels': set()})
    
    # First pass: collect label information for each domain
    for _, row in df.iterrows():
        domain = row['domain']
        label = row['label']
        domain_stats[domain]['count'] += 1
        domain_stats[domain]['labels'].add(label)
    
    # Filter consistent domains (domains with only one type of label)
    consistent_domains = []
    for domain, stats in domain_stats.items():
        if len(stats['labels']) == 1:  # Only one unique label
            label = list(stats['labels'])[0]  # Get the single label
            consistent_domains.append({
                'domain': domain,
                'label': label,
                'article_count': stats['count']
            })
    
    # Create dataframe of consistent publishers
    consistent_df = pd.DataFrame(consistent_domains)
    
    # Sort by article count for better visibility
    return consistent_df.sort_values(by='article_count', ascending=False)