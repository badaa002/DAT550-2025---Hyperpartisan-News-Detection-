import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, balanced_accuracy_score, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt
  

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

