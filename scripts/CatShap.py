import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import seaborn as sns
from functools import partial
from skopt import gp_minimize
from skopt.space import Real, Integer
from pathlib import Path
import shap
shap.initjs()


def load_path(path):
    data = pd.read_csv(path)
    return data


def evaluate_detection(y_true, y_pred, y_scores):
    y_true_binary = (y_true == 1).astype(int)
    precision = precision_score(y_true_binary, y_pred)
    recall = recall_score(y_true_binary, y_pred)
    f1 = f1_score(y_true_binary, y_pred)
    roc_auc = roc_auc_score(y_true_binary, y_scores)
    pr_auc = average_precision_score(y_true_binary, y_scores)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


def optimization(param_names, params, X_train, y_train):
    params_dict = dict(zip(param_names, params))
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s = []
    for train_idx, test_idx in kf.split(X_train, y_train):
        Xtrain = X_train[train_idx]
        Xtest = X_train[test_idx]
        ytrain = y_train[train_idx]
        ytest = y_train[test_idx]

        X_tr_sub, X_es, y_tr_sub, y_es = train_test_split(Xtrain, ytrain, test_size=0.2, random_state=42, stratify=ytrain)

        clf = CatBoostClassifier(**params_dict, 
                                 loss_function='Logloss', 
                                 eval_metric='F1', 
                                 random_state=42, 
                                 verbose=0)
                                    
        clf.fit(Xtrain, ytrain,
                eval_set=(X_es, y_es),
                early_stopping_rounds=100,
                use_best_model=True
                )
        
        y_prob = clf.predict_proba(Xtest)[:, 1]  
        y_pred = (y_prob >= 0.5).astype(int)    
        f1 = f1_score(ytest, y_pred)
        f1s.append(f1)
    
    return -1 * np.mean(f1s)


def bayesian_search_params(X_train, y_train, X_test, y_test):
    param_space = [
        Integer(1, 5, name='max_depth'),
        Integer(10, 1000, name='iterations'), 
        Real(0.001, 0.1, name='learning_rate', prior='log-uniform'),
        Integer(1, 20, name='l2_leaf_reg'),
        Integer(1, 40, name='min_data_in_leaf')
    ]

    param_names = [
        'max_depth',
        'iterations',
        'learning_rate',
        'l2_leaf_reg',
        'min_data_in_leaf'
    ]

    optimization_function = partial(optimization, param_names, X_train=X_train, y_train=y_train)

    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=20,
        random_state=42,
        n_random_starts=5,
        verbose=True
    )

    best_params = {
        'max_depth': int(result.x[0]),
        'iterations': int(result.x[1]),
        'learning_rate': result.x[2],
        'l2_leaf_reg': int(result.x[3]),
        'min_data_in_leaf': int(result.x[4])
    }
    best_f1 = -result.fun
    print(f"Best params: {best_params}, Best F1: {best_f1:.4f}")



def categoricalboost(X_train, y_train, X_test):
    clf = CatBoostClassifier(max_depth=3,
                            iterations=500,
                            learning_rate=0.5,
                            l2_leaf_reg=1,
                            min_data_in_leaf=10,
                            scale_pos_weight=3,
                            objective='Logloss',
                            eval_metric='F1',
                            random_state=42, 
                            boosting_type='Ordered',
                            verbose=0
    )
    clf.fit(X_train, y_train, early_stopping_rounds=30)
    y_pred = clf.predict(X_test) 
    y_scores = clf.predict_proba(X_test)[:, 1]
    return y_pred, y_scores, clf


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"])
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def plot_histogram(y_test, y_pred, y_scores):
    tn_indices = (y_test == 0) & (y_pred == 0)  # True Negative
    fp_indices = (y_test == 0) & (y_pred == 1)  # False Positive
    fn_indices = (y_test == 1) & (y_pred == 0)  # False Negative
    tp_indices = (y_test == 1) & (y_pred == 1)
    idx_fn = np.where(fn_indices)
    print(idx_fn[0])
    plt.figure(figsize=(12, 7))
    plt.hist(y_scores[tn_indices], bins=50, alpha=0.5, label='True Negative (Normal)', color='green')
    plt.hist(y_scores[fp_indices], bins=50, alpha=0.5, label='False Positive (Normal misclassified)', color='orange')
    plt.hist(y_scores[fn_indices], bins=50, alpha=0.5, label='False Negative (Anomaly missed)', color='red')
    plt.hist(y_scores[tp_indices], bins=50, alpha=0.5, label='True Positive (Anomaly)', color='blue')
    plt.axvline(0.5, color='black', linestyle='--', label='Default Threshold (0.5)')
    plt.title('Distribution of Anomaly Scores with Classification Results (catboost)')
    plt.xlabel('Anomaly Score (Probability)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def plot_roc_curve(y_true, y_scores, title="ROC Curve"):
    y_true_binary = (y_true == 1).astype(int)
    fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
    roc_auc = roc_auc_score(y_true_binary, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def plot_pr_curve(y_true, y_scores, title="Precision-Recall Curve"):
    y_true_binary = (y_true == 1).astype(int)
    precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
    pr_auc = average_precision_score(y_true_binary, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()



def background_data(X_train, y_train):
    bg_size = 1000
    normal_idx = np.where(y_train == 0)[0]
    idx = np.random.choice(normal_idx, size=bg_size, replace=False)
    background = X_train[idx].astype(np.float32) 
    np.save("bg_1000.npy", background)              
    print("Saved background data:", background.shape)
    return background


def shap_explain(X_train, X_test, y_test, clf, data, background):
    feature_names = data.drop(columns=['Pass/Fail']).columns
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    explainer = shap.TreeExplainer(clf, background)
    shap_values = explainer(X_test_df)
    
    order = list(range(len(feature_names)))
    shap_values_ordered = shap_values[:, order]
    X_test_df_ordered = X_test_df[feature_names]

    shap.summary_plot(shap_values_ordered, X_test_df_ordered, plot_type="bar",
                      feature_names=feature_names, max_display=10, sort=False)
 
    shap.summary_plot(shap_values_ordered, X_test_df_ordered, feature_names=feature_names, sort=False,  max_display=10)
    
    shap.plots.bar(shap_values_ordered)
    shap.plots.beeswarm(shap_values_ordered)

    anomaly_idx = np.where(y_test == 1)[0]
    sample_idx = anomaly_idx[0] 
    shap_values_single = explainer(X_test_df.iloc[[sample_idx]])
    shap.plots.waterfall(shap_values_single[0])

def model_save(clf, threshold = 0.5):
    clf.save_model("model.cbm")        
    np.save("threshold.npy", threshold)


def main():
    ROOT = Path(__file__).parent
    path = ROOT / "combined_data_vaegan.csv"
    data = load_path(path)
    print("Data shape:", data.shape)
    
    X = data.drop(columns=['Pass/Fail']).values  
    y = data['Pass/Fail'].values
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    y_binary = (y == 1).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)  

    #best_param = bayesian_search_params(X_train, y_train, X_test, y_test)
    
    
    y_pred, y_scores, clf = categoricalboost(X_train, y_train, X_test)

    metrics = evaluate_detection(y_test, y_pred, y_scores)
    print("Detection performance on test set:", metrics)

    plot_histogram(y_test, y_pred, y_scores)
    plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix for Catboost")

    # plot_roc_curve(y_test, y_scores, title="ROC Curve for Catboost")
    # plot_pr_curve(y_test, y_scores, title="Precision-Recall Curve for Catboost")

    model_save(clf)
    
    background = background_data(X_train, y_train)
    #shap_explain(X_train, X_test, y_test, clf, data, background)
    

if __name__ == "__main__":
    main()