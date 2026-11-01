#Extract 10 samples for test (6 normal, 4 anomaly)
import pandas as pd, joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, KernelPCA

import tensorflow as tf


def load_data(file_path):
    data = pd.read_csv(file_path, encoding="utf-8")
    return data


def get_test_data(df, n_normal=7, n_anomaly=3):
    df_remained = df.copy()
    normal_data = df_remained[df_remained['Pass/Fail'] == -1].sample(n=n_normal, random_state=42)
    normal_idx = normal_data.index
    anomaly_data = df_remained[df_remained['Pass/Fail'] == 1].sample(n=n_anomaly, random_state=42)
    anomaly_idx = anomaly_data.index

    test_data = pd.concat([normal_data, anomaly_data]).reset_index(drop=True)

    indices_to_drop = normal_idx.union(anomaly_idx)
    df_remained.drop(indices_to_drop, inplace=True)

    return test_data, df_remained

def handle_missing_value(df):
    numeric_cols = df.select_dtypes(include=['float64']).columns
    cols_to_drop = []
    for col in numeric_cols:
        null_proportion = df[col].isnull().sum() / df.shape[0]
        if null_proportion > 0.2:
            cols_to_drop.append(col)
        else:
            df[col] = df[col].fillna(df[col].mean()) 
            # df[col] = df[col].fillna(0) 
    df = df.drop(columns=cols_to_drop)  
    return df


def drop_datetime(df, time_col='Time'):
    df = df.drop(columns=[time_col])
    return df


def remove_constant_columns(df, verbose=True):
    columns_to_drop = [col for col in df.columns if df[col].nunique() == 1]
    df_cleaned = df.drop(columns=columns_to_drop)
    # if verbose and columns_to_drop:
    #     print(f"Removed columns with no variation : {columns_to_drop}")
    return df_cleaned


def remove_low_variance(df):
    sel = VarianceThreshold(threshold=0.05)
    sel.fit(df)
    mask = sel.get_support()    
    drop_cols = df.columns[~mask]
    # print("Columns to drop :",drop_cols)
    df_reduced = df.drop(columns = drop_cols)
    return df_reduced


def remove_low_corr_with_target(df, threshold, target_col, verbose=True):
    num_cols = df.select_dtypes(include=['float64']).drop(columns=[target_col], errors='ignore').columns
    correlations = df[num_cols].corrwith(df[target_col]).abs()
    low_corr_cols = correlations[correlations < threshold].index.tolist()
    df_reduced = df.drop(columns=low_corr_cols)
    # if verbose and low_corr_cols:
    #     print(f"Removed {len(low_corr_cols)} columns with low correlation (< {threshold}) with {target_col}: {low_corr_cols}")
    return df_reduced


def remove_high_corr(df, threshold, target_col):
    data_num = df.select_dtypes(include=['float64']).drop(columns=[target_col], errors='ignore')
    corr = data_num.corr().abs()
    drop_cols = set()
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if corr.iloc[i, j] >= threshold:
                drop_cols.add(corr.columns[j])
    #print("Columns to drop:", drop_cols)
    df_reduced = df.drop(columns=drop_cols)
    return df_reduced


def data_standardization(df, target_col):
    num_cols = df.select_dtypes(include=['float64']).drop(columns=[target_col], errors='ignore').columns
    normal_df = df[df["Pass/Fail"] == -1][num_cols] 
    scaler = StandardScaler() 
    scaler.fit(normal_df)
    df[num_cols] = scaler.transform(df[num_cols])
    return df, scaler


def plot_pass_fail_distribution(df, target_col):
    size = df[target_col].value_counts()
    labels = ["Normal", "Anomalies"]
    colors = ["blue", "red"]
    explode = (0, 0.2)
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _, autotexts = ax.pie(
        size,
        labels=None,                 
        colors=colors,
        explode=explode,
        autopct="%2.2f%%",
        pctdistance=1.15,            
        startangle=0,
        shadow=False
    )
    fig.legend(wedges, labels, title="Class", loc="upper right", fontsize=12)
    ax.set_title("Proportion of Normal vs. Anomalies")
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(8,6))
    ax = df[target_col].value_counts().plot(
            kind="bar",
            color=['blue','red']
    )
    for i, v in enumerate(df[target_col].value_counts()):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom')  
    plt.title("Pass/Fail Counts")
    plt.xlabel("Pass/Fail")
    plt.ylabel("Count")
    plt.xticks(ticks=[0, 1], labels=['Pass', 'Fail'], rotation=0)  
    plt.show()


def plot_correlation_heatmap(df, target_col):
    data_num = (
        df
        .select_dtypes(include=['float64'])
        .drop(columns=[target_col], errors='ignore')
    )
    data_corr = data_num.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(data_corr, cmap="YlGnBu")
    plt.title('Data Correlation', fontsize=20)
    plt.show()


def plot_distribution(df, target_col):
    pass_data = df[df[target_col] == -1] 
    fail_data = df[df[target_col] == 1] 
    for feature_idx in range(3):
        plt.figure(figsize=(8,6))
        plt.hist(pass_data.iloc[:, feature_idx], bins=50, density=True, alpha=0.5, label='Pass Data', color='red')
        plt.hist(fail_data.iloc[:, feature_idx], bins=50, density=True, alpha=0.5, label='Fail Data', color='blue')
        sns.kdeplot(pass_data.iloc[:, feature_idx], label='Pass Data Distribution', color='red')
        sns.kdeplot(fail_data.iloc[:, feature_idx], label='Fail Data Distribution', color='blue')
        plt.title(f'Distribution of Feature {feature_idx}')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

def pca_check_distribution(standard_df):
    normal_data = standard_df[standard_df['Pass/Fail'] == -1].drop(columns=['Pass/Fail'])
    anomaly_data = standard_df[standard_df['Pass/Fail'] == 1].drop(columns=['Pass/Fail'])
    normal_values = normal_data.values
    anomaly_values = anomaly_data.values
    pca = KernelPCA(n_components=2, kernel='rbf')
    normal_pca = pca.fit_transform(normal_values)
    anomaly_pca = pca.fit_transform(anomaly_values)
    plt.figure(figsize=(8,6))
    plt.scatter(normal_pca[:, 0], normal_pca[:, 1], label="Normal Data", alpha=0.5, color="blue")
    plt.scatter(anomaly_pca[:, 0], anomaly_pca[:, 1], label="Anomaly Data", alpha=0.8, color="red", marker="x")
    plt.title("KernelPCA Distribution of Normal vs. Anomalies")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()


def save_real_anomaly_data(df, target_col):
    fail_df = df[df[target_col] == 1]
    return fail_df
    

def main():
    ROOT = Path(__file__).parent
    file_path = ROOT / "uci-secom.csv"
    output_path = ROOT / "uci-secom-cleaned.csv"  
    anomaly_data_path = ROOT / "anomaly_data.csv"

    data = load_data(file_path)
    print("Data loaded, shape:\n", data.shape)
    # print(data.describe())
    # print(data.info())

    test_data, data = get_test_data(data)
    test_data.to_csv(ROOT / "test_data.csv", index=False)
    print("Test data prepared with 6 normal and 4 anomalous samples, shape:\n", test_data.shape)
    print("The remained data, shape:\n", data.shape)


    data = drop_datetime(data)
    print("After dropped time column, shape:\n", data.shape)

    ALL_COLS = data.drop(columns="Pass/Fail").columns.tolist()
    np.save(ROOT / "all_cols.npy", ALL_COLS)
    print("Saved all_cols.npy")

    data = handle_missing_value(data)
    print("After handled missing values, shape:\n", data.shape)
    # print(data.isnull().any().any())
    
    data_filter = remove_constant_columns(data)
    print("After removing constant features, shape:\n", data_filter.shape)

    data_filter = remove_low_variance(data_filter)
    print("After removing low variance features, shape:\n", data_filter.shape)

    data_filter = remove_low_corr_with_target(data_filter, threshold=0.05, target_col='Pass/Fail')
    print("After removing low correlation features with Pass/Fail, shape:\n", data_filter.shape)
    
    data_filter = remove_high_corr(data_filter, threshold=0.75, target_col='Pass/Fail')
    print("After removing highly correlated features, shape:\n", data_filter.shape)

    data_standard, scaler = data_standardization(data_filter, target_col='Pass/Fail')
    print("After standarlized data, shape:\n", data_standard.shape)
    joblib.dump(scaler, ROOT / "scaler.pkl")
    print("Saved scaler.pkl")

    data_standard.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")
    
    plot_pass_fail_distribution(data_standard, target_col='Pass/Fail')
    # plot_correlation_heatmap(data_standard, target_col='Pass/Fail')
    pca_check_distribution(data_standard)

    plot_distribution(data_standard, target_col='Pass/Fail')
    
    fail_df = save_real_anomaly_data(data_standard, target_col='Pass/Fail')
    print("Real anomaly data shape:\n", fail_df.shape)
    fail_df.to_csv(anomaly_data_path, index = False)
    print(f"Anomaly data saved to : {anomaly_data_path}")

    remain_features = fail_df.drop(columns="Pass/Fail").columns.tolist()
    IDX = np.array([ALL_COLS.index(name) for name in remain_features], dtype=np.int32)  #找出name在ALL_COLS.index的位置編號
    np.save(ROOT / "idx.npy", IDX)
    print("Saved idx.npy")

    
if __name__ == "__main__":
    main()