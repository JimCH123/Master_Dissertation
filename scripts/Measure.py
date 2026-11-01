import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel
from pathlib import Path

def load_data(anomaly_path, synthetic_vae_path, synthetic_vaegan_path):
    anomaly_df = pd.read_csv(anomaly_path)
    synthetic_vae_df = pd.read_csv(synthetic_vae_path)
    synthetic_vaegan_df = pd.read_csv(synthetic_vaegan_path)
    return anomaly_df, synthetic_vae_df, synthetic_vaegan_df

def kl_divergence(anomaly_df, synthetic_df, bins=50):
    anomaly_df = anomaly_df.drop(columns=['Pass/Fail'])
    synthetic_df = synthetic_df.drop(columns=['Pass/Fail'])
    input_dim = anomaly_df.shape[1]
    kl_divs = []
    for feature_idx in range(input_dim):
        anomaly_hist, bin_edges = np.histogram(anomaly_df.iloc[:, feature_idx], bins=bins, density=True)
        synthetic_hist, _ = np.histogram(synthetic_df.iloc[:, feature_idx], bins=bin_edges, density=True)
        anomaly_hist = np.where(anomaly_hist == 0, 1e-10, anomaly_hist)
        synthetic_hist = np.where(synthetic_hist == 0, 1e-10, synthetic_hist)
        kl_div = entropy(anomaly_hist, synthetic_hist)
        if np.isinf(kl_div) or np.isnan(kl_div):
            kl_div = 10
        kl_divs.append(kl_div)
    return np.mean(kl_divs), np.sum(kl_divs)

def js_divergence(anomaly_df, synthetic_df, bins=50):
    anomaly_df = anomaly_df.drop(columns=['Pass/Fail'])
    synthetic_df = synthetic_df.drop(columns=['Pass/Fail'])
    input_dim = anomaly_df.shape[1]
    js_divs = []
    for feature_idx in range(input_dim):
        anomaly_hist, bin_edges = np.histogram(anomaly_df.iloc[:, feature_idx], bins=bins, density=True)
        synthetic_hist, _ = np.histogram(synthetic_df.iloc[:, feature_idx], bins=bin_edges, density=True)
        anomaly_hist = np.where(anomaly_hist == 0, 1e-10, anomaly_hist)
        synthetic_hist = np.where(synthetic_hist == 0, 1e-10, synthetic_hist)
        m = (anomaly_hist + synthetic_hist) * 0.5
        js_div = (entropy(anomaly_hist, m) + entropy(synthetic_hist, m)) * 0.5
        js_divs.append(js_div)
    return np.mean(js_divs), np.sum(js_divs)

def compute_wasserstein_distance(anomaly_df, synthetic_df):
    anomaly_df = anomaly_df.drop(columns=['Pass/Fail'])
    synthetic_df = synthetic_df.drop(columns=['Pass/Fail'])
    input_dim = anomaly_df.shape[1]
    distances = []
    for feature_idx in range(input_dim):
        distance = wasserstein_distance(anomaly_df.iloc[:, feature_idx], synthetic_df.iloc[:, feature_idx])
        distances.append(distance)
    return np.mean(distances), np.sum(distances)

def maximum_mean_discrepancy(anomaly_df, synthetic_df, sigma=1):
    anomaly_df = anomaly_df.drop(columns=['Pass/Fail'])
    synthetic_df = synthetic_df.drop(columns=['Pass/Fail'])
    input_dim = anomaly_df.shape[1]
    m = anomaly_df.shape[0]
    n = synthetic_df.shape[0]
    mmds = []

    for feature_idx in range(input_dim):
        anomaly_feature = anomaly_df.iloc[:, feature_idx].values.reshape(-1, 1)
        synthetic_feature = synthetic_df.iloc[:, feature_idx].values.reshape(-1, 1)
        gamma = 1 / (2 * sigma**2)
        Kxx = rbf_kernel(anomaly_feature, anomaly_feature, gamma=gamma)
        Kyy = rbf_kernel(synthetic_feature, synthetic_feature, gamma=gamma)
        Kxy = rbf_kernel(anomaly_feature, synthetic_feature, gamma=gamma)

        np.fill_diagonal(Kxx, 0)  
        mean_Kxx = np.sum(Kxx) / (m * (m - 1))
        np.fill_diagonal(Kyy, 0)
        mean_Kyy = np.sum(Kyy) / (n * (n - 1))
        mean_Kxy = np.mean(Kxy)
        
        mmd_squared = mean_Kxx + mean_Kyy - 2 * mean_Kxy
        mmd = np.sqrt(max(mmd_squared, 0))
        mmds.append(mmd)
    return np.mean(mmds), np.sum(mmds) 

def plot_distance(total_kl_div_vae, total_js_div_vae, total_wd_vae, total_mmd_vae, 
                  total_kl_div_vaegan, total_js_div_vaegan, total_wd_vaegan, total_mmd_vaegan):
    metrics = ['KL Divergence', 'JS Divergence', 'Wasserstein Distance', 'MMD']
    vae_total = [total_kl_div_vae, total_js_div_vae, total_wd_vae, total_mmd_vae]
    vaegan_total = [total_kl_div_vaegan, total_js_div_vaegan, total_wd_vaegan, total_mmd_vaegan]
    x = np.arange(len(metrics))  
    width = 0.35  

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, vae_total, width, label='VAE', color='blue')
    rects2 = ax.bar(x + width/2, vaegan_total, width, label='VAE-GAN', color='green')
    ax.set_ylabel('Total Value')
    ax.set_title('Comparison of VAE and VAE-GAN Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

def main():
    ROOT = Path(__file__).parent
    anomaly_df_path = ROOT / "anomaly_data.csv"
    synthetic_vae_path = ROOT / "synthetic_vae.csv"
    synthetic_vaegan_path = ROOT / "synthetic_vaegan.csv"
    
    anomaly_df, synthetic_vae_df, synthetic_vaegan_df = load_data(anomaly_df_path, synthetic_vae_path, synthetic_vaegan_path)
    print("Load anomaly data.\n", anomaly_df.shape)
    print("Load synthetic VAE data.\n", synthetic_vae_df.shape)
    print("Load synthetic VAEGAN data.\n", synthetic_vaegan_df.shape)

    mean_kl_div_vae, total_kl_div_vae = kl_divergence(anomaly_df, synthetic_vae_df)
    print(f"Average KL Divergence for VAE (Histogram): {mean_kl_div_vae}")
    print(f"Total KL Divergence for VAE (Histogram): {total_kl_div_vae}")
    mean_kl_div_vaegan, total_kl_div_vaegan = kl_divergence(anomaly_df, synthetic_vaegan_df)
    print(f"Average KL Divergence for VAEGAN (Histogram): {mean_kl_div_vaegan}")
    print(f"Total KL Divergence for VAEGAN (Histogram): {total_kl_div_vaegan}")

    mean_js_div_vae, total_js_div_vae = js_divergence(anomaly_df, synthetic_vae_df)
    print(f"Average JS Divergence for VAE (Histogram): {mean_js_div_vae}")
    print(f"Total JS Divergence for VAE (Histogram): {total_js_div_vae}")
    mean_js_div_vaegan, total_js_div_vaegan = js_divergence(anomaly_df, synthetic_vaegan_df)
    print(f"Average JS Divergence for VAEGAN (Histogram): {mean_js_div_vaegan}")
    print(f"Total JS Divergence for VAEGAN (Histogram): {total_js_div_vaegan}")

    mean_wd_vae, total_wd_vae = compute_wasserstein_distance(anomaly_df, synthetic_vae_df)
    print(f"Average Wasserstein Distance for VAE : {mean_wd_vae}")
    print(f"Total Wasserstein Distance for VAE : {total_wd_vae}")
    mean_wd_vaegan, total_wd_vaegan = compute_wasserstein_distance(anomaly_df, synthetic_vaegan_df)
    print(f"Average Wasserstein Distance for VAEGAN : {mean_wd_vaegan}")
    print(f"Total Wasserstein Distance for VAEGAN : {total_wd_vaegan}")

    mean_mmd_vae, total_mmd_vae = maximum_mean_discrepancy(anomaly_df, synthetic_vae_df)
    print(f"Average Maximum Mean Discrepancy for VAE : {mean_mmd_vae}")
    print(f"Total Maximum Mean Discrepancy for VAE : {total_mmd_vae}")
    mean_mmd_vaegan, total_mmd_vaegan = maximum_mean_discrepancy(anomaly_df, synthetic_vaegan_df)
    print(f"Average Maximum Mean Discrepancy for VAEGAN : {mean_mmd_vaegan}")
    print(f"Total Maximum Mean Discrepancy for VAEGAN : {total_mmd_vaegan}")

    plot_distance(total_kl_div_vae, total_js_div_vae, total_wd_vae, total_mmd_vae, 
                  total_kl_div_vaegan, total_js_div_vaegan, total_wd_vaegan, total_mmd_vaegan)

if __name__ == "__main__":
    main()