#Best parameters: {'latent_dim': 39, 'enc_learning_rate': 0.0001, 'dis_learning_rate': 1e-05, 'epochs': 1000, 'batch_size': 64, 'feature_layer_idx': 2}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA

from src.models.VAEGAN import VAEGAN
import tensorflow as tf
from pathlib import Path

def load_data(file_path):
    data = pd.read_csv(file_path, encoding = "utf-8")
    return data


def prepare_vae_data(df):
    fail_df = df[df['Pass/Fail']==1].select_dtypes(include=['float64','int64']).drop('Pass/Fail',axis=1)
    synthetic_num = df.shape[0] - len(fail_df) - len(fail_df)
    return fail_df, synthetic_num 


def generate_anomaly_data(fail_df, synthetic_num, data_standard):
    fail_data = fail_df.values
    input_dim = fail_data.shape[1]
    vae_gan = VAEGAN(input_dim=input_dim, latent_dim=32, feature_layer_idx=2)  
    vae_gan.compile(
        enc_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        dec_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)  
    )
    vae_gan.fit(fail_data, epochs=200, batch_size=64, verbose=1) 

    num_synthetic_samples = synthetic_num
    synthetic_anomalies = vae_gan.generate(num_synthetic_samples, scale=1)

    synthetic_df = pd.DataFrame(
        synthetic_anomalies,
        columns = data_standard.select_dtypes(include=['float64', 'int64']).drop(labels=['Pass/Fail'], axis=1).columns
    )
    synthetic_df['Pass/Fail'] = 1
    return synthetic_df


def KDE_plot(fail_df, synthetic_df):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))  
    axes = axes.ravel()                               

    for feature_idx in range(4):
        ax = axes[feature_idx]                        

        ax.hist(fail_df.iloc[:, feature_idx],
                bins=50, density=True, alpha=0.5,
                label='Real Anomalies',  color='red')
        ax.hist(synthetic_df.iloc[:, feature_idx],
                bins=50, density=True, alpha=0.5,
                label='Generated Anomalies', color='blue')

        sns.kdeplot(fail_df.iloc[:, feature_idx],
                     label='Real Anomalies', color='red',  ax=ax)
        sns.kdeplot(synthetic_df.iloc[:, feature_idx],
                     label='Generated Anomalies', color='blue', ax=ax)

        ax.set_title(f'Feature {feature_idx}: Real vs Generated', fontsize=11)
        ax.set_xlabel('Feature Value')
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def pca_check_generalize(fail_df, synthetic_df, synthetic_num):
    synthetic_df = synthetic_df.drop(columns = ['Pass/Fail'])
    synthetic_data = synthetic_df.values
    fail_data = fail_df.values
    combined_data = np.vstack([synthetic_data, fail_data])
    kpca = KernelPCA(n_components=2, kernel='rbf')
    combined_pca = kpca.fit_transform(combined_data)
    synthetic_pca = combined_pca[:synthetic_num]  
    reserved_pca = combined_pca[synthetic_num:]
    plt.figure(figsize=(8, 6))
    plt.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], label="Generated Anomalies", alpha=0.5, color="blue")
    plt.scatter(reserved_pca[:, 0], reserved_pca[:, 1], label="Real Anomalies", alpha=0.8, color="red", marker="x")
    plt.title("KernelPCA Distribution of Generated vs. Real Anomalies")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_distribution(real_data, synthetic_data):
    for feature_idx in range(5): 
        plt.figure(figsize=(8, 6))
        plt.hist(real_data.iloc[:, feature_idx], bins=50, density=True, alpha=0.5, label='Real Data', color='red')
        plt.hist(synthetic_data.iloc[:, feature_idx], bins=50, density=True, alpha=0.5, label='Synthetic Data', color='blue')
        plt.title(f'Distribution of Feature {feature_idx}')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        plt.legend()
        plt.show()



def main():
    ROOT = Path(__file__).parent
    file_path = ROOT / "uci-secom-cleaned.csv"
    synthetic_path = ROOT / "synthetic_vaegan.csv"
    combined_path = ROOT / "combined_data_vaegan.csv"

    data = load_data(file_path)
    print("Data loaded. Shape:\n", data.shape)

    fail_df, synthetic_num = prepare_vae_data(data)
    print("Train anomaly data shape:\n", fail_df.shape)

    synthetic_num = int(synthetic_num * 1)
    
    synthetic_df = generate_anomaly_data(fail_df, synthetic_num, data)
    print("Synthetic anomaly data shape:\n", synthetic_df.shape)
    print(synthetic_df.head())

    synthetic_df.to_csv(synthetic_path, index = False)

    KDE_plot(fail_df, synthetic_df)

    pca_check_generalize(fail_df, synthetic_df, synthetic_num)
    # plot_distribution(data, synthetic_df)

    combined_df = pd.concat([data, synthetic_df], ignore_index=True)
    print("Combined data shape:\n", combined_df.shape)

    combined_df.to_csv(combined_path, index=False)
    print("Combined data saved to 'combined_data_vaegan.csv'")


if __name__ == "__main__":
    main()