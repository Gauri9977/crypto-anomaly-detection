import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import shap
import zipfile
from google.colab import files

# =========================
# 1. Load Dataset Manually
# =========================

# Upload ZIP file manually (for Google Colab)
uploaded = files.upload()

# Get uploaded file name
zip_file_path = next(iter(uploaded))
extract_path = "crypto_data"

# Extract dataset
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Identify the subfolder containing CSV files
crypto_coins_path = os.path.join(extract_path, "Top 100 Crypto Coins")

# Debug: List extracted files
if os.path.exists(crypto_coins_path):
    extracted_files = os.listdir(crypto_coins_path)
    print("Extracted CSV files:", extracted_files)

    # Load the first CSV file
    first_csv = next((f for f in extracted_files if f.endswith(".csv")), None)
    if first_csv:
        csv_file_path = os.path.join(crypto_coins_path, first_csv)
        crypto_df = pd.read_csv(csv_file_path)
        print("Dataset Preview:")
        print(crypto_df.head())
    else:
        print("Error: No CSV files found in the extracted folder.")
else:
    print("Error: 'Top 100 Crypto Coins' folder not found in extracted ZIP.")

# ============================
# 2. Feature Engineering
# ============================

def calculate_technical_indicators(df):
    """Calculate technical indicators such as RSI and Bollinger Bands."""
    df['RSI'] = df['Close'].diff().apply(lambda x: x if x > 0 else 0).rolling(14).mean() / df['Close'].diff().abs().rolling(14).mean()
    df['Rolling_Mean'] = df['Close'].rolling(window=20).mean()
    df['Rolling_Std'] = df['Close'].rolling(window=20).std()
    df['Upper_BB'] = df['Rolling_Mean'] + (df['Rolling_Std'] * 2)
    df['Lower_BB'] = df['Rolling_Mean'] - (df['Rolling_Std'] * 2)
    return df

# Apply feature engineering
crypto_df = calculate_technical_indicators(crypto_df)
crypto_df = crypto_df.dropna()

# ============================
# 3. Data Preprocessing
# ============================

# Select relevant features
features = ['Close', 'Volume', 'RSI', 'Rolling_Mean', 'Upper_BB', 'Lower_BB']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(crypto_df[features])

# ============================
# 4. Anomaly Detection Methods
# ============================

# ---- Isolation Forest ----
print("Running Isolation Forest...")
iso_forest = IsolationForest(contamination=0.02, random_state=42)
crypto_df['anomaly_iso'] = iso_forest.fit_predict(scaled_data)

# ---- Autoencoder ----
print("Training Autoencoder...")
input_dim = scaled_data.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(10, activation='relu')(input_layer)
encoded = Dense(5, activation='relu')(encoded)
decoded = Dense(10, activation='relu')(encoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(scaled_data, scaled_data, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Compute reconstruction errors
reconstructions = autoencoder.predict(scaled_data)
mse = np.mean(np.abs(reconstructions - scaled_data), axis=1)
threshold = np.percentile(mse, 95)
crypto_df['anomaly_auto'] = (mse > threshold).astype(int)

# ---- Z-Score Method ----
def detect_anomalies_zscore(df, column, threshold=3):
    """Detect anomalies using the Z-Score method."""
    mean = df[column].mean()
    std = df[column].std()
    df['z_score'] = (df[column] - mean) / std
    df['anomaly_zscore'] = (abs(df['z_score']) > threshold).astype(int)
    return df

crypto_df = detect_anomalies_zscore(crypto_df, 'Close')

# ============================
# 5. Model Comparison
# ============================

def compare_anomalies(df):
    """Compare anomaly detection results from different models."""
    comparison = df[['anomaly_iso', 'anomaly_auto', 'anomaly_zscore']].sum()
    return pd.DataFrame({'Method': comparison.index, 'Anomalies Detected': comparison.values})

comparison_table = compare_anomalies(crypto_df)
print("\nAnomaly Detection Comparison:")
print(comparison_table)

# ============================
# 6. Feature Importance (SHAP)
# ============================
print("Computing SHAP values...")
explainer = shap.Explainer(iso_forest)
shap_values = explainer(scaled_data)
shap.summary_plot(shap_values, features)

# ============================
# 7. Data Visualization
# ============================

# Scatter plot for anomaly detection
plt.figure(figsize=(12,6))
sns.scatterplot(x=crypto_df.index, y=crypto_df['Close'], hue=crypto_df['anomaly_auto'], palette={0:'blue', 1:'red'})
plt.title("Anomaly Detection in Crypto Prices using Autoencoder")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.show()

# Heatmap of feature correlations
plt.figure(figsize=(10,6))
sns.heatmap(crypto_df[features].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Pairplot for better insights
sns.pairplot(crypto_df, hue='anomaly_iso', diag_kind='kde')
plt.title("Pairplot of Crypto Features with Anomalies")
plt.show()
