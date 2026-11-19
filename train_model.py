import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load the clean database
df = pd.read_csv("music_database.csv")

import numpy as np

FEATURES = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "loudness", "speechiness",
    "mode", "key_sin", "key_cos"
]

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Handle undetected key (-1) safely
    clean_key = df["key"].replace(-1, 0)
    df["key_sin"] = np.sin(2 * np.pi * clean_key / 12)
    df["key_cos"] = np.cos(2 * np.pi * clean_key / 12)
    return df[FEATURES]

# 2. Scale features (essential for Euclidean distance and K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(prepare_features(df))

# 3. Train K-Means unsupervised model
kmeans = KMeans(n_clusters=8, random_state=42, n_init="auto")
df["cluster"] = kmeans.fit_predict(X_scaled)

# 4. Save clustered dataset and fitted models
df.to_csv("music_clustered.csv", index=False)
joblib.dump(scaler, "scaler.joblib")
joblib.dump(kmeans, "kmeans.joblib")

print("Done! Saved 'music_clustered.csv', 'scaler.joblib', and 'kmeans.joblib'.")
print("\nSongs per cluster:")
print(df["cluster"].value_counts().sort_index())