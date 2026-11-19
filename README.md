# 🎧 Spotify Vibe Recommender

An end-to-end unsupervised Machine Learning music recommender built with **Python**, **Spotipy**, **ReccoBeats API**, **Scikit-Learn**, and **Streamlit**. Search for any song on Spotify and get the **Top 10 closest sonic matches** from a curated personal music library.

## 🚀 How It Works

1. **Data Collection (`harvest_tracks.py`):** Extracts personal saved tracks and playlists via the Spotify Web API (`Spotipy`), deduplicating entries into a unified catalog.
2. **Audio Feature Enrichment (`fetch_reccobeats.py`):** Queries the ReccoBeats API to retrieve 10 core acoustic descriptors (`danceability`, `energy`, `valence`, `tempo`, `acousticness`, `instrumentalness`, `loudness`, `speechiness`, `key`, and `mode`).
3. **Feature Engineering & Model Training (`train_model.py`):**
   - Applies **circular encoding** ($\sin$ and $\cos$) to musical `key` to preserve harmonic continuity.
   - Standardizes all **11 features** using `StandardScaler`.
   - Trains an unsupervised **K-Means ($K=8$)** model selected after benchmarking against Agglomerative Clustering and DBSCAN.
4. **Real-Time Matching (`app.py`):** Fetches live audio features for any user-selected track, assigns it to its matching K-Means cluster, and ranks candidates inside that cluster using **Euclidean distance (`L2`)**.

## 🛠️ Tech Stack
* **Data & APIs:** Python, Pandas, NumPy, Spotipy (Spotify Web API), ReccoBeats API
* **Machine Learning:** Scikit-Learn (`StandardScaler`, `KMeans`), SciPy (`cdist`), Joblib
* **Frontend:** Streamlit, `streamlit-searchbox`

## ⚙️ Local Setup

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
