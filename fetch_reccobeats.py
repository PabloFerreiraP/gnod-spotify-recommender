import time
import pandas as pd
import requests

# 1. Load the tracks collected from Spotify
df_spotify = pd.read_csv("my_spotify_tracks.csv")
print(f"Initial tracks in CSV: {len(df_spotify)}")

# Extra cleanup: remove exact duplicates of (track_name + artist)
df_spotify = df_spotify.drop_duplicates(subset=["track_name", "artist"]).reset_index(drop=True)
print(f"Tracks after removing duplicates by name and artist: {len(df_spotify)}")

# 2. Query ReccoBeats in batches of 40 IDs
track_ids = df_spotify["track_id"].tolist()
batch_size = 40
recco_features = []

print("Fetching audio features from ReccoBeats...")
for i in range(0, len(track_ids), batch_size):
    batch = track_ids[i : i + batch_size]
    ids_param = ",".join(batch)
    
    try:
        response = requests.get(
            "https://api.reccobeats.com/v1/audio-features",
            params={"ids": ids_param},
            timeout=15
        )
        if response.status_code == 200:
            items = response.json().get("content", [])
            recco_features.extend(items)
        else:
            print(f"Warning in batch {i}: status {response.status_code}")
    except Exception as e:
        print(f"Error in batch {i}: {e}")
    
    # Short pause to avoid overloading the free API
    time.sleep(0.3)

df_features = pd.DataFrame(recco_features)
print(f"Tracks found in ReccoBeats: {len(df_features)}")

# 3. Extract the Spotify ID from ReccoBeats' 'href' field and merge tables
# ReccoBeats returns 'href' as 'https://open.spotify.com/track/<spotify_id>'
df_features["track_id"] = df_features["href"].str.split("/").str[-1]

feature_cols = [
    "track_id", "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "loudness", "speechiness",
    "key", "mode"
]
df_features = df_features[feature_cols].drop_duplicates(subset=["track_id"])

# Inner merge to keep only tracks that have both metadata and audio features
df_final = pd.merge(df_spotify, df_features, on="track_id", how="inner").dropna()
df_final.to_csv("music_database.csv", index=False)

print(f"Done! Final database saved to 'music_database.csv' with {len(df_final)} complete tracks.")