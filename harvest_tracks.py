import os
import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

# 1. Load variables from .env
load_dotenv()

if not os.getenv("SPOTIPY_CLIENT_ID") or not os.getenv("SPOTIPY_CLIENT_SECRET"):
    raise ValueError("Missing SPOTIPY_CLIENT_ID or SPOTIPY_CLIENT_SECRET in your .env file.")

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    scope="user-library-read playlist-read-private playlist-read-collaborative"
))

tracks_data = []

def extract_track_info(track, source_name):
    """Safely extracts ID and metadata, skipping local files or podcasts."""
    if not track or not track.get("id") or track.get("is_local"):
        return
    tracks_data.append({
        "track_id": track["id"],
        "track_name": track["name"],
        "artist": ", ".join([a["name"] for a in track["artists"]]),
        "album": track["album"]["name"],
        "album_image": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
        "source": source_name
    })

# --- PART A: Harvest "Canciones que te gustan" (Liked Songs) ---
print("Fetching 'Canciones que te gustan' (Liked Songs)...")
results = sp.current_user_saved_tracks(limit=50)
while results:
    for entry in results["items"]:
        extract_track_info(entry.get("track") or entry.get("item"), "Canciones que te gustan")
    results = sp.next(results) if results["next"] else None

print(f"Collected {len(tracks_data)} tracks from Liked Songs.")

# --- PART B: Find and Harvest "This is Me" ---
print("Searching your playlists for 'This is Me'...")
target_playlist_id = None
playlists = sp.current_user_playlists(limit=50)

while playlists:
    for pl in playlists["items"]:
        if pl["name"].strip().lower() == "this is me":
            target_playlist_id = pl["id"]
            break
    if target_playlist_id or not playlists["next"]:
        break
    playlists = sp.next(playlists)

if target_playlist_id:
    print(f"Found 'This is Me' (ID: {target_playlist_id}). Fetching tracks...")
    before_count = len(tracks_data)
    
    # Call Spotify's 2026 /playlists/{id}/items endpoint directly
    pl_items = sp._get(f"playlists/{target_playlist_id}/items", limit=100)
    while pl_items:
        for entry in pl_items["items"]:
            # Support both 2026 schema ("item") and legacy schema ("track")
            track_obj = entry.get("item") or entry.get("track")
            extract_track_info(track_obj, "This is Me")
        pl_items = sp.next(pl_items) if pl_items["next"] else None
        
    print(f"Collected {len(tracks_data) - before_count} tracks from 'This is Me'.")
else:
    print("Warning: Could not find a playlist named 'This is Me' on your account.")

# --- PART C: Deduplicate and Save ---
df = pd.DataFrame(tracks_data)
print(f"Total tracks collected (with duplicates): {len(df)}")

df = df.drop_duplicates(subset=["track_id"]).reset_index(drop=True)
df.to_csv("my_spotify_tracks.csv", index=False)

print(f"Saved {len(df)} unique tracks to 'my_spotify_tracks.csv'!")