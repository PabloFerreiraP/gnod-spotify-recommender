import os
import joblib
import numpy as np
import pandas as pd
import requests
import spotipy
import streamlit as st
from dotenv import load_dotenv
from scipy.spatial.distance import cdist
from spotipy.oauth2 import SpotifyClientCredentials
from streamlit_searchbox import st_searchbox

load_dotenv()

st.set_page_config(page_title="Spotify Vibe Recommender", page_icon="🎧", layout="wide")

# --- SPOTIFY DARK UI THEME (CSS) ---
st.markdown("""
<style>
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
    }
    header[data-testid="stHeader"] {
        background-color: rgba(18, 18, 18, 0.85);
    }
    h1, h2, h3, p, label, li, .stMarkdown {
        color: #FFFFFF !important;
        font-family: 'Circular', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* Pestañas estilo Spotify */
    button[data-baseweb="tab"] {
        color: #B3B3B3 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        background-color: transparent !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1DB954 !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #1DB954 !important;
    }
    div[data-baseweb="tab-border"] {
        background-color: #282828 !important;
    }

    /* Spotify green pill button */
    div.stButton > button[kind="primary"] {
        background-color: #1DB954 !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 500px !important;
        padding: 0.6rem 2rem !important;
        letter-spacing: 0.5px;
        margin-top: 8px;
        transition: transform 0.15s ease, background-color 0.15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1ED760 !important;
        transform: scale(1.03);
    }

    /* Top-right Spotify profile pill */
    .profile-pill {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background-color: #000000;
        padding: 4px 14px 4px 6px;
        border-radius: 500px;
        border: 1px solid #282828;
        transition: background-color 0.2s ease;
    }
    .profile-pill:hover {
        background-color: #282828;
    }

    /* Tarjetas explicativas estilo Spotify */
    .info-card {
        background-color: #181818;
        border: 1px solid #282828;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 18px;
        transition: background-color 0.2s ease;
    }
    .info-card:hover {
        background-color: #202020;
    }
    .step-badge {
        display: inline-block;
        background-color: #1DB954;
        color: #000000;
        font-weight: 800;
        font-size: 0.78rem;
        padding: 3px 12px;
        border-radius: 500px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 10px;
    }
    .stat-box {
        background-color: #181818;
        border: 1px solid #282828;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #1DB954;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #B3B3B3;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "loudness", "speechiness",
    "mode", "key_sin", "key_cos"
]

SPOTIFY_USER_ID = "Pablowy"

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    clean_key = df["key"].replace(-1, 0)
    df["key_sin"] = np.sin(2 * np.pi * clean_key / 12)
    df["key_cos"] = np.cos(2 * np.pi * clean_key / 12)
    return df[FEATURES]

# 1. Initialize Spotipy
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
))

# 2. Load database, scaler & trained K-Means (K=8) model
@st.cache_resource
def load_assets():
    df = pd.read_csv("music_clustered.csv")
    scaler = joblib.load("scaler.joblib")
    kmeans = joblib.load("kmeans.joblib")
    X_scaled = scaler.transform(prepare_features(df))
    return df, scaler, kmeans, X_scaled

@st.cache_data(ttl=3600)
def get_user_profile(user_id: str):
    try:
        user = sp.user(user_id)
        name = user.get("display_name") or user_id
        url = user.get("external_urls", {}).get("spotify", f"https://open.spotify.com/user/{user_id}")
        images = user.get("images", [])
        avatar = images[0]["url"] if images else "https://upload.wikimedia.org/wikipedia/commons/8/84/Spotify_icon.svg"
        return name, url, avatar
    except Exception:
        return user_id, f"https://open.spotify.com/user/{user_id}", "https://upload.wikimedia.org/wikipedia/commons/8/84/Spotify_icon.svg"

df, scaler, kmeans, X_db_scaled = load_assets()
profile_name, profile_url, profile_avatar = get_user_profile(SPOTIFY_USER_ID)

def get_live_features(track_id: str):
    resp = requests.get(
        "https://api.reccobeats.com/v1/audio-features",
        params={"ids": track_id},
        timeout=10
    )
    if resp.status_code == 200:
        content = resp.json().get("content", [])
        if content:
            return content[0]
    return None

def search_spotify_tracks(searchterm: str):
    if not searchterm or len(searchterm.strip()) < 2:
        return []
    results = sp.search(q=searchterm, type="track", limit=8)
    tracks = results["tracks"]["items"]
    suggestions = []
    for t in tracks:
        artists = ", ".join(a["name"] for a in t["artists"])
        album = t["album"]["name"]
        label = f"{t['name']} — {artists} ({album})"
        suggestions.append((label, t))
    return suggestions

# 3. Spotify-style Header
header_left, header_right = st.columns([4, 1])

with header_left:
    st.markdown("<h1 style='margin-bottom: 4px;'>🎧 May I recommend you a song?</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #B3B3B3 !important; font-size: 1.05rem;'>Search any song you like, and I will recommend you <b>the 10 closest matches</b> of my <b>favourite songs</b>.</p>", unsafe_allow_html=True)

with header_right:
    st.markdown(
        f"""
        <div style="display: flex; justify-content: flex-end; padding-top: 14px;">
            <a href="{profile_url}" target="_blank" style="text-decoration: none;">
                <div class="profile-pill">
                    <img src="{profile_avatar}" style="width: 34px; height: 34px; border-radius: 50%; object-fit: cover;" />
                    <span style="color: #FFFFFF; font-weight: 700; font-size: 0.9rem;">{profile_name} ↗</span>
                </div>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# 4. Navigation Tabs
tab_recommender, tab_how_it_works = st.tabs(["🎧 Song Recommender", "⚙️ How It Works"])

# ==========================================
# TAB 1: SONG RECOMMENDER
# ==========================================
with tab_recommender:
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("**Which song do you like?**")
    selected_track = st_searchbox(
        search_spotify_tracks,
        placeholder="Type a song or artist and pick the exact track...",
        key="spotify_single_searchbar",
        clear_on_submit=False,
    )

    if selected_track:
        if st.button("Recommend Songs", type="primary"):
            with st.spinner("Analyzing audio features & finding your matches..."):
                raw_features = get_live_features(selected_track["id"])

            if not raw_features:
                st.error("Sorry! ReccoBeats doesn't have audio features for this specific track version yet. Try selecting another version or song from the search bar.")
            else:
                seed_df = prepare_features(pd.DataFrame([raw_features]))
                seed_scaled = scaler.transform(seed_df)

                # Predict K-Means cluster (K=8) and rank closest matches inside that cluster
                seed_cluster = int(kmeans.predict(seed_scaled)[0])

                distances = cdist(seed_scaled, X_db_scaled, metric="euclidean")[0]
                rec_df = df.copy()
                rec_df["euclidean_distance"] = distances

                rec_df = rec_df[
                    (rec_df["cluster"] == seed_cluster) &
                    (rec_df["track_id"] != selected_track["id"])
                ]
                top_10 = rec_df.sort_values("euclidean_distance").head(10).reset_index(drop=True)

                st.subheader(f"Top 10 Closest Matches (Vibe Cluster #{seed_cluster})")

                for _, row in top_10.iterrows():
                    st.components.v1.iframe(
                        src=f"https://open.spotify.com/embed/track/{row['track_id']}?utm_source=generator&theme=0",
                        height=80,
                        scrolling=False
                    )

# ==========================================
# TAB 2: HOW IT WORKS (ARCHITECTURE & PIPELINE)
# ==========================================
with tab_how_it_works:
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.subheader("How This Recommender Was Built")
    st.markdown(
        "<p style='color: #B3B3B3 !important; font-size: 1rem;'>"
        "This tool combines live music search with <b>Unsupervised Machine Learning (K-Means, K=8)</b> "
        "and intra-cluster Euclidean distance metrics to match any track in the world against a my personal library."
        "</p>",
        unsafe_allow_html=True
    )

    # Key Dataset Stats
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='stat-box'><div class='stat-number'>{len(df):,}</div><div class='stat-label'>Spotify Tracks in DB</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='stat-box'><div class='stat-number'>11</div><div class='stat-label'>ReccoBeats Audio Features</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='stat-box'><div class='stat-number'>8</div><div class='stat-label'>K-Means Clusters</div></div>", unsafe_allow_html=True)


    st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)

    # Step 1, Step 2, Step 3 & Step 4
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="info-card">
            <span class="step-badge">Step 1 • Data Collection</span>
            <h3 style="margin-top: 4px;">Building a Unified Personal Music Database</h3>
            <p style="color: #B3B3B3 !important;">
                Using <b>Spotipy</b> (Python wrapper for the Spotify Web API), I extracted the tracks from two personal playlists:
            </p>
            <ul style="color: #E0E0E0;">
                <li><b>"Liked Songs"</b> (Liked Songs library via <code>current_user_saved_tracks</code>)</li>
                <li><b>"This is Me"</b> (Personal playlist via <code>playlists/{id}/items</code>)</li>
            </ul>
            <p style="color: #B3B3B3 !important; margin-bottom: 0;">
                After removing duplicates, both playlists are merged into a single unified catalog so recommendations match sound similarity regardless of playlist source.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <span class="step-badge">Step 3 • Feature Engineering & Scaling</span>
            <h3 style="margin-top: 4px;">Preparing Musical Attributes</h3>
            <p style="color: #B3B3B3 !important;">
                Raw audio attributes live on different scales (e.g., <i>tempo</i> in BPM vs. <i>energy</i> in 0–1). Two key transformations were applied:
            </p>
            <ul style="color: #E0E0E0;">
                <li><b>Circular Key Encoding:</b> Pitch classes (<code>0 = C</code> to <code>11 = B</code>) are encoded via <code>sin(2π·key/12)</code> and <code>cos(2π·key/12)</code> to preserve harmonic continuity.</li>
                <li><b>StandardScaler (Z-score):</b> Centers all 11 features (<code>danceability</code>, <code>energy</code>, <code>valence</code>, <code>tempo</code>, <code>acousticness</code>, <code>instrumentalness</code>, <code>loudness</code>, <code>speechiness</code>, <code>mode</code>, <code>key_sin</code>, <code>key_cos</code>) to mean <code>0</code> and standard deviation <code>1</code>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="info-card">
            <span class="step-badge">Step 2 • Database Enrichment</span>
            <h3 style="margin-top: 4px;">Including ReccoBeats Audio-Features data</h3>
            <p style="color: #B3B3B3 !important;">
                The <b>ReccoBeats API</b> was integrated to extract acoustic descriptors for the catalog previously downloaded:
            </p>
            <ul style="color: #E0E0E0;">
                <li><b>Rhythm & Dynamics:</b> <code>danceability</code>, <code>energy</code>, <code>tempo</code>, <code>loudness</code></li>
                <li><b>Mood & Timbre:</b> <code>valence</code>, <code>acousticness</code>, <code>instrumentalness</code>, <code>speechiness</code></li>
                <li><b>Harmony:</b> <code>key</code> and <code>mode</code> (Major vs. Minor)</li>
            </ul>
            <p style="color: #B3B3B3 !important; margin-bottom: 0;">
                Merging Spotify metadata with ReccoBeats audio vectors produced the final structured dataset.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <span class="step-badge">Step 4 • Model Selection & Live Matching</span>
            <h3 style="margin-top: 4px;">K-Means (K=8) & Closest-Cluster Euclidean Search</h3>
            <p style="color: #B3B3B3 !important;">
                After benchmarking K-Means, Agglomerative Clustering (Ward, Complete, Average), and DBSCAN across Silhouette, Calinski-Harabasz, and Davies-Bouldin metrics, <b>K-Means with K=8</b> was selected for offering balanced sonic clusters with zero noise:
            </p>
            <ul style="color: #E0E0E0;">
                <li><b>Real-Time Feature Extraction:</b> Spotipy searches for the exact Spotify Track ID, and ReccoBeats fetches its live audio features.</li>
                <li><b>Cluster Assignment:</b> The saved <code>StandardScaler</code> features and the <code>K-Means (K=8)</code> model projects the songs into a 11D space and assign it to its matching cluster.</li>
                <li><b>Intra-Cluster Ranking (<code>closest_cluster</code>):</b> <b>Euclidean Distance</b> (<code>scipy.spatial.distance.cdist</code>) ranks all candidate tracks inside that predicted cluster (excluding the query song itself) to return the <b>Top 10 nearest neighbors</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Mathematical formula & Stack summary
    st.markdown(
        "<p style='color: #B3B3B3 !important;'>"
        "<b>Tech Stack:</b> Python • Spotipy (Spotify Web API) • ReccoBeats API • Pandas & NumPy • Scikit-Learn (StandardScaler & K-Means K=8) • SciPy • Joblib • Streamlit"
        "</p>",
        unsafe_allow_html=True
    )