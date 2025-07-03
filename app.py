import streamlit as st
import sys
import os

# Add src to system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Imports from src
from preprocess import load_ratings, load_movies, create_user_item_matrix
from similarity import compute_user_similarity, compute_item_similarity
from recommend import user_based_recommendations, item_based_recommendations
from posters import get_movie_poster

# Load data
ratings = load_ratings("data/ratings.csv")
movies = load_movies("data/movies.csv")
user_item_matrix = create_user_item_matrix(ratings)

# Compute similarities
user_sim_matrix = compute_user_similarity(user_item_matrix)
item_sim_matrix = compute_item_similarity(user_item_matrix)

# ---------- 🎬 UI Template ----------
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System")
st.markdown("""
Welcome to the Movie Recommendation Engine powered by Collaborative Filtering.  
Select a tab below to get personalized or similar movie suggestions!
""")

# ---------- 🔄 Tabs ----------
tab1, tab2 = st.tabs(["👤 User-Based", "🎞 Item-Based"])

# ---------- Tab 1: User-Based Filtering ----------
with tab1:
    st.subheader("👤 Recommend Movies for a User")
    user_ids = user_item_matrix.index.tolist()
    user_id = st.selectbox("Choose User ID", user_ids)
    n_recs = st.slider("Number of Recommendations", 1, 20, 5)

    if st.button("Get User-Based Recommendations"):
        rec_ids = user_based_recommendations(user_id, user_item_matrix, user_sim_matrix, n=n_recs)
        recommended_movies = movies[movies["movieId"].isin(rec_ids)]

        if not recommended_movies.empty:
            st.success("Here are your recommendations:")
            cols = st.columns(3)  # 3 posters per row
            for i, row in enumerate(recommended_movies.itertuples()):
                poster_url = get_movie_poster(row.title)
                with cols[i % 3]:
                    if poster_url:
                        st.image(poster_url, width=150)
                    else:
                        st.image("https://via.placeholder.com/150x225?text=No+Image", width=150)
                    st.markdown(f"**🎬 {row.title}**")
        else:
            st.warning("No recommendations found for this user.")

# ---------- Tab 2: Item-Based Filtering ----------
with tab2:
    st.subheader("🎞 Recommend Similar Movies")
    movie_title = st.selectbox("Pick a movie you like", sorted(movies["title"].unique()))
    n_recs_item = st.slider("Number of Similar Movies", 1, 20, 5, key="item_slider")

    if st.button("Get Similar Movies"):
        similar_ids = item_based_recommendations(movie_title, user_item_matrix, movies, item_sim_matrix, n=n_recs_item)
        similar_movies = movies[movies["movieId"].isin(similar_ids)]

        if not similar_movies.empty:
            st.success(f"Movies similar to *{movie_title}*:")
            cols = st.columns(3)  # 3 posters per row
            for i, row in enumerate(similar_movies.itertuples()):
                poster_url = get_movie_poster(row.title)
                with cols[i % 3]:
                    if poster_url:
                        st.image(poster_url, width=150)
                    else:
                        st.image("https://via.placeholder.com/150x225?text=No+Image", width=150)
                    st.markdown(f"**🎬 {row.title}**")
        else:
            st.warning("No similar movies found.")
