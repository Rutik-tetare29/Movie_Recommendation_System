# enhanced_app.py

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Add src to system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Imports from src
from preprocess import load_ratings, load_movies, create_user_item_matrix
from similarity import compute_user_similarity, compute_item_similarity
from recommend import user_based_recommendations, item_based_recommendations
from posters import get_movie_poster
from matrix_factorization import MatrixFactorization, compare_matrix_factorization_methods
from database import MovieRecommenderDB, initialize_database_with_csv
from hybrid_recommender import HybridRecommendationSystem, create_hybrid_system
from ab_testing import ABTestingFramework, create_simple_ab_test

# ---------- 🔧 Configuration ----------
st.set_page_config(
    page_title="Advanced Movie Recommender", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- 💾 Initialize Database and Models ----------
@st.cache_resource
def initialize_system():
    """Initialize the complete recommendation system"""
    with st.spinner("🚀 Initializing Advanced Recommendation System..."):
        # Initialize database
        db = initialize_database_with_csv()
        
        # Load data
        user_item_matrix = db.get_user_item_matrix()
        movies_df = db.get_movies_info()
        
        # Compute traditional similarities
        user_sim_matrix = compute_user_similarity(user_item_matrix)
        item_sim_matrix = compute_item_similarity(user_item_matrix)
        
        # Initialize Matrix Factorization models
        svd_model = MatrixFactorization(method='svd', n_components=50)
        svd_model.fit(user_item_matrix)
        
        nmf_model = MatrixFactorization(method='nmf', n_components=50)
        nmf_model.fit(user_item_matrix)
        
        # Create Hybrid System
        hybrid_system = create_hybrid_system(
            user_item_matrix, 
            movies_df,
            mf_model=svd_model
        )
        
        # Initialize A/B Testing Framework
        ab_framework = ABTestingFramework(db)
        
        return {
            'db': db,
            'user_item_matrix': user_item_matrix,
            'movies_df': movies_df,
            'user_sim_matrix': user_sim_matrix,
            'item_sim_matrix': item_sim_matrix,
            'svd_model': svd_model,
            'nmf_model': nmf_model,
            'hybrid_system': hybrid_system,
            'ab_framework': ab_framework
        }

# Initialize system
system = initialize_system()

# ---------- 🎨 UI Design ----------
st.title("🎬 Advanced Movie Recommendation System")
st.markdown("""
### 🚀 Next-Generation Movie Discovery Engine
*Powered by Matrix Factorization, Hybrid Algorithms, and Real-time A/B Testing*
""")

# ---------- 📊 Sidebar Controls ----------
st.sidebar.header("🎛️ Control Panel")

# Algorithm Selection
algorithm_choice = st.sidebar.selectbox(
    "🧠 Choose Recommendation Algorithm",
    ["Collaborative Filtering", "Matrix Factorization (SVD)", "Matrix Factorization (NMF)", 
     "Hybrid System", "A/B Testing"]
)

# User Selection
user_ids = system['user_item_matrix'].index.tolist()
selected_user = st.sidebar.selectbox("👤 Select User ID", user_ids)

# Number of recommendations
n_recommendations = st.sidebar.slider("📈 Number of Recommendations", 1, 20, 10)

# Advanced Options
with st.sidebar.expander("⚙️ Advanced Options"):
    show_explanations = st.checkbox("📝 Show Recommendation Explanations", value=True)
    show_diversity_score = st.checkbox("🎯 Calculate Diversity Score", value=True)
    cache_recommendations = st.checkbox("💾 Cache Recommendations", value=True)

# ---------- 📈 System Statistics ----------
with st.expander("📊 System Statistics", expanded=False):
    stats = system['db'].get_system_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Total Users", stats['total_users'])
    with col2:
        st.metric("🎬 Total Movies", stats['total_movies'])
    with col3:
        st.metric("⭐ Total Ratings", stats['total_ratings'])
    with col4:
        st.metric("🔄 Interactions", stats['total_interactions'])
    
    # Rating Distribution Chart
    if stats['rating_distribution']:
        fig = px.bar(
            x=list(stats['rating_distribution'].keys()),
            y=list(stats['rating_distribution'].values()),
            title="Rating Distribution",
            labels={'x': 'Rating', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------- 🔥 Main Content ----------

if algorithm_choice == "Collaborative Filtering":
    st.header("👥 Collaborative Filtering")
    
    tab1, tab2 = st.tabs(["🧑‍🤝‍🧑 User-Based", "🎞️ Item-Based"])
    
    with tab1:
        st.subheader("User-Based Collaborative Filtering")
        
        if st.button("🎯 Get User-Based Recommendations", key="user_cf"):
            # Check cache first
            cached = system['db'].get_cached_recommendations(selected_user, 'user_collaborative')
            
            if cached and cache_recommendations:
                st.info("📋 Retrieved from cache")
                rec_ids = [rec[0] for rec in cached['recommendations']]
            else:
                rec_ids = user_based_recommendations(
                    selected_user, system['user_item_matrix'], 
                    system['user_sim_matrix'], n=n_recommendations
                )
                
                # Cache recommendations
                if cache_recommendations:
                    rec_data = [(movie_id, 1.0) for movie_id in rec_ids]  # Dummy scores
                    system['db'].cache_recommendations(selected_user, 'user_collaborative', rec_data)
            
            if rec_ids:
                recommended_movies = system['movies_df'][system['movies_df']["movieId"].isin(rec_ids)]
                
                # Display recommendations
                cols = st.columns(3)
                for i, row in enumerate(recommended_movies.itertuples()):
                    with cols[i % 3]:
                        poster_url = get_movie_poster(row.title)
                        if poster_url:
                            st.image(poster_url, width=150)
                        else:
                            st.image("https://via.placeholder.com/150x225?text=No+Image", width=150)
                        st.markdown(f"**🎬 {row.title}**")
                        
                        # User interaction buttons
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("👍", key=f"like_user_{i}"):
                                system['db'].log_user_interaction(selected_user, row.movieId, 'like')
                                st.success("Liked!")
                        with col_b:
                            if st.button("👁️", key=f"view_user_{i}"):
                                system['db'].log_user_interaction(selected_user, row.movieId, 'view')
                                st.info("Viewed!")
            else:
                st.warning("No recommendations found for this user.")
    
    with tab2:
        st.subheader("Item-Based Collaborative Filtering")
        movie_title = st.selectbox("🎬 Pick a movie you like", sorted(system['movies_df']["title"].unique()))
        
        if st.button("🔍 Find Similar Movies", key="item_cf"):
            similar_ids = item_based_recommendations(
                movie_title, system['user_item_matrix'], 
                system['movies_df'], system['item_sim_matrix'], n=n_recommendations
            )
            
            if similar_ids:
                similar_movies = system['movies_df'][system['movies_df']["movieId"].isin(similar_ids)]
                
                cols = st.columns(3)
                for i, row in enumerate(similar_movies.itertuples()):
                    with cols[i % 3]:
                        poster_url = get_movie_poster(row.title)
                        if poster_url:
                            st.image(poster_url, width=150)
                        else:
                            st.image("https://via.placeholder.com/150x225?text=No+Image", width=150)
                        st.markdown(f"**🎬 {row.title}**")

elif algorithm_choice == "Matrix Factorization (SVD)":
    st.header("🔢 Matrix Factorization (SVD)")
    
    # Model Information
    model_info = system['svd_model'].get_model_info()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🧮 Components", model_info['n_components'])
    with col2:
        st.metric("👥 Users", model_info['n_users'])
    with col3:
        st.metric("🎬 Movies", model_info['n_items'])
    
    if st.button("🎯 Get SVD Recommendations", key="svd_recs"):
        recommendations = system['svd_model'].get_user_recommendations(selected_user, n_recommendations)
        
        if recommendations:
            # Cache recommendations
            if cache_recommendations:
                system['db'].cache_recommendations(selected_user, 'svd', recommendations)
            
            movie_ids = [rec[0] for rec in recommendations]
            scores = [rec[1] for rec in recommendations]
            
            recommended_movies = system['movies_df'][system['movies_df']["movieId"].isin(movie_ids)]
            
            # Create a mapping of movie_id to score
            score_map = dict(zip(movie_ids, scores))
            
            cols = st.columns(3)
            for i, row in enumerate(recommended_movies.itertuples()):
                with cols[i % 3]:
                    poster_url = get_movie_poster(row.title)
                    if poster_url:
                        st.image(poster_url, width=150)
                    else:
                        st.image("https://via.placeholder.com/150x225?text=No+Image", width=150)
                    
                    score = score_map.get(row.movieId, 0)
                    st.markdown(f"**🎬 {row.title}**")
                    st.markdown(f"⭐ Predicted Rating: {score:.2f}")
        else:
            st.warning("No SVD recommendations found for this user.")

elif algorithm_choice == "Matrix Factorization (NMF)":
    st.header("🔢 Matrix Factorization (NMF)")
    
    # Model Information
    model_info = system['nmf_model'].get_model_info()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🧮 Components", model_info['n_components'])
    with col2:
        st.metric("👥 Users", model_info['n_users'])
    with col3:
        st.metric("🎬 Movies", model_info['n_items'])
    
    if st.button("🎯 Get NMF Recommendations", key="nmf_recs"):
        recommendations = system['nmf_model'].get_user_recommendations(selected_user, n_recommendations)
        
        if recommendations:
            # Cache recommendations
            if cache_recommendations:
                system['db'].cache_recommendations(selected_user, 'nmf', recommendations)
            
            movie_ids = [rec[0] for rec in recommendations]
            scores = [rec[1] for rec in recommendations]
            
            recommended_movies = system['movies_df'][system['movies_df']["movieId"].isin(movie_ids)]
            
            # Create a mapping of movie_id to score
            score_map = dict(zip(movie_ids, scores))
            
            cols = st.columns(3)
            for i, row in enumerate(recommended_movies.itertuples()):
                with cols[i % 3]:
                    poster_url = get_movie_poster(row.title)
                    if poster_url:
                        st.image(poster_url, width=150)
                    else:
                        st.image("https://via.placeholder.com/150x225?text=No+Image", width=150)
                    
                    score = score_map.get(row.movieId, 0)
                    st.markdown(f"**🎬 {row.title}**")
                    st.markdown(f"⭐ Predicted Rating: {score:.2f}")
        else:
            st.warning("No NMF recommendations found for this user.")

elif algorithm_choice == "Hybrid System":
    st.header("🔀 Hybrid Recommendation System")
    
    # System Info
    hybrid_info = system['hybrid_system'].get_system_info()
    
    st.subheader("⚙️ System Configuration")
    weights_df = pd.DataFrame([hybrid_info['weights']]).T
    weights_df.columns = ['Weight']
    st.dataframe(weights_df)
    
    # Weight adjustment
    with st.expander("🎛️ Adjust Algorithm Weights"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            w_collab = st.slider("Collaborative", 0.0, 1.0, hybrid_info['weights']['collaborative'], 0.1)
        with col2:
            w_content = st.slider("Content-Based", 0.0, 1.0, hybrid_info['weights']['content'], 0.1)
        with col3:
            w_mf = st.slider("Matrix Factorization", 0.0, 1.0, hybrid_info['weights']['matrix_factorization'], 0.1)
        
        if st.button("🔄 Update Weights"):
            new_weights = {
                'collaborative': w_collab,
                'content': w_content,
                'matrix_factorization': w_mf
            }
            system['hybrid_system'].update_weights(new_weights)
            st.success("Weights updated!")
            st.rerun()
    
    if st.button("🎯 Get Hybrid Recommendations", key="hybrid_recs"):
        recommendations = system['hybrid_system'].get_hybrid_recommendations(
            selected_user, n_recommendations, explain=show_explanations
        )
        
        if recommendations:
            # Cache recommendations
            if cache_recommendations:
                cache_data = [(rec[0], rec[1]) for rec in recommendations]
                system['db'].cache_recommendations(selected_user, 'hybrid', cache_data)
            
            # Calculate diversity if requested
            diversity_score = None
            if show_diversity_score:
                diversity_score = system['hybrid_system'].get_recommendation_diversity(recommendations)
                st.metric("🎯 Diversity Score", f"{diversity_score:.3f}")
            
            if show_explanations:
                movie_ids = [rec[0] for rec in recommendations]
                scores = [rec[1] for rec in recommendations]
                explanations = [rec[2] for rec in recommendations]
            else:
                movie_ids = [rec[0] for rec in recommendations]
                scores = [rec[1] for rec in recommendations]
                explanations = [""] * len(recommendations)
            
            recommended_movies = system['movies_df'][system['movies_df']["movieId"].isin(movie_ids)]
            
            # Create mapping
            score_map = dict(zip(movie_ids, scores))
            explanation_map = dict(zip(movie_ids, explanations))
            
            cols = st.columns(2)
            for i, row in enumerate(recommended_movies.itertuples()):
                with cols[i % 2]:
                    poster_url = get_movie_poster(row.title)
                    
                    col_img, col_info = st.columns([1, 2])
                    with col_img:
                        if poster_url:
                            st.image(poster_url, width=100)
                        else:
                            st.image("https://via.placeholder.com/100x150?text=No+Image", width=100)
                    
                    with col_info:
                        score = score_map.get(row.movieId, 0)
                        st.markdown(f"**🎬 {row.title}**")
                        st.markdown(f"⭐ Hybrid Score: {score:.3f}")
                        
                        if show_explanations and explanation_map.get(row.movieId):
                            with st.expander("📝 Explanation"):
                                st.text(explanation_map[row.movieId])
                        
                        # Interaction buttons
                        if st.button("👍", key=f"like_hybrid_{i}"):
                            system['db'].log_user_interaction(selected_user, row.movieId, 'like')
                            st.success("Liked!")
                        if st.button("👁️", key=f"view_hybrid_{i}"):
                            system['db'].log_user_interaction(selected_user, row.movieId, 'view')
                            st.success("Viewed!")
                        rating = st.selectbox("Rate", [0, 1, 2, 3, 4, 5], key=f"rate_hybrid_{i}")
                        if rating > 0:
                            system['db'].add_user_rating(selected_user, row.movieId, rating)
                            st.success(f"Rated {rating}⭐")
        else:
            st.warning("No hybrid recommendations found for this user.")

elif algorithm_choice == "A/B Testing":
    st.header("🧪 A/B Testing Framework")
    
    # Create or manage experiments
    st.subheader("🔬 Experiment Management")
    
    # Create new experiment
    with st.expander("➕ Create New Experiment"):
        exp_name = st.text_input("Experiment Name", value="svd_vs_hybrid_test")
        
        # Algorithm selection for A/B test
        algo_a = st.selectbox("Algorithm A", ["SVD", "NMF", "Collaborative", "Hybrid"])
        algo_b = st.selectbox("Algorithm B", ["SVD", "NMF", "Collaborative", "Hybrid"])
        
        traffic_split_a = st.slider("Traffic % for Algorithm A", 0, 100, 50) / 100
        traffic_split_b = 1.0 - traffic_split_a
        
        if st.button("🚀 Create Experiment"):
            # Map algorithm names to models
            algo_map = {
                "SVD": system['svd_model'],
                "NMF": system['nmf_model'],
                "Collaborative": None,  # Use traditional collaborative filtering
                "Hybrid": system['hybrid_system']
            }
            
            algorithms = {
                'variant_a': {'model': algo_map[algo_a], 'params': {}},
                'variant_b': {'model': algo_map[algo_b], 'params': {}}
            }
            
            traffic_split = {
                'variant_a': traffic_split_a,
                'variant_b': traffic_split_b
            }
            
            experiment = system['ab_framework'].create_experiment(
                experiment_name=exp_name,
                algorithms=algorithms,
                traffic_split=traffic_split
            )
            
            st.success(f"✅ Created experiment '{exp_name}'")
    
    # Show active experiments
    st.subheader("📊 Active Experiments")
    experiments = system['ab_framework'].get_experiment_status()
    
    if experiments:
        for exp_name, exp_data in experiments.items():
            if exp_data['status'] == 'active':
                with st.expander(f"🧪 {exp_name}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Traffic Split:**")
                        st.json(exp_data['traffic_split'])
                    
                    with col2:
                        st.write("**Variants:**")
                        st.write(list(exp_data['algorithms'].keys()))
                    
                    # Get recommendations for this experiment
                    if st.button(f"🎯 Get Recommendations", key=f"ab_{exp_name}"):
                        recommendations, variant, metadata = system['ab_framework'].get_recommendations_for_experiment(
                            exp_name, selected_user, n_recommendations
                        )
                        
                        st.info(f"Using variant: **{variant}**")
                        
                        if recommendations:
                            movie_ids = [rec[0] for rec in recommendations] if isinstance(recommendations[0], tuple) else recommendations
                            recommended_movies = system['movies_df'][system['movies_df']["movieId"].isin(movie_ids)]
                            
                            cols = st.columns(3)
                            for i, row in enumerate(recommended_movies.itertuples()):
                                with cols[i % 3]:
                                    poster_url = get_movie_poster(row.title)
                                    if poster_url:
                                        st.image(poster_url, width=150)
                                    else:
                                        st.image("https://via.placeholder.com/150x225?text=No+Image", width=150)
                                    st.markdown(f"**🎬 {row.title}**")
                                    
                                    # Log interactions for A/B test
                                    if st.button("👍", key=f"ab_like_{exp_name}_{i}"):
                                        system['ab_framework'].log_user_interaction(
                                            exp_name, selected_user, row.movieId, 'like', variant=variant
                                        )
                                        st.success("Interaction logged!")
                    
                    # Show experiment metrics
                    if st.button(f"📈 View Metrics", key=f"metrics_{exp_name}"):
                        metrics = system['ab_framework'].calculate_experiment_metrics(exp_name)
                        st.json(metrics)
                    
                    # Generate report
                    if st.button(f"📋 Generate Report", key=f"report_{exp_name}"):
                        report = system['ab_framework'].generate_experiment_report(exp_name)
                        
                        st.subheader("📊 Experiment Report")
                        
                        # Show metrics comparison
                        if 'metrics' in report:
                            metrics_df = pd.DataFrame(report['metrics']).T
                            st.dataframe(metrics_df)
                        
                        # Show statistical tests
                        if 'statistical_tests' in report:
                            st.subheader("📈 Statistical Tests")
                            for metric, test in report['statistical_tests'].items():
                                st.write(f"**{metric}:**")
                                if test.get('is_significant'):
                                    p_val = test.get('p_value', 'N/A')
                                    if isinstance(p_val, (int, float)):
                                        st.success(f"✅ Significant difference (p={p_val:.4f})")
                                    else:
                                        st.success(f"✅ Significant difference (p={p_val})")
                                else:
                                    p_val = test.get('p_value', 'N/A')
                                    if isinstance(p_val, (int, float)):
                                        st.info(f"ℹ️ No significant difference (p={p_val:.4f})")
                                    else:
                                        st.info(f"ℹ️ No significant difference (p={p_val})")
                        
                        # Show recommendations
                        if 'recommendations' in report:
                            st.subheader("💡 Recommendations")
                            for rec in report['recommendations']:
                                st.write(f"**{rec['type']}:** {rec['reason']}")
    else:
        st.info("No experiments created yet. Create one above to get started!")

# ---------- 👤 User Profile Section ----------
with st.expander("👤 User Profile & Statistics"):
    user_stats = system['db'].get_user_statistics(selected_user)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Ratings", user_stats['total_ratings'])
    with col2:
        st.metric("⭐ Avg Rating", user_stats['avg_rating'])
    with col3:
        st.metric("📉 Min Rating", user_stats['min_rating'])
    with col4:
        st.metric("📈 Max Rating", user_stats['max_rating'])
    
    if user_stats['top_genres']:
        st.subheader("🎭 Top Genres")
        genres_df = pd.DataFrame(user_stats['top_genres'], columns=['Genre', 'Avg Rating', 'Count'])
        st.dataframe(genres_df)

# ---------- 🧹 Admin Section ----------
with st.expander("🔧 Admin Panel"):
    st.subheader("Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Clean Expired Cache"):
            system['db'].cleanup_expired_cache()
            st.success("Cache cleaned!")
    
    with col2:
        if st.button("📊 Refresh System Stats"):
            st.rerun()

# ---------- 📱 Footer ----------
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🎬 Advanced Movie Recommendation System v2.0</p>
    <p>Powered by Matrix Factorization, Hybrid Algorithms & A/B Testing</p>
</div>
""", unsafe_allow_html=True)
