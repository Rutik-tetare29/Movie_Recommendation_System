

def user_based_recommendations(user_id, user_item_matrix, user_similarity, n=5):
    if user_id not in user_similarity:
        return []

    user_ratings = user_item_matrix.loc[user_id]
    similar_users = user_similarity.loc[user_id].drop(user_id)
    similar_users_ratings = user_item_matrix.loc[similar_users.index]

    weighted_scores = similar_users_ratings.T.dot(similar_users)
    normalized_scores = weighted_scores / similar_users.sum()

    unrated_movies = user_ratings[user_ratings == 0].index
    recommendations = normalized_scores.loc[unrated_movies]

    return recommendations.sort_values(ascending=False).head(n).index.tolist()


def item_based_recommendations(movie_title, user_item_matrix, movies_df, item_similarity, n=5):
    """
    Get item-based recommendations for a given movie
    
    Args:
        movie_title: Title of the movie to find similar movies for
        user_item_matrix: User-item rating matrix
        movies_df: DataFrame with movie information
        item_similarity: Item-item similarity matrix
        n: Number of recommendations to return
    
    Returns:
        List of similar movie IDs
    """
    # Get movieId from title
    movie = movies_df[movies_df["title"] == movie_title]
    if movie.empty:
        print(f"⚠️ Movie '{movie_title}' not found in database")
        return []

    movie_id = movie["movieId"].values[0]

    # Check if movie exists in similarity matrix
    if movie_id not in item_similarity.index:
        print(f"⚠️ Movie '{movie_title}' (ID: {movie_id}) not in similarity matrix")
        
        # Fallback: Find movies in same genre or with similar ratings
        movie_genres = movie["genres"].values[0] if "genres" in movie.columns else ""
        
        if movie_genres and movie_genres != "(no genres listed)":
            # Find movies with similar genres
            genre_matches = movies_df[
                movies_df["genres"].str.contains(movie_genres.split("|")[0], na=False)
            ]
            
            # Get movies that are in the similarity matrix
            available_movies = genre_matches[
                genre_matches["movieId"].isin(item_similarity.index)
            ]
            
            if not available_movies.empty:
                print(f"📋 Found {len(available_movies)} movies with similar genres")
                return available_movies["movieId"].head(n).tolist()
        
        # Last resort: return most popular movies from similarity matrix
        print("📋 Returning most popular movies as fallback")
        popular_movies = user_item_matrix[item_similarity.index].count(axis=0)
        return popular_movies.sort_values(ascending=False).head(n).index.tolist()

    # Normal case: movie exists in similarity matrix
    similar_scores = item_similarity.loc[movie_id].drop(movie_id)
    
    # Remove any NaN values and sort
    similar_scores = similar_scores.dropna()
    top_similar = similar_scores.sort_values(ascending=False).head(n)

    return top_similar.index.tolist()
