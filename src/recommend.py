

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
    # Get movieId from title
    movie = movies_df[movies_df["title"] == movie_title]
    if movie.empty:
        return []

    movie_id = movie["movieId"].values[0]

    # Check if movie exists in similarity matrix
    if movie_id not in item_similarity:
        return []

    similar_scores = item_similarity.loc[movie_id].drop(movie_id)
    top_similar = similar_scores.sort_values(ascending=False).head(n)

    return top_similar.index.tolist()
