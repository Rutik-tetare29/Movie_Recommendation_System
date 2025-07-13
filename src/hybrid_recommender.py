# src/hybrid_recommender.py

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class HybridRecommendationSystem:
    """
    Advanced Hybrid Recommendation System
    Combines multiple recommendation approaches for better accuracy
    """
    
    def __init__(self, weights=None):
        """
        Initialize Hybrid Recommendation System
        
        Args:
            weights (dict): Weights for different recommendation methods
                          Default: {'collaborative': 0.4, 'content': 0.3, 'matrix_factorization': 0.3}
        """
        self.weights = weights or {
            'collaborative': 0.4,
            'content': 0.3, 
            'matrix_factorization': 0.3
        }
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        self.collaborative_model = None
        self.content_model = None
        self.mf_model = None
        self.movies_df = None
        self.user_item_matrix = None
        self.content_features = None
        self.is_fitted = False
        
        print(f"🔧 Hybrid system initialized with weights: {self.weights}")
    
    def fit(self, user_item_matrix, movies_df, collaborative_model=None, mf_model=None):
        """
        Fit the hybrid recommendation system
        
        Args:
            user_item_matrix (pd.DataFrame): User-item rating matrix
            movies_df (pd.DataFrame): Movies metadata
            collaborative_model: Fitted collaborative filtering model
            mf_model: Fitted matrix factorization model
        """
        print("🚀 Training Hybrid Recommendation System...")
        
        self.user_item_matrix = user_item_matrix
        self.movies_df = movies_df
        self.collaborative_model = collaborative_model
        self.mf_model = mf_model
        
        # Build content-based features
        self._build_content_features()
        
        self.is_fitted = True
        print("✅ Hybrid system training completed!")
    
    def _build_content_features(self):
        """Build content-based recommendation features"""
        print("📝 Building content-based features...")
        
        # Prepare movie features
        features = []
        
        # Process genres
        if 'genres' in self.movies_df.columns:
            # Split genres and create feature strings
            genre_features = self.movies_df['genres'].fillna('').str.replace('|', ' ')
            features.append(genre_features)
        
        # Process titles (extract keywords)
        if 'title' in self.movies_df.columns:
            # Clean titles and extract meaningful words
            title_features = self.movies_df['title'].fillna('').str.lower()
            title_features = title_features.str.replace(r'[^\w\s]', ' ', regex=True)
            features.append(title_features)
        
        # Combine all features
        if len(features) > 1:
            combined_features = features[0].astype(str) + ' ' + features[1].astype(str)
        elif len(features) == 1:
            combined_features = features[0].astype(str)
        else:
            combined_features = pd.Series([''] * len(self.movies_df))
        
        # Fill NaN values
        combined_features = combined_features.fillna('')
        
        # Create TF-IDF vectors
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        
        self.content_features = self.tfidf_vectorizer.fit_transform(combined_features.fillna(''))
        print(f"✅ Content features created: {self.content_features.shape}")
    
    def _get_collaborative_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations from collaborative filtering"""
        if self.collaborative_model is None:
            return [], []
        
        try:
            # Assuming collaborative model has a method to get recommendations
            if hasattr(self.collaborative_model, 'get_user_recommendations'):
                recommendations = self.collaborative_model.get_user_recommendations(user_id, n_recommendations)
                movie_ids, scores = zip(*recommendations) if recommendations else ([], [])
                return list(movie_ids), list(scores)
            else:
                # Fallback: compute similarity manually
                if user_id not in self.user_item_matrix.index:
                    return [], []
                
                user_ratings = self.user_item_matrix.loc[user_id]
                unrated_movies = user_ratings[user_ratings == 0].index
                
                # Simple user-based collaborative filtering
                user_similarities = cosine_similarity([user_ratings], self.user_item_matrix)[0]
                similar_users = self.user_item_matrix.index[np.argsort(user_similarities)[::-1]][1:11]
                
                # Get recommendations from similar users
                recommendations = {}
                for similar_user in similar_users:
                    similar_ratings = self.user_item_matrix.loc[similar_user]
                    for movie_id in unrated_movies:
                        if similar_ratings[movie_id] > 0:
                            if movie_id not in recommendations:
                                recommendations[movie_id] = 0
                            recommendations[movie_id] += similar_ratings[movie_id] * user_similarities[self.user_item_matrix.index.get_loc(similar_user)]
                
                # Sort and return top N
                sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
                movie_ids, scores = zip(*sorted_recs) if sorted_recs else ([], [])
                return list(movie_ids), list(scores)
                
        except Exception as e:
            print(f"⚠️ Error in collaborative filtering: {e}")
            return [], []
    
    def _get_content_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations from content-based filtering"""
        try:
            if user_id not in self.user_item_matrix.index:
                return [], []
            
            # Get user's rated movies and their ratings
            user_ratings = self.user_item_matrix.loc[user_id]
            rated_movies = user_ratings[user_ratings > 0]
            
            if len(rated_movies) == 0:
                return [], []
            
            # Calculate user profile based on rated movies
            user_profile = np.zeros(self.content_features.shape[1])
            total_weight = 0
            
            for movie_id, rating in rated_movies.items():
                if movie_id in self.movies_df['movieId'].values:
                    movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
                    movie_features = self.content_features[movie_idx].toarray()[0]
                    user_profile += movie_features * rating
                    total_weight += rating
            
            if total_weight > 0:
                user_profile /= total_weight
            
            # Find similar movies
            content_similarities = cosine_similarity([user_profile], self.content_features)[0]
            
            # Get unrated movies
            unrated_movies = user_ratings[user_ratings == 0].index
            
            # Score unrated movies
            recommendations = {}
            for movie_id in unrated_movies:
                if movie_id in self.movies_df['movieId'].values:
                    movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
                    recommendations[movie_id] = content_similarities[movie_idx]
            
            # Sort and return top N
            sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
            movie_ids, scores = zip(*sorted_recs) if sorted_recs else ([], [])
            return list(movie_ids), list(scores)
            
        except Exception as e:
            print(f"⚠️ Error in content-based filtering: {e}")
            return [], []
    
    def _get_matrix_factorization_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations from matrix factorization"""
        if self.mf_model is None:
            return [], []
        
        try:
            if hasattr(self.mf_model, 'get_user_recommendations'):
                recommendations = self.mf_model.get_user_recommendations(user_id, n_recommendations)
                movie_ids, scores = zip(*recommendations) if recommendations else ([], [])
                return list(movie_ids), list(scores)
            else:
                return [], []
                
        except Exception as e:
            print(f"⚠️ Error in matrix factorization: {e}")
            return [], []
    
    def get_hybrid_recommendations(self, user_id, n_recommendations=10, explain=False):
        """
        Get hybrid recommendations combining all methods
        
        Args:
            user_id: Target user ID
            n_recommendations (int): Number of recommendations to return
            explain (bool): Whether to return explanation of recommendation sources
            
        Returns:
            List of (movie_id, hybrid_score) tuples, optionally with explanations
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        print(f"🎯 Generating hybrid recommendations for user {user_id}...")
        
        # Get recommendations from each method
        collaborative_movies, collaborative_scores = self._get_collaborative_recommendations(user_id, n_recommendations * 2)
        content_movies, content_scores = self._get_content_recommendations(user_id, n_recommendations * 2)
        mf_movies, mf_scores = self._get_matrix_factorization_recommendations(user_id, n_recommendations * 2)
        
        # Normalize scores to [0, 1] range
        def normalize_scores(scores):
            if not scores or len(scores) == 0:
                return scores
            min_score, max_score = min(scores), max(scores)
            if max_score == min_score:
                return [1.0] * len(scores)
            return [(s - min_score) / (max_score - min_score) for s in scores]
        
        collaborative_scores = normalize_scores(collaborative_scores)
        content_scores = normalize_scores(content_scores)
        mf_scores = normalize_scores(mf_scores)
        
        # Combine recommendations
        hybrid_scores = {}
        explanations = {}
        
        # Add collaborative filtering recommendations
        for movie_id, score in zip(collaborative_movies, collaborative_scores):
            if movie_id not in hybrid_scores:
                hybrid_scores[movie_id] = 0
                explanations[movie_id] = []
            hybrid_scores[movie_id] += score * self.weights['collaborative']
            explanations[movie_id].append(f"Collaborative: {score:.3f}")
        
        # Add content-based recommendations
        for movie_id, score in zip(content_movies, content_scores):
            if movie_id not in hybrid_scores:
                hybrid_scores[movie_id] = 0
                explanations[movie_id] = []
            hybrid_scores[movie_id] += score * self.weights['content']
            explanations[movie_id].append(f"Content: {score:.3f}")
        
        # Add matrix factorization recommendations
        for movie_id, score in zip(mf_movies, mf_scores):
            if movie_id not in hybrid_scores:
                hybrid_scores[movie_id] = 0
                explanations[movie_id] = []
            hybrid_scores[movie_id] += score * self.weights['matrix_factorization']
            explanations[movie_id].append(f"Matrix Factorization: {score:.3f}")
        
        # Sort by hybrid score
        sorted_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        final_recommendations = sorted_recommendations[:n_recommendations]
        
        print(f"✅ Generated {len(final_recommendations)} hybrid recommendations")
        
        if explain:
            result = []
            for movie_id, score in final_recommendations:
                explanation = " | ".join(explanations.get(movie_id, []))
                result.append((movie_id, score, explanation))
            return result
        else:
            return final_recommendations
    
    def get_movie_recommendations_by_content(self, movie_id, n_recommendations=10):
        """
        Get movies similar to a given movie based on content
        
        Args:
            movie_id: Target movie ID
            n_recommendations (int): Number of recommendations
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        try:
            if movie_id not in self.movies_df['movieId'].values:
                print(f"⚠️ Movie {movie_id} not found")
                return []
            
            # Get movie index
            movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
            
            # Calculate content similarities
            movie_features = self.content_features[movie_idx]
            similarities = cosine_similarity(movie_features, self.content_features)[0]
            
            # Get top similar movies (excluding the movie itself)
            similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
            
            recommendations = []
            for idx in similar_indices:
                similar_movie_id = self.movies_df.iloc[idx]['movieId']
                similarity_score = similarities[idx]
                recommendations.append((similar_movie_id, similarity_score))
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ Error in content-based movie similarity: {e}")
            return []
    
    def update_weights(self, new_weights):
        """
        Update the weights for different recommendation methods
        
        Args:
            new_weights (dict): New weights dictionary
        """
        # Normalize weights
        total_weight = sum(new_weights.values())
        self.weights = {k: v/total_weight for k, v in new_weights.items()}
        print(f"🔧 Updated weights: {self.weights}")
    
    def get_recommendation_diversity(self, recommendations, top_n=10):
        """
        Calculate diversity of recommendations
        
        Args:
            recommendations: List of movie IDs
            top_n (int): Number of top recommendations to consider
            
        Returns:
            float: Diversity score (0-1, higher is more diverse)
        """
        if len(recommendations) < 2:
            return 0.0
        
        try:
            # Get movie features for recommended movies
            movie_indices = []
            for rec in recommendations[:top_n]:
                # Handle different recommendation formats
                if isinstance(rec, tuple):
                    movie_id = rec[0]
                else:
                    movie_id = rec
                
                if movie_id in self.movies_df['movieId'].values:
                    idx = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
                    movie_indices.append(idx)
            
            if len(movie_indices) < 2:
                return 0.0
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(movie_indices)):
                for j in range(i+1, len(movie_indices)):
                    similarity = cosine_similarity(
                        self.content_features[movie_indices[i]], 
                        self.content_features[movie_indices[j]]
                    )[0][0]
                    similarities.append(similarity)
            
            # Diversity = 1 - average similarity
            avg_similarity = np.mean(similarities)
            diversity = 1 - avg_similarity
            
            return max(0.0, min(1.0, diversity))
            
        except Exception as e:
            print(f"⚠️ Error calculating diversity: {e}")
            return 0.0
    
    def get_system_info(self):
        """Get information about the hybrid system"""
        if not self.is_fitted:
            return "Hybrid system not fitted yet"
        
        info = {
            'weights': self.weights,
            'content_features_shape': self.content_features.shape,
            'n_movies': len(self.movies_df),
            'n_users': len(self.user_item_matrix),
            'has_collaborative_model': self.collaborative_model is not None,
            'has_matrix_factorization_model': self.mf_model is not None,
        }
        
        return info


def create_hybrid_system(user_item_matrix, movies_df, collaborative_model=None, mf_model=None, weights=None):
    """
    Convenience function to create and fit a hybrid recommendation system
    
    Args:
        user_item_matrix: User-item rating matrix
        movies_df: Movies dataframe
        collaborative_model: Fitted collaborative filtering model
        mf_model: Fitted matrix factorization model
        weights: Custom weights for different methods
        
    Returns:
        HybridRecommendationSystem: Fitted hybrid system
    """
    hybrid_system = HybridRecommendationSystem(weights)
    hybrid_system.fit(user_item_matrix, movies_df, collaborative_model, mf_model)
    return hybrid_system
