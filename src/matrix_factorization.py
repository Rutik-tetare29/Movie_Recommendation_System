# src/matrix_factorization.py

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class MatrixFactorization:
    """
    Advanced Matrix Factorization for Movie Recommendations
    Supports SVD (Singular Value Decomposition) and NMF (Non-negative Matrix Factorization)
    """
    
    def __init__(self, method='svd', n_components=50, random_state=42):
        """
        Initialize Matrix Factorization model
        
        Args:
            method (str): 'svd' or 'nmf'
            n_components (int): Number of latent factors
            random_state (int): Random seed for reproducibility
        """
        self.method = method.lower()
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        self.user_means = None
        self.item_means = None
        self.is_fitted = False
        
    def _prepare_data(self, user_item_matrix):
        """Prepare data for matrix factorization"""
        # Convert to sparse matrix for memory efficiency
        self.original_matrix = user_item_matrix.copy()
        
        # Calculate means for bias handling
        self.global_mean = user_item_matrix[user_item_matrix > 0].mean()
        self.user_means = user_item_matrix.mean(axis=1)
        self.item_means = user_item_matrix.mean(axis=0)
        
        # Create mask for known ratings
        self.mask = (user_item_matrix > 0).astype(int)
        
        return user_item_matrix
    
    def fit(self, user_item_matrix):
        """
        Fit the matrix factorization model
        
        Args:
            user_item_matrix (pd.DataFrame): User-item rating matrix
        """
        print(f"Training {self.method.upper()} model with {self.n_components} components...")
        
        # Prepare data
        matrix = self._prepare_data(user_item_matrix)
        
        if self.method == 'svd':
            self.model = TruncatedSVD(
                n_components=self.n_components,
                random_state=self.random_state,
                algorithm='randomized'
            )
            
            # Fit SVD on the user-item matrix
            self.user_factors = self.model.fit_transform(matrix)
            self.item_factors = self.model.components_.T
            
        elif self.method == 'nmf':
            # NMF requires non-negative values
            matrix_non_neg = matrix.copy()
            matrix_non_neg[matrix_non_neg < 0] = 0
            
            self.model = NMF(
                n_components=self.n_components,
                random_state=self.random_state,
                max_iter=500,
                alpha_W=0.1,
                alpha_H=0.1
            )
            
            self.user_factors = self.model.fit_transform(matrix_non_neg)
            self.item_factors = self.model.components_.T
            
        else:
            raise ValueError("Method must be 'svd' or 'nmf'")
        
        self.is_fitted = True
        print(f"✅ {self.method.upper()} model training completed!")
        
        # Calculate and print reconstruction error
        reconstruction_error = self._calculate_rmse(matrix)
        print(f"📊 Reconstruction RMSE: {reconstruction_error:.4f}")
        
    def _calculate_rmse(self, original_matrix):
        """Calculate RMSE between original and reconstructed matrix"""
        if not self.is_fitted:
            return None
            
        reconstructed = self.user_factors @ self.item_factors.T
        
        # Only calculate RMSE for known ratings
        mask = self.mask.values
        known_original = original_matrix.values[mask == 1]
        known_reconstructed = reconstructed[mask == 1]
        
        return np.sqrt(mean_squared_error(known_original, known_reconstructed))
    
    def predict_rating(self, user_id, movie_id):
        """
        Predict rating for a specific user-movie pair
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            user_idx = self.original_matrix.index.get_loc(user_id)
            item_idx = self.original_matrix.columns.get_loc(movie_id)
            
            # Calculate prediction using dot product of latent factors
            prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
            
            # Add bias terms
            user_bias = self.user_means.iloc[user_idx] - self.global_mean
            item_bias = self.item_means.iloc[item_idx] - self.global_mean
            
            prediction = prediction + self.global_mean + user_bias + item_bias
            
            # Clip to valid rating range (assuming 0.5-5.0)
            return max(0.5, min(5.0, prediction))
            
        except (KeyError, ValueError):
            # Return global mean for unknown users/items
            return self.global_mean
    
    def get_user_recommendations(self, user_id, n_recommendations=10, exclude_rated=True):
        """
        Get movie recommendations for a user
        
        Args:
            user_id: Target user ID
            n_recommendations (int): Number of recommendations
            exclude_rated (bool): Whether to exclude already rated movies
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if user_id not in self.original_matrix.index:
            print(f"⚠️ User {user_id} not found in training data")
            return []
        
        user_idx = self.original_matrix.index.get_loc(user_id)
        
        # Get all movie predictions for this user
        user_vector = self.user_factors[user_idx]
        all_predictions = self.item_factors @ user_vector
        
        # Add bias terms
        user_bias = self.user_means.iloc[user_idx] - self.global_mean
        item_biases = self.item_means.values - self.global_mean
        all_predictions = all_predictions + self.global_mean + user_bias + item_biases
        
        # Create recommendations dataframe
        recommendations = pd.DataFrame({
            'movie_id': self.original_matrix.columns,
            'predicted_rating': all_predictions
        })
        
        # Exclude already rated movies if requested
        if exclude_rated:
            rated_movies = self.original_matrix.loc[user_id]
            unrated_mask = rated_movies == 0
            recommendations = recommendations[unrated_mask.values]
        
        # Sort by predicted rating and return top N
        recommendations = recommendations.sort_values('predicted_rating', ascending=False)
        
        return list(zip(recommendations['movie_id'].head(n_recommendations), 
                       recommendations['predicted_rating'].head(n_recommendations)))
    
    def get_similar_items(self, movie_id, n_similar=10):
        """
        Find movies similar to the given movie using item factors
        
        Args:
            movie_id: Target movie ID
            n_similar (int): Number of similar movies to return
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding similarities")
        
        if movie_id not in self.original_matrix.columns:
            print(f"⚠️ Movie {movie_id} not found in training data")
            return []
        
        item_idx = self.original_matrix.columns.get_loc(movie_id)
        target_factors = self.item_factors[item_idx]
        
        # Calculate cosine similarity with all other items
        similarities = []
        for i, other_movie_id in enumerate(self.original_matrix.columns):
            if other_movie_id != movie_id:
                other_factors = self.item_factors[i]
                
                # Cosine similarity
                dot_product = np.dot(target_factors, other_factors)
                norm_product = np.linalg.norm(target_factors) * np.linalg.norm(other_factors)
                
                if norm_product > 0:
                    similarity = dot_product / norm_product
                    similarities.append((other_movie_id, similarity))
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]
    
    def get_model_info(self):
        """Get information about the fitted model"""
        if not self.is_fitted:
            return "Model not fitted yet"
        
        info = {
            'method': self.method.upper(),
            'n_components': self.n_components,
            'n_users': self.user_factors.shape[0],
            'n_items': self.item_factors.shape[0],
            'sparsity': f"{(1 - self.mask.sum().sum() / (self.mask.shape[0] * self.mask.shape[1])) * 100:.2f}%",
            'global_mean_rating': f"{float(self.global_mean.iloc[0] if hasattr(self.global_mean, 'iloc') else self.global_mean):.2f}"
        }
        
        return info


def compare_matrix_factorization_methods(user_item_matrix, n_components=50):
    """
    Compare SVD and NMF methods on the same dataset
    
    Args:
        user_item_matrix: User-item rating matrix
        n_components: Number of latent factors to use
        
    Returns:
        Dictionary with comparison results
    """
    print("🔄 Comparing Matrix Factorization Methods...")
    
    methods = ['svd', 'nmf']
    results = {}
    
    for method in methods:
        print(f"\n--- Testing {method.upper()} ---")
        mf_model = MatrixFactorization(method=method, n_components=n_components)
        mf_model.fit(user_item_matrix)
        
        # Get model info
        info = mf_model.get_model_info()
        results[method] = {
            'model': mf_model,
            'info': info,
            'rmse': mf_model._calculate_rmse(user_item_matrix)
        }
    
    # Print comparison
    print("\n" + "="*50)
    print("📊 COMPARISON RESULTS")
    print("="*50)
    
    for method, result in results.items():
        print(f"{method.upper()}:")
        print(f"  RMSE: {result['rmse']:.4f}")
        print(f"  Components: {result['info']['n_components']}")
        print()
    
    return results
