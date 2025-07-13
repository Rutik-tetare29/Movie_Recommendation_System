# src/database.py

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

class MovieRecommenderDB:
    """
    Database manager for Movie Recommendation System
    Handles data persistence, user interactions, and recommendation caching
    """
    
    def __init__(self, db_path="data/movie_recommender.db"):
        """
        Initialize database connection
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database tables
        self._init_database()
        
    def _init_database(self):
        """Create database tables if they don't exist"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Movies table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS movies (
                    movie_id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    genres TEXT,
                    year INTEGER,
                    poster_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Users table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE,
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Ratings table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ratings (
                    rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    movie_id INTEGER,
                    rating REAL NOT NULL,
                    timestamp INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (movie_id) REFERENCES movies (movie_id),
                    UNIQUE(user_id, movie_id)
                )
            """)
            
            # User interactions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_interactions (
                    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    movie_id INTEGER,
                    interaction_type TEXT,
                    interaction_value REAL DEFAULT 1.0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (movie_id) REFERENCES movies (movie_id)
                )
            """)
            
            # Recommendations cache table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recommendation_cache (
                    cache_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    algorithm TEXT,
                    recommendations TEXT,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            # Model performance metrics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    algorithm TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    parameters TEXT,
                    dataset_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # User feedback table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    movie_id INTEGER,
                    recommendation_algorithm TEXT,
                    feedback_type TEXT,
                    feedback_value REAL,
                    comments TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (movie_id) REFERENCES movies (movie_id)
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ratings_user_id ON ratings(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ratings_movie_id ON ratings(movie_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON user_interactions(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_movie_id ON user_interactions(movie_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_user_algorithm ON recommendation_cache(user_id, algorithm)")
            
            conn.commit()
        
        print("✅ Database initialized successfully!")
    
    def load_csv_data(self, movies_csv="data/movies.csv", ratings_csv="data/ratings.csv"):
        """
        Load data from CSV files into database
        
        Args:
            movies_csv (str): Path to movies CSV file
            ratings_csv (str): Path to ratings CSV file
        """
        print("📊 Loading CSV data into database...")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load movies
                if os.path.exists(movies_csv):
                    movies_df = pd.read_csv(movies_csv)
                    
                    # Extract year from title if exists
                    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype('Int64')
                    
                    # Rename columns to match database schema
                    movies_df = movies_df.rename(columns={'movieId': 'movie_id'})
                    
                    # Insert movies data
                    movies_df.to_sql('movies', conn, if_exists='replace', index=False, 
                                   dtype={'movie_id': 'INTEGER', 'title': 'TEXT', 'genres': 'TEXT', 'year': 'INTEGER'})
                    print(f"✅ Loaded {len(movies_df)} movies")
                
                # Load ratings
                if os.path.exists(ratings_csv):
                    ratings_df = pd.read_csv(ratings_csv)
                    
                    # Rename columns to match database schema
                    ratings_df = ratings_df.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})
                    
                    # Insert ratings data
                    ratings_df.to_sql('ratings', conn, if_exists='replace', index=False,
                                    dtype={'user_id': 'INTEGER', 'movie_id': 'INTEGER', 'rating': 'REAL', 'timestamp': 'INTEGER'})
                    print(f"✅ Loaded {len(ratings_df)} ratings")
                
                # Create user entries
                unique_users = pd.read_sql_query("SELECT DISTINCT user_id FROM ratings", conn)
                unique_users.to_sql('users', conn, if_exists='replace', index=False,
                                  dtype={'user_id': 'INTEGER'})
                print(f"✅ Created {len(unique_users)} user records")
                
        except Exception as e:
            print(f"❌ Error loading CSV data: {e}")
    
    def get_user_item_matrix(self):
        """Get user-item rating matrix as pandas DataFrame"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT user_id, movie_id, rating 
                FROM ratings
            """
            ratings_df = pd.read_sql_query(query, conn)
            
            # Create pivot table (user-item matrix)
            user_item_matrix = ratings_df.pivot_table(
                index='user_id', 
                columns='movie_id', 
                values='rating'
            ).fillna(0)
            
            return user_item_matrix
    
    def get_movies_info(self):
        """Get movies information as pandas DataFrame"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT movie_id as movieId, title, genres, year
                FROM movies
            """
            return pd.read_sql_query(query, conn)
    
    def add_user_rating(self, user_id, movie_id, rating):
        """Add or update user rating"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ratings (user_id, movie_id, rating, timestamp)
                VALUES (?, ?, ?, ?)
            """, (user_id, movie_id, rating, int(datetime.now().timestamp())))
            conn.commit()
    
    def log_user_interaction(self, user_id, movie_id, interaction_type, value=1.0):
        """Log user interaction (view, like, click, etc.)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO user_interactions (user_id, movie_id, interaction_type, interaction_value)
                VALUES (?, ?, ?, ?)
            """, (user_id, movie_id, interaction_type, value))
            conn.commit()
    
    def cache_recommendations(self, user_id, algorithm, recommendations, expires_hours=24):
        """Cache recommendations for faster retrieval"""
        with sqlite3.connect(self.db_path) as conn:
            # Delete old cache for this user/algorithm
            conn.execute("""
                DELETE FROM recommendation_cache 
                WHERE user_id = ? AND algorithm = ?
            """, (user_id, algorithm))
            
            # Insert new cache
            expires_at = datetime.now() + timedelta(hours=expires_hours)
            recommendations_json = json.dumps(recommendations)
            
            conn.execute("""
                INSERT INTO recommendation_cache (user_id, algorithm, recommendations, expires_at)
                VALUES (?, ?, ?, ?)
            """, (user_id, algorithm, recommendations_json, expires_at))
            conn.commit()
    
    def get_cached_recommendations(self, user_id, algorithm):
        """Get cached recommendations if not expired"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT recommendations, created_at 
                FROM recommendation_cache 
                WHERE user_id = ? AND algorithm = ? AND expires_at > ?
            """, (user_id, algorithm, datetime.now()))
            
            result = cursor.fetchone()
            if result:
                return {
                    'recommendations': json.loads(result[0]),
                    'created_at': result[1]
                }
            return None
    
    def cleanup_expired_cache(self):
        """Remove expired cache entries"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM recommendation_cache 
                WHERE expires_at < ?
            """, (datetime.now(),))
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count
    
    def get_user_statistics(self, user_id):
        """Get comprehensive user statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Basic rating stats
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_ratings,
                    AVG(rating) as avg_rating,
                    MIN(rating) as min_rating,
                    MAX(rating) as max_rating
                FROM ratings 
                WHERE user_id = ?
            """, (user_id,))
            
            stats = dict(zip(['total_ratings', 'avg_rating', 'min_rating', 'max_rating'], 
                           cursor.fetchone()))
            
            # Top genres
            cursor = conn.execute("""
                SELECT genres, AVG(rating) as avg_rating, COUNT(*) as count
                FROM ratings r
                JOIN movies m ON r.movie_id = m.movie_id
                WHERE r.user_id = ? AND m.genres IS NOT NULL
                GROUP BY genres
                ORDER BY avg_rating DESC, count DESC
                LIMIT 5
            """, (user_id,))
            
            stats['top_genres'] = cursor.fetchall()
            
            return stats
    
    def get_system_statistics(self):
        """Get overall system statistics"""
        stats = {}
        
        with sqlite3.connect(self.db_path) as conn:
            # Basic counts
            stats['total_users'] = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            stats['total_movies'] = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
            stats['total_ratings'] = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
            stats['total_interactions'] = conn.execute("SELECT COUNT(*) FROM user_interactions").fetchone()[0]
            
            # Rating distribution
            cursor = conn.execute("""
                SELECT rating, COUNT(*) as count
                FROM ratings
                GROUP BY rating
                ORDER BY rating
            """)
            stats['rating_distribution'] = dict(cursor.fetchall())
            
            # Active users (users with ratings in last 30 days)
            # Note: This might not work with timestamp format, but keeping for completeness
            try:
                cursor = conn.execute("""
                    SELECT COUNT(DISTINCT user_id)
                    FROM ratings
                    WHERE created_at > datetime('now', '-30 days')
                """)
                stats['active_users_30d'] = cursor.fetchone()[0]
            except:
                stats['active_users_30d'] = 0
            
            # Popular movies
            cursor = conn.execute("""
                SELECT m.title, COUNT(*) as rating_count, AVG(r.rating) as avg_rating
                FROM ratings r
                JOIN movies m ON r.movie_id = m.movie_id
                GROUP BY m.movie_id, m.title
                ORDER BY rating_count DESC, avg_rating DESC
                LIMIT 10
            """)
            stats['popular_movies'] = cursor.fetchall()
            
            return stats
    
    def cleanup_old_data(self, days=30):
        """Clean up old interaction data"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM user_interactions 
                WHERE timestamp < ?
            """, (cutoff_date,))
            deleted_count = cursor.rowcount
            conn.commit()
            
            return deleted_count


def initialize_database_with_csv(db_path="data/movie_recommender.db", 
                                movies_csv="data/movies.csv", 
                                ratings_csv="data/ratings.csv"):
    """
    Initialize database and load CSV data if database is empty
    
    Args:
        db_path (str): Path to database file
        movies_csv (str): Path to movies CSV
        ratings_csv (str): Path to ratings CSV
    
    Returns:
        MovieRecommenderDB: Initialized database instance
    """
    db = MovieRecommenderDB(db_path)
    
    # Check if database has data
    with sqlite3.connect(db_path) as conn:
        movie_count = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
        
        if movie_count == 0:
            print("📊 Empty database detected. Loading CSV data...")
            db.load_csv_data(movies_csv, ratings_csv)
        else:
            print(f"✅ Database already contains {movie_count} movies")
    
    return db
