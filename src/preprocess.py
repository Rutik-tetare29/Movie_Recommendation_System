# src/preprocess.py

import pandas as pd

def load_ratings(path="data/ratings.csv"):
    return pd.read_csv(path)

def load_movies(path="data/movies.csv"):
    return pd.read_csv(path)

def create_user_item_matrix(ratings_df):
    return ratings_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
