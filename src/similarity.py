

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def compute_user_similarity(user_item_matrix):
    similarity = cosine_similarity(user_item_matrix)
    return pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def compute_item_similarity(user_item_matrix):
    similarity = cosine_similarity(user_item_matrix.T)
    return pd.DataFrame(similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
