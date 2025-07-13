

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import warnings

def compute_user_similarity(user_item_matrix):
    """Compute user-user similarity matrix"""
    similarity = cosine_similarity(user_item_matrix)
    return pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def compute_item_similarity(user_item_matrix, max_items=5000, use_sparse=True):
    """
    Compute item-item similarity matrix with memory optimization
    
    Args:
        user_item_matrix: User-item rating matrix
        max_items: Maximum number of items to compute similarity for (to save memory)
        use_sparse: Whether to use sparse matrix representation
    
    Returns:
        DataFrame: Item-item similarity matrix
    """
    print(f"📊 Computing item similarity for {user_item_matrix.shape[1]} items...")
    
    # If too many items, use top-rated items only
    if user_item_matrix.shape[1] > max_items:
        print(f"⚠️ Too many items ({user_item_matrix.shape[1]}), using top {max_items} most-rated items")
        
        # Get items with most ratings
        item_counts = user_item_matrix.count(axis=0)
        top_items = item_counts.nlargest(max_items).index
        user_item_subset = user_item_matrix[top_items]
        
        print(f"📊 Reduced to {len(top_items)} items for similarity computation")
    else:
        user_item_subset = user_item_matrix
        top_items = user_item_matrix.columns
    
    try:
        # Convert to sparse matrix if requested and beneficial
        if use_sparse and user_item_subset.shape[1] > 1000:
            print("🔧 Using sparse matrix representation for memory efficiency...")
            
            # Fill NaN with 0 for sparse representation
            item_matrix = user_item_subset.T.fillna(0)
            sparse_matrix = csr_matrix(item_matrix.values)
            
            # Compute similarity in chunks to save memory
            chunk_size = min(500, item_matrix.shape[0] // 4)
            n_items = item_matrix.shape[0]
            
            # Initialize similarity matrix
            similarity_matrix = np.zeros((n_items, n_items))
            
            print(f"🔄 Computing similarity in chunks of {chunk_size}...")
            for i in range(0, n_items, chunk_size):
                end_i = min(i + chunk_size, n_items)
                chunk_similarity = cosine_similarity(sparse_matrix[i:end_i], sparse_matrix)
                similarity_matrix[i:end_i, :] = chunk_similarity
                
                if i % (chunk_size * 4) == 0:  # Progress indicator
                    progress = (end_i / n_items) * 100
                    print(f"  Progress: {progress:.1f}%")
            
            similarity_df = pd.DataFrame(
                similarity_matrix, 
                index=top_items, 
                columns=top_items
            )
            
        else:
            # Standard dense computation for smaller matrices
            print("🔧 Using standard dense matrix computation...")
            item_matrix = user_item_subset.T.fillna(0)
            similarity = cosine_similarity(item_matrix)
            similarity_df = pd.DataFrame(
                similarity, 
                index=top_items, 
                columns=top_items
            )
        
        print(f"✅ Item similarity computed: {similarity_df.shape}")
        return similarity_df
        
    except MemoryError as e:
        print(f"❌ Memory error in item similarity: {e}")
        print("🔧 Falling back to minimal similarity matrix...")
        
        # Fallback: Create a minimal similarity matrix with just top 100 items
        item_counts = user_item_matrix.count(axis=0)
        top_100_items = item_counts.nlargest(100).index
        minimal_subset = user_item_matrix[top_100_items].T.fillna(0)
        
        similarity = cosine_similarity(minimal_subset)
        similarity_df = pd.DataFrame(
            similarity, 
            index=top_100_items, 
            columns=top_100_items
        )
        
        print(f"✅ Minimal similarity matrix created: {similarity_df.shape}")
        return similarity_df
    
    except Exception as e:
        print(f"❌ Error computing item similarity: {e}")
        # Return identity matrix as last resort
        warnings.warn("Returning identity matrix due to computation error")
        identity_matrix = pd.DataFrame(
            np.eye(len(top_items)), 
            index=top_items, 
            columns=top_items
        )
        return identity_matrix
