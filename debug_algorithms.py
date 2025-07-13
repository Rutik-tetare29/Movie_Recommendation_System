# debug_algorithms.py
# Script to debug specific algorithm issues

import sys
import os
import pandas as pd
import numpy as np
import traceback

# Add src to system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Imports
from database import initialize_database_with_csv
from matrix_factorization import MatrixFactorization
from hybrid_recommender import create_hybrid_system
from ab_testing import ABTestingFramework
from recommend import user_based_recommendations, item_based_recommendations
from similarity import compute_user_similarity, compute_item_similarity

def test_collaborative_filtering():
    """Test collaborative filtering specifically"""
    print("🔧 Testing Collaborative Filtering...")
    try:
        # Initialize database
        db = initialize_database_with_csv()
        user_item_matrix = db.get_user_item_matrix()
        movies_df = db.get_movies_info()
        
        # Test user similarity computation
        print("  Computing user similarity...")
        user_sim_matrix = compute_user_similarity(user_item_matrix)
        print(f"  ✅ User similarity matrix: {user_sim_matrix.shape}")
        
        # Test item similarity computation  
        print("  Computing item similarity...")
        item_sim_matrix = compute_item_similarity(user_item_matrix)
        print(f"  ✅ Item similarity matrix: {item_sim_matrix.shape}")
        
        # Test user-based recommendations
        test_user = user_item_matrix.index[0]
        print(f"  Testing user-based recommendations for user {test_user}...")
        user_recs = user_based_recommendations(test_user, user_item_matrix, user_sim_matrix, n=5)
        print(f"  ✅ User-based recommendations: {len(user_recs)}")
        
        # Test item-based recommendations
        test_movie = movies_df['title'].iloc[0]
        print(f"  Testing item-based recommendations for movie '{test_movie}'...")
        item_recs = item_based_recommendations(test_movie, user_item_matrix, movies_df, item_sim_matrix, n=5)
        print(f"  ✅ Item-based recommendations: {len(item_recs)}")
        
        return True, "Collaborative filtering working correctly"
        
    except Exception as e:
        print(f"  ❌ Collaborative filtering error: {e}")
        traceback.print_exc()
        return False, str(e)

def test_matrix_factorization():
    """Test matrix factorization algorithms"""
    print("🔧 Testing Matrix Factorization...")
    try:
        # Initialize database
        db = initialize_database_with_csv()
        user_item_matrix = db.get_user_item_matrix()
        test_user = user_item_matrix.index[0]
        
        # Test SVD
        print("  Testing SVD...")
        svd_model = MatrixFactorization(method='svd', n_components=20)
        svd_model.fit(user_item_matrix)
        svd_recs = svd_model.get_user_recommendations(test_user, n_recommendations=5)
        print(f"  ✅ SVD recommendations: {len(svd_recs)}")
        print(f"  First recommendation: {svd_recs[0] if svd_recs else 'None'}")
        
        # Test NMF
        print("  Testing NMF...")
        nmf_model = MatrixFactorization(method='nmf', n_components=20)
        nmf_model.fit(user_item_matrix)
        nmf_recs = nmf_model.get_user_recommendations(test_user, n_recommendations=5)
        print(f"  ✅ NMF recommendations: {len(nmf_recs)}")
        print(f"  First recommendation: {nmf_recs[0] if nmf_recs else 'None'}")
        
        return True, "Matrix factorization working correctly"
        
    except Exception as e:
        print(f"  ❌ Matrix factorization error: {e}")
        traceback.print_exc()
        return False, str(e)

def test_hybrid_system():
    """Test hybrid recommendation system"""
    print("🔧 Testing Hybrid System...")
    try:
        # Initialize components
        db = initialize_database_with_csv()
        user_item_matrix = db.get_user_item_matrix()
        movies_df = db.get_movies_info()
        
        # Create SVD model for hybrid system
        svd_model = MatrixFactorization(method='svd', n_components=20)
        svd_model.fit(user_item_matrix)
        
        # Create hybrid system
        print("  Creating hybrid system...")
        hybrid_system = create_hybrid_system(user_item_matrix, movies_df, mf_model=svd_model)
        print("  ✅ Hybrid system created")
        
        # Test recommendations
        test_user = user_item_matrix.index[0]
        print(f"  Testing hybrid recommendations for user {test_user}...")
        hybrid_recs = hybrid_system.get_hybrid_recommendations(test_user, n_recommendations=5)
        print(f"  ✅ Hybrid recommendations: {len(hybrid_recs)}")
        print(f"  First recommendation: {hybrid_recs[0] if hybrid_recs else 'None'}")
        
        # Test diversity calculation
        print("  Testing diversity calculation...")
        diversity = hybrid_system.get_recommendation_diversity(hybrid_recs)
        print(f"  ✅ Diversity score: {diversity:.3f}")
        
        return True, "Hybrid system working correctly"
        
    except Exception as e:
        print(f"  ❌ Hybrid system error: {e}")
        traceback.print_exc()
        return False, str(e)

def test_ab_testing():
    """Test A/B testing framework"""
    print("🔧 Testing A/B Testing...")
    try:
        # Initialize components
        db = initialize_database_with_csv()
        user_item_matrix = db.get_user_item_matrix()
        
        # Create models
        svd_model = MatrixFactorization(method='svd', n_components=20)
        svd_model.fit(user_item_matrix)
        
        nmf_model = MatrixFactorization(method='nmf', n_components=20)
        nmf_model.fit(user_item_matrix)
        
        # Create A/B framework
        print("  Creating A/B testing framework...")
        ab_framework = ABTestingFramework(db)
        print("  ✅ A/B framework created")
        
        # Create experiment
        print("  Creating test experiment...")
        algorithms = {
            'variant_a': {'model': svd_model, 'params': {}},
            'variant_b': {'model': nmf_model, 'params': {}}
        }
        traffic_split = {'variant_a': 0.5, 'variant_b': 0.5}
        
        experiment = ab_framework.create_experiment(
            experiment_name='debug_test',
            algorithms=algorithms,
            traffic_split=traffic_split
        )
        print("  ✅ Experiment created")
        
        # Test getting recommendations
        test_user = user_item_matrix.index[0]
        print(f"  Testing A/B recommendations for user {test_user}...")
        recommendations, variant, metadata = ab_framework.get_recommendations_for_experiment(
            'debug_test', test_user, 5
        )
        print(f"  ✅ A/B recommendations: {len(recommendations)} from variant {variant}")
        
        return True, "A/B testing working correctly"
        
    except Exception as e:
        print(f"  ❌ A/B testing error: {e}")
        traceback.print_exc()
        return False, str(e)

def main():
    """Run all debugging tests"""
    print("🎬 Algorithm Debugging Suite")
    print("=" * 50)
    
    results = {}
    
    # Test each component
    results['collaborative'] = test_collaborative_filtering()
    print()
    
    results['matrix_factorization'] = test_matrix_factorization()
    print()
    
    results['hybrid'] = test_hybrid_system()
    print()
    
    results['ab_testing'] = test_ab_testing()
    print()
    
    # Summary
    print("=" * 50)
    print("📋 Debug Results Summary:")
    for component, (success, message) in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {component.title()}: {message}")
    
    working_count = sum(1 for success, _ in results.values() if success)
    print(f"\n🎯 {working_count}/4 components working correctly!")
    
    if working_count < 4:
        print("\n⚠️ Issues found! Check error messages above for details.")
    else:
        print("\n🎉 All components working! The issue might be in the Streamlit UI.")

if __name__ == "__main__":
    main()
