# demo_enhanced_features.py

import sys
import os
import pandas as pd
import numpy as np

# Add src to system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Imports from src
from database import initialize_database_with_csv
from matrix_factorization import MatrixFactorization
from hybrid_recommender import create_hybrid_system
from ab_testing import ABTestingFramework

def test_database():
    """Test database functionality"""
    print("🔧 Testing Database...")
    try:
        db = initialize_database_with_csv()
        stats = db.get_system_statistics()
        print(f"✅ Database OK - Users: {stats['total_users']}, Movies: {stats['total_movies']}")
        return db
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return None

def test_matrix_factorization(user_item_matrix):
    """Test matrix factorization"""
    print("🔧 Testing Matrix Factorization...")
    
    # Test SVD
    try:
        svd_model = MatrixFactorization(method='svd', n_components=20)
        svd_model.fit(user_item_matrix)
        print("✅ SVD Model OK")
        
        # Test recommendation
        sample_user = user_item_matrix.index[0]
        recs = svd_model.get_user_recommendations(sample_user, n_recommendations=5)
        print(f"✅ SVD Recommendations: {len(recs)}")
        
    except Exception as e:
        print(f"❌ SVD Error: {e}")
        svd_model = None
    
    # Test NMF
    try:
        nmf_model = MatrixFactorization(method='nmf', n_components=20)
        nmf_model.fit(user_item_matrix)
        print("✅ NMF Model OK")
        
        # Test recommendation
        sample_user = user_item_matrix.index[0]
        recs = nmf_model.get_user_recommendations(sample_user, n_recommendations=5)
        print(f"✅ NMF Recommendations: {len(recs)}")
        
    except Exception as e:
        print(f"❌ NMF Error: {e}")
        nmf_model = None
    
    return svd_model, nmf_model

def test_hybrid_system(user_item_matrix, movies_df, svd_model):
    """Test hybrid recommendation system"""
    print("🔧 Testing Hybrid System...")
    try:
        hybrid_system = create_hybrid_system(user_item_matrix, movies_df, mf_model=svd_model)
        print("✅ Hybrid System Created")
        
        # Test recommendation
        sample_user = user_item_matrix.index[0]
        recs = hybrid_system.get_hybrid_recommendations(sample_user, n_recommendations=5)
        print(f"✅ Hybrid Recommendations: {len(recs)}")
        
        # Test diversity calculation
        try:
            diversity = hybrid_system.get_recommendation_diversity(recs)
            print(f"✅ Diversity Score: {diversity:.3f}")
        except Exception as e:
            print(f"⚠️ Diversity Error: {e}")
        
        return hybrid_system
    except Exception as e:
        print(f"❌ Hybrid Error: {e}")
        return None

def test_ab_testing(db, svd_model, nmf_model):
    """Test A/B testing framework"""
    print("🔧 Testing A/B Testing...")
    try:
        ab_framework = ABTestingFramework(db)
        print("✅ A/B Framework Created")
        
        # Create a simple experiment
        algorithms = {
            'variant_a': {'model': svd_model, 'params': {}},
            'variant_b': {'model': nmf_model, 'params': {}}
        }
        traffic_split = {'variant_a': 0.5, 'variant_b': 0.5}
        
        experiment = ab_framework.create_experiment(
            experiment_name='test_experiment',
            algorithms=algorithms,
            traffic_split=traffic_split
        )
        print("✅ A/B Experiment Created")
        
        return ab_framework
    except Exception as e:
        print(f"❌ A/B Testing Error: {e}")
        return None

def main():
    """Run all tests"""
    print("🎬 Enhanced Movie Recommender - Component Tests")
    print("=" * 50)
    
    # Test Database
    db = test_database()
    if not db:
        print("❌ Cannot continue without database")
        return
    
    # Get data for other tests
    user_item_matrix = db.get_user_item_matrix()
    movies_df = db.get_movies_info()
    print(f"📊 Data loaded: {user_item_matrix.shape[0]} users, {user_item_matrix.shape[1]} movies")
    
    # Test Matrix Factorization
    svd_model, nmf_model = test_matrix_factorization(user_item_matrix)
    
    # Test Hybrid System (only if SVD works)
    hybrid_system = None
    if svd_model:
        hybrid_system = test_hybrid_system(user_item_matrix, movies_df, svd_model)
    
    # Test A/B Testing (only if both models work)
    ab_framework = None
    if svd_model and nmf_model:
        ab_framework = test_ab_testing(db, svd_model, nmf_model)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print(f"  Database: {'✅' if db else '❌'}")
    print(f"  SVD Model: {'✅' if svd_model else '❌'}")
    print(f"  NMF Model: {'✅' if nmf_model else '❌'}")
    print(f"  Hybrid System: {'✅' if hybrid_system else '❌'}")
    print(f"  A/B Testing: {'✅' if ab_framework else '❌'}")
    
    working_components = sum([bool(db), bool(svd_model), bool(nmf_model), bool(hybrid_system), bool(ab_framework)])
    print(f"\n🎯 {working_components}/5 components working properly!")
    
    if working_components == 5:
        print("🎉 All components are working! You can use the enhanced app.")
    else:
        print("⚠️ Some components need fixing. Check error messages above.")

if __name__ == "__main__":
    main()
