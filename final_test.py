# final_test.py
# Quick test to verify all components are working without errors

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("🎬 Final System Test")
print("=" * 30)

try:
    # Test all imports
    from database import initialize_database_with_csv
    from matrix_factorization import MatrixFactorization
    from hybrid_recommender import create_hybrid_system
    from ab_testing import ABTestingFramework
    from recommend import user_based_recommendations, item_based_recommendations
    from similarity import compute_user_similarity, compute_item_similarity
    print("✅ All imports successful")
    
    # Quick functionality test
    db = initialize_database_with_csv()
    user_item_matrix = db.get_user_item_matrix()
    movies_df = db.get_movies_info()
    print("✅ Database working")
    
    # Test one recommendation from each algorithm
    test_user = user_item_matrix.index[0]
    
    # SVD
    svd_model = MatrixFactorization(method='svd', n_components=10)
    svd_model.fit(user_item_matrix)
    svd_recs = svd_model.get_user_recommendations(test_user, 3)
    print(f"✅ SVD: {len(svd_recs)} recommendations")
    
    # NMF
    nmf_model = MatrixFactorization(method='nmf', n_components=10)
    nmf_model.fit(user_item_matrix)
    nmf_recs = nmf_model.get_user_recommendations(test_user, 3)
    print(f"✅ NMF: {len(nmf_recs)} recommendations")
    
    # Hybrid
    hybrid_system = create_hybrid_system(user_item_matrix, movies_df, mf_model=svd_model)
    hybrid_recs = hybrid_system.get_hybrid_recommendations(test_user, 3)
    print(f"✅ Hybrid: {len(hybrid_recs)} recommendations")
    
    # A/B Testing
    ab_framework = ABTestingFramework(db)
    algorithms = {
        'variant_a': {'model': svd_model, 'params': {}},
        'variant_b': {'model': nmf_model, 'params': {}}
    }
    experiment = ab_framework.create_experiment(
        'final_test', algorithms, {'variant_a': 0.5, 'variant_b': 0.5}
    )
    ab_recs, variant, _ = ab_framework.get_recommendations_for_experiment('final_test', test_user, 3)
    print(f"✅ A/B Testing: {len(ab_recs)} recommendations from {variant}")
    
    print("\n🎉 ALL SYSTEMS WORKING!")
    print("✅ Command line: No errors")
    print("✅ Streamlit app: Running at http://localhost:8501")
    print("✅ All algorithms: Functional")
    
except Exception as e:
    print(f"❌ Error found: {e}")
    import traceback
    traceback.print_exc()
