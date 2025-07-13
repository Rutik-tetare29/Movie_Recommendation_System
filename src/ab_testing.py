# src/ab_testing.py

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import json
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ABTestingFramework:
    """
    A/B Testing Framework for Movie Recommendation Systems
    Allows comparison of different recommendation algorithms and parameters
    """
    
    def __init__(self, db_manager=None):
        """
        Initialize A/B Testing Framework
        
        Args:
            db_manager: Database manager instance for logging
        """
        self.db_manager = db_manager
        self.experiments = {}
        self.user_assignments = {}
        self.results = {}
        
        print("🧪 A/B Testing Framework initialized")
    
    def create_experiment(self, experiment_name, algorithms, traffic_split=None, 
                         start_date=None, end_date=None, success_metrics=None):
        """
        Create a new A/B test experiment
        
        Args:
            experiment_name (str): Name of the experiment
            algorithms (dict): Dictionary of algorithm configurations
                              Format: {'variant_name': {'model': model_instance, 'params': dict}}
            traffic_split (dict): Traffic allocation percentages
            start_date (datetime): Experiment start date
            end_date (datetime): Experiment end date
            success_metrics (list): List of metrics to track
        """
        if traffic_split is None:
            # Equal split among variants
            split_size = 1.0 / len(algorithms)
            traffic_split = {variant: split_size for variant in algorithms.keys()}
        
        # Normalize traffic split
        total_traffic = sum(traffic_split.values())
        traffic_split = {k: v/total_traffic for k, v in traffic_split.items()}
        
        if success_metrics is None:
            success_metrics = ['click_through_rate', 'rating_accuracy', 'diversity', 'coverage']
        
        experiment = {
            'name': experiment_name,
            'algorithms': algorithms,
            'traffic_split': traffic_split,
            'start_date': start_date or datetime.now(),
            'end_date': end_date or (datetime.now() + timedelta(days=30)),
            'success_metrics': success_metrics,
            'status': 'active',
            'created_at': datetime.now(),
            'user_assignments': {},
            'results': {variant: {} for variant in algorithms.keys()}
        }
        
        self.experiments[experiment_name] = experiment
        
        print(f"✅ Created experiment '{experiment_name}' with variants: {list(algorithms.keys())}")
        print(f"📊 Traffic split: {traffic_split}")
        
        return experiment
    
    def assign_user_to_variant(self, experiment_name, user_id):
        """
        Assign a user to a specific variant for an experiment
        
        Args:
            experiment_name (str): Name of the experiment
            user_id: User ID
            
        Returns:
            str: Assigned variant name
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        experiment = self.experiments[experiment_name]
        
        # Check if user is already assigned
        if user_id in experiment['user_assignments']:
            return experiment['user_assignments'][user_id]
        
        # Check if experiment is active
        now = datetime.now()
        if now < experiment['start_date'] or now > experiment['end_date']:
            # Return control variant if experiment is not active
            variants = list(experiment['algorithms'].keys())
            return variants[0] if variants else None
        
        # Assign user to variant based on traffic split
        random.seed(hash(str(user_id) + experiment_name) % (2**32))
        rand_value = random.random()
        
        cumulative_split = 0
        assigned_variant = None
        
        for variant, split in experiment['traffic_split'].items():
            cumulative_split += split
            if rand_value <= cumulative_split:
                assigned_variant = variant
                break
        
        if assigned_variant is None:
            assigned_variant = list(experiment['algorithms'].keys())[0]
        
        # Store assignment
        experiment['user_assignments'][user_id] = assigned_variant
        
        return assigned_variant
    
    def get_recommendations_for_experiment(self, experiment_name, user_id, n_recommendations=10):
        """
        Get recommendations for a user in an A/B test experiment
        
        Args:
            experiment_name (str): Name of the experiment
            user_id: User ID
            n_recommendations (int): Number of recommendations
            
        Returns:
            tuple: (recommendations, variant_used, metadata)
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        # Assign user to variant
        variant = self.assign_user_to_variant(experiment_name, user_id)
        experiment = self.experiments[experiment_name]
        
        # Get algorithm for this variant
        algorithm_config = experiment['algorithms'][variant]
        model = algorithm_config['model']
        params = algorithm_config.get('params', {})
        
        # Generate recommendations
        try:
            if hasattr(model, 'get_user_recommendations'):
                recommendations = model.get_user_recommendations(user_id, n_recommendations, **params)
            elif hasattr(model, 'get_hybrid_recommendations'):
                recommendations = model.get_hybrid_recommendations(user_id, n_recommendations, **params)
            else:
                # Fallback for simple models
                recommendations = []
            
            metadata = {
                'experiment': experiment_name,
                'variant': variant,
                'user_id': user_id,
                'timestamp': datetime.now(),
                'n_recommendations': len(recommendations)
            }
            
            return recommendations, variant, metadata
            
        except Exception as e:
            print(f"⚠️ Error generating recommendations for experiment {experiment_name}, variant {variant}: {e}")
            return [], variant, {}
    
    def log_user_interaction(self, experiment_name, user_id, movie_id, interaction_type, 
                           interaction_value=1.0, variant=None):
        """
        Log user interaction for A/B testing
        
        Args:
            experiment_name (str): Name of the experiment
            user_id: User ID
            movie_id: Movie ID
            interaction_type (str): Type of interaction
            interaction_value (float): Value of interaction
            variant (str): Variant name (optional, will be looked up if not provided)
        """
        if experiment_name not in self.experiments:
            return
        
        if variant is None:
            variant = self.experiments[experiment_name]['user_assignments'].get(user_id)
        
        if variant is None:
            return
        
        # Initialize results structure if needed
        experiment = self.experiments[experiment_name]
        if variant not in experiment['results']:
            experiment['results'][variant] = {}
        
        if 'interactions' not in experiment['results'][variant]:
            experiment['results'][variant]['interactions'] = []
        
        # Log interaction
        interaction = {
            'user_id': user_id,
            'movie_id': movie_id,
            'interaction_type': interaction_type,
            'interaction_value': interaction_value,
            'timestamp': datetime.now().isoformat(),
            'variant': variant
        }
        
        experiment['results'][variant]['interactions'].append(interaction)
        
        # Log to database if available
        if self.db_manager:
            try:
                self.db_manager.log_user_interaction(user_id, movie_id, interaction_type, interaction_value)
                
                # Log experiment-specific data
                self.db_manager.add_user_feedback(
                    user_id=user_id,
                    movie_id=movie_id,
                    algorithm=f"{experiment_name}_{variant}",
                    feedback_type=interaction_type,
                    feedback_value=interaction_value,
                    comments=f"A/B Test: {experiment_name}"
                )
            except Exception as e:
                print(f"⚠️ Error logging to database: {e}")
    
    def calculate_experiment_metrics(self, experiment_name):
        """
        Calculate metrics for all variants in an experiment
        
        Args:
            experiment_name (str): Name of the experiment
            
        Returns:
            dict: Metrics for each variant
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        experiment = self.experiments[experiment_name]
        metrics = {}
        
        for variant, variant_results in experiment['results'].items():
            variant_metrics = {}
            interactions = variant_results.get('interactions', [])
            
            if not interactions:
                metrics[variant] = variant_metrics
                continue
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(interactions)
            
            # Basic metrics
            variant_metrics['total_interactions'] = len(interactions)
            variant_metrics['unique_users'] = df['user_id'].nunique()
            variant_metrics['unique_movies'] = df['movie_id'].nunique()
            
            # Click-through rate
            clicks = df[df['interaction_type'] == 'click']
            views = df[df['interaction_type'] == 'view']
            if len(views) > 0:
                variant_metrics['click_through_rate'] = len(clicks) / len(views)
            else:
                variant_metrics['click_through_rate'] = 0
            
            # Average rating (if available)
            ratings = df[df['interaction_type'] == 'rating']
            if len(ratings) > 0:
                variant_metrics['average_rating'] = ratings['interaction_value'].mean()
                variant_metrics['rating_std'] = ratings['interaction_value'].std()
            
            # Engagement rate
            engaged_interactions = df[df['interaction_type'].isin(['click', 'like', 'rating'])]
            if len(df) > 0:
                variant_metrics['engagement_rate'] = len(engaged_interactions) / len(df)
            else:
                variant_metrics['engagement_rate'] = 0
            
            # Time-based metrics
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if len(df) > 1:
                time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600  # hours
                variant_metrics['interactions_per_hour'] = len(df) / max(time_span, 1)
            
            metrics[variant] = variant_metrics
        
        return metrics
    
    def run_statistical_test(self, experiment_name, metric_name, alpha=0.05):
        """
        Run statistical significance test between variants
        
        Args:
            experiment_name (str): Name of the experiment
            metric_name (str): Metric to test
            alpha (float): Significance level
            
        Returns:
            dict: Statistical test results
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        experiment = self.experiments[experiment_name]
        variants = list(experiment['algorithms'].keys())
        
        if len(variants) < 2:
            return {"error": "Need at least 2 variants for statistical testing"}
        
        # Get metric values for each variant
        variant_values = {}
        
        for variant in variants:
            interactions = experiment['results'][variant].get('interactions', [])
            
            if metric_name == 'click_through_rate':
                clicks = sum(1 for i in interactions if i['interaction_type'] == 'click')
                views = sum(1 for i in interactions if i['interaction_type'] == 'view')
                variant_values[variant] = clicks / max(views, 1)
                
            elif metric_name == 'average_rating':
                ratings = [i['interaction_value'] for i in interactions if i['interaction_type'] == 'rating']
                variant_values[variant] = np.mean(ratings) if ratings else 0
                
            elif metric_name == 'engagement_rate':
                engaged = sum(1 for i in interactions if i['interaction_type'] in ['click', 'like', 'rating'])
                total = len(interactions)
                variant_values[variant] = engaged / max(total, 1)
        
        # Perform statistical tests
        results = {}
        
        if len(variants) == 2:
            # Two-sample t-test for two variants
            variant_a, variant_b = variants[0], variants[1]
            
            # Get sample data
            data_a = self._get_metric_samples(experiment_name, variant_a, metric_name)
            data_b = self._get_metric_samples(experiment_name, variant_b, metric_name)
            
            if len(data_a) > 1 and len(data_b) > 1:
                t_stat, p_value = stats.ttest_ind(data_a, data_b)
                
                results = {
                    'test_type': 'two_sample_t_test',
                    'variants': [variant_a, variant_b],
                    'metric': metric_name,
                    'variant_means': {variant_a: np.mean(data_a), variant_b: np.mean(data_b)},
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'is_significant': p_value < alpha,
                    'alpha': alpha,
                    'sample_sizes': {variant_a: len(data_a), variant_b: len(data_b)}
                }
        else:
            # ANOVA for multiple variants
            sample_groups = []
            for variant in variants:
                data = self._get_metric_samples(experiment_name, variant, metric_name)
                if len(data) > 0:
                    sample_groups.append(data)
            
            if len(sample_groups) >= 2 and all(len(group) > 1 for group in sample_groups):
                f_stat, p_value = stats.f_oneway(*sample_groups)
                
                results = {
                    'test_type': 'anova',
                    'variants': variants,
                    'metric': metric_name,
                    'variant_means': {v: variant_values[v] for v in variants},
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'is_significant': p_value < alpha,
                    'alpha': alpha
                }
        
        return results
    
    def _get_metric_samples(self, experiment_name, variant, metric_name):
        """Get sample data for a specific metric and variant"""
        experiment = self.experiments[experiment_name]
        interactions = experiment['results'][variant].get('interactions', [])
        
        if metric_name == 'click_through_rate':
            # Return binary values for CTR
            samples = []
            for interaction in interactions:
                if interaction['interaction_type'] in ['view', 'click']:
                    samples.append(1 if interaction['interaction_type'] == 'click' else 0)
            return samples
            
        elif metric_name == 'average_rating':
            return [i['interaction_value'] for i in interactions if i['interaction_type'] == 'rating']
            
        elif metric_name == 'engagement_rate':
            # Return binary values for engagement
            samples = []
            for interaction in interactions:
                engaged = interaction['interaction_type'] in ['click', 'like', 'rating']
                samples.append(1 if engaged else 0)
            return samples
        
        return []
    
    def generate_experiment_report(self, experiment_name):
        """
        Generate a comprehensive report for an experiment
        
        Args:
            experiment_name (str): Name of the experiment
            
        Returns:
            dict: Detailed experiment report
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        experiment = self.experiments[experiment_name]
        
        # Calculate metrics
        metrics = self.calculate_experiment_metrics(experiment_name)
        
        # Run statistical tests
        statistical_tests = {}
        for metric in ['click_through_rate', 'average_rating', 'engagement_rate']:
            test_result = self.run_statistical_test(experiment_name, metric)
            if 'error' not in test_result:
                statistical_tests[metric] = test_result
        
        # Generate recommendations
        recommendations = self._generate_recommendations(experiment_name, metrics, statistical_tests)
        
        report = {
            'experiment_name': experiment_name,
            'experiment_config': {
                'start_date': experiment['start_date'].isoformat(),
                'end_date': experiment['end_date'].isoformat(),
                'traffic_split': experiment['traffic_split'],
                'variants': list(experiment['algorithms'].keys())
            },
            'metrics': metrics,
            'statistical_tests': statistical_tests,
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def _generate_recommendations(self, experiment_name, metrics, statistical_tests):
        """Generate recommendations based on experiment results"""
        recommendations = []
        
        # Find best performing variant for each metric
        best_variants = {}
        for metric_name, test_result in statistical_tests.items():
            if test_result.get('is_significant', False):
                variant_means = test_result.get('variant_means', {})
                if variant_means:
                    best_variant = max(variant_means.items(), key=lambda x: x[1])
                    best_variants[metric_name] = best_variant[0]
        
        if best_variants:
            # Check if one variant consistently performs best
            variant_counts = {}
            for variant in best_variants.values():
                variant_counts[variant] = variant_counts.get(variant, 0) + 1
            
            most_winning_variant = max(variant_counts.items(), key=lambda x: x[1])
            
            if most_winning_variant[1] >= len(best_variants) * 0.6:  # 60% of metrics
                recommendations.append({
                    'type': 'winner',
                    'variant': most_winning_variant[0],
                    'reason': f"Performs best in {most_winning_variant[1]} out of {len(best_variants)} significant metrics"
                })
            else:
                recommendations.append({
                    'type': 'mixed_results',
                    'reason': "No single variant dominates all metrics",
                    'best_per_metric': best_variants
                })
        else:
            recommendations.append({
                'type': 'no_significant_difference',
                'reason': "No statistically significant differences found between variants"
            })
        
        return recommendations
    
    def stop_experiment(self, experiment_name):
        """Stop an active experiment"""
        if experiment_name in self.experiments:
            self.experiments[experiment_name]['status'] = 'stopped'
            self.experiments[experiment_name]['stopped_at'] = datetime.now()
            print(f"🛑 Stopped experiment '{experiment_name}'")
    
    def get_experiment_status(self, experiment_name=None):
        """Get status of experiments"""
        if experiment_name:
            if experiment_name in self.experiments:
                return self.experiments[experiment_name]
            else:
                return None
        else:
            return {name: exp for name, exp in self.experiments.items()}


# Utility functions
def create_simple_ab_test(algorithm_a, algorithm_b, test_name="simple_ab_test", 
                         traffic_split=None, db_manager=None):
    """
    Create a simple A/B test with two algorithms
    
    Args:
        algorithm_a: First algorithm to test
        algorithm_b: Second algorithm to test
        test_name (str): Name of the test
        traffic_split (dict): Traffic allocation
        db_manager: Database manager instance
        
    Returns:
        ABTestingFramework: Configured A/B testing framework
    """
    if traffic_split is None:
        traffic_split = {'variant_a': 0.5, 'variant_b': 0.5}
    
    ab_framework = ABTestingFramework(db_manager)
    
    algorithms = {
        'variant_a': {'model': algorithm_a, 'params': {}},
        'variant_b': {'model': algorithm_b, 'params': {}}
    }
    
    ab_framework.create_experiment(
        experiment_name=test_name,
        algorithms=algorithms,
        traffic_split=traffic_split
    )
    
    return ab_framework
