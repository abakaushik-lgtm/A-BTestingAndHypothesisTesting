import unittest
import pandas as pd
import numpy as np

# Import our modules
from data_generator import generate_experiment_data
from data_cleaner import inspect_data_quality, clean_experiment_data
from stats_engine import run_z_test, run_chi_square_test, run_t_test, analyze_segments
from recommendation_engine import calculate_revenue_impact, generate_recommendations
from ml_module import perform_user_segmentation, get_conversion_drivers

class TestABTestingPlatform(unittest.TestCase):
    
    def setUp(self):
        # Generate a small synthetic dataset for testing
        self.df = generate_experiment_data(
            num_users=1000,
            conversion_rate_a=0.08,
            conversion_rate_b=0.10,
            inject_anomalies=True,
            random_seed=42
        )

    def test_data_generation_and_cleaning(self):
        self.assertTrue(len(self.df) > 0)
        self.assertIn("User ID", self.df.columns)
        self.assertIn("Variant", self.df.columns)
        
        # Quality report should find anomalies since inject_anomalies=True
        report = inspect_data_quality(self.df)
        self.assertGreaterEqual(report["total_records"], 1000)
        
        # Clean the dataset
        cleaned, summary = clean_experiment_data(self.df)
        self.assertTrue(len(cleaned) > 0)
        self.assertGreaterEqual(summary["duplicates_removed"], 0)
        
        # Leaked users should be gone
        clean_report = inspect_data_quality(cleaned)
        self.assertEqual(clean_report["leaked_users"], 0)
        self.assertEqual(clean_report["negative_revenue"], 0)
        self.assertEqual(clean_report["duplicate_records"], 0)

    def test_statistical_engine(self):
        # Test Z-test function
        # conv_a=80, n_a=1000 (8% CR), conv_b=100, n_b=1000 (10% CR)
        z_res = run_z_test(80, 1000, 100, 1000, alpha=0.05)
        self.assertAlmostEqual(z_res["rate_a"], 0.08)
        self.assertAlmostEqual(z_res["rate_b"], 0.10)
        self.assertAlmostEqual(z_res["abs_lift"], 0.02)
        self.assertAlmostEqual(z_res["rel_lift"], 0.25)
        self.assertTrue(z_res["z_stat"] > 0)
        
        # Test Chi-Square test function
        chi_res = run_chi_square_test(80, 1000, 100, 1000)
        self.assertIn("chi2_stat", chi_res)
        self.assertIn("p_value", chi_res)
        
        # Test Welch's T-Test function
        vals_a = np.random.normal(50, 10, 500)
        vals_b = np.random.normal(55, 10, 500)
        t_res = run_t_test(vals_a, vals_b, alpha=0.05)
        self.assertTrue(t_res["significant"])
        self.assertGreater(t_res["t_stat"], 0)

    def test_recommendation_and_impact(self):
        # Proportions: A=8%, B=10%, Traffic=100k, AOV=$75
        impact = calculate_revenue_impact(100000, 0.08, 0.10, 75.0)
        self.assertAlmostEqual(impact["abs_lift"], 0.02)
        self.assertEqual(impact["monthly_conversions_gain"], 2000)
        self.assertEqual(impact["monthly_revenue_gain"], 150000.0)
        
        # Check recommendations generator
        z_res = run_z_test(80, 1000, 120, 1000, alpha=0.05) # Significant B > A
        recs = generate_recommendations(z_res, revenue_impact=impact)
        self.assertEqual(recs["decision"], "DEPLOY_B")
        self.assertTrue(any("deploy" in text.lower() for text in recs["rationale"] + [recs["decision_text"]]))

    def test_ml_utilities(self):
        cleaned, _ = clean_experiment_data(self.df)
        
        # Test K-Means segmentation
        user_df, cl_summary = perform_user_segmentation(cleaned, n_clusters=3)
        self.assertIn("Segment Name", user_df.columns)
        self.assertEqual(len(cl_summary), 3)
        
        # Test feature importances
        importances = get_conversion_drivers(cleaned)
        self.assertTrue(len(importances) > 0)
        self.assertIn("importance", importances[0])

if __name__ == "__main__":
    unittest.main()
