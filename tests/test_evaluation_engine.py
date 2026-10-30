"""Unit tests for the Model Evaluation & Error Analysis Engine."""

import unittest
from src.cropcare.evaluation_engine import (
    BENCHMARK_SUMMARY,
    CROP_WISE_BENCHMARKS,
    CLASS_CONFUSION_PAIRS,
    STRESS_TEST_ABLATION,
    get_benchmark_summary,
    get_crop_wise_benchmarks,
    get_top_confusion_pairs,
    get_stress_test_ablation,
)


class TestEvaluationEngine(unittest.TestCase):
    """Test retrieval and structural integrity of benchmark metrics."""

    def test_benchmark_summary_metrics(self):
        summary = get_benchmark_summary()
        self.assertGreater(summary["overall_accuracy"], 95.0)
        self.assertGreater(summary["macro_f1"], 95.0)
        self.assertEqual(summary["total_classes"], 38)
        self.assertEqual(summary["total_crops"], 14)
        self.assertIn("MobileNetV3", summary["model_name"])

    def test_crop_wise_benchmarks(self):
        crop_metrics = get_crop_wise_benchmarks()
        self.assertIn("Tomato", crop_metrics)
        self.assertIn("Potato", crop_metrics)
        self.assertIn("Corn_(maize)", crop_metrics)
        self.assertEqual(crop_metrics["Tomato"]["num_classes"], 10)
        for crop, meta in crop_metrics.items():
            self.assertIn("accuracy", meta)
            self.assertIn("f1_score", meta)
            self.assertIn("top_confusion", meta)

    def test_confusion_pairs(self):
        pairs = get_top_confusion_pairs()
        self.assertGreaterEqual(len(pairs), 4)
        first_pair = pairs[0]
        self.assertIn("true_class", first_pair)
        self.assertIn("predicted_class", first_pair)
        self.assertIn("biological_root_cause", first_pair)
        self.assertIn("mitigation_strategy", first_pair)

    def test_stress_test_ablation_categories(self):
        ablation = get_stress_test_ablation()
        self.assertIn("lighting_variations", ablation)
        self.assertIn("image_quality_blur", ablation)
        self.assertIn("unseen_conditions_ood", ablation)
        lighting = ablation["lighting_variations"]
        self.assertIn("experiments", lighting)
        self.assertGreaterEqual(len(lighting["experiments"]), 3)


if __name__ == "__main__":
    unittest.main()

