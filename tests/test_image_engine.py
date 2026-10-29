"""Offline unit tests for Leaf Image Intelligence Module (image_engine.py)."""

from __future__ import annotations

import io
import json
import math
import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from src.cropcare.image_engine import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    check_image_quality,
    predict_image,
    preprocess_image,
)
from src.cropcare.training.train_model import build_model


class TestImageEngineOffline(unittest.TestCase):
    """Unit tests for inference, error handling, contract stability, and uncertainty."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.dummy_mapping_path = os.path.join(self.test_dir, "test_class_mapping.json")
        self.dummy_model_path = os.path.join(self.test_dir, "test_model.pth")

        # Create a mini 3-class test mapping
        self.classes = [
            "Apple___Apple_scab",
            "Apple___healthy",
            "Tomato___Early_blight",
        ]
        self.mapping = {
            "num_classes": 3,
            "classes": self.classes,
            "class_to_idx": {c: i for i, c in enumerate(self.classes)},
            "idx_to_class": {str(i): c for i, c in enumerate(self.classes)},
            "class_details": {
                "Apple___Apple_scab": {"crop": "Apple", "disease": "Apple scab"},
                "Apple___healthy": {"crop": "Apple", "disease": "Healthy"},
                "Tomato___Early_blight": {"crop": "Tomato", "disease": "Early blight"},
            },
        }
        with open(self.dummy_mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f)

        # Build and save a tiny test model
        model = build_model(num_classes=3, pretrained=False)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_name": "MobileNetV3-Small-Test",
            "num_classes": 3,
            "class_to_idx": self.mapping["class_to_idx"],
            "image_size": 224,
            "best_epoch": 1,
            "best_val_accuracy": 0.85,
        }
        torch.save(checkpoint, self.dummy_model_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_class_mapping_integrity(self):
        """Verify official class mapping has 38 unique classes and consistent indices."""
        real_mapping_path = "models/class_mapping.json"
        self.assertTrue(os.path.exists(real_mapping_path))
        with open(real_mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        self.assertEqual(mapping["num_classes"], 38)
        self.assertEqual(len(mapping["classes"]), 38)
        self.assertEqual(len(mapping["class_to_idx"]), 38)

        # Ensure bijective indexing
        for cls_name, idx in mapping["class_to_idx"].items():
            self.assertEqual(mapping["classes"][idx], cls_name)
            self.assertEqual(mapping["idx_to_class"][str(idx)], cls_name)

    def test_missing_checkpoint_graceful_handling(self):
        """Verify missing model file does not crash and returns structured warnings."""
        dummy_img = Image.new("RGB", (224, 224), color=(120, 180, 70))
        result = predict_image(
            dummy_img,
            model_path="non_existent_weights_xyz.pth",
            mapping_path=self.dummy_mapping_path,
        )

        self.assertIsNone(result["predicted_class"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertTrue(result["uncertain"])
        self.assertTrue(any("not found" in w for w in result["warnings"]))

    def test_unreadable_or_corrupt_image_inputs(self):
        """Verify corrupt bytes, non-existent files, and bad types fail safely."""
        # Non-existent file path
        res1 = predict_image("non_existent_image_123.jpg", model_path=self.dummy_model_path, mapping_path=self.dummy_mapping_path)
        self.assertIsNone(res1["predicted_class"])
        self.assertTrue(any("does not exist" in w for w in res1["warnings"]))

        # Corrupt raw bytes
        res2 = predict_image(b"not a valid jpeg or png binary stream", model_path=self.dummy_model_path, mapping_path=self.dummy_mapping_path)
        self.assertIsNone(res2["predicted_class"])
        self.assertTrue(any("Failed to read" in w for w in res2["warnings"]))

        # Unsupported type
        res3 = predict_image(12345, model_path=self.dummy_model_path, mapping_path=self.dummy_mapping_path)
        self.assertIsNone(res3["predicted_class"])
        self.assertTrue(any("Unsupported image input type" in w for w in res3["warnings"]))

    def test_image_quality_heuristics(self):
        """Verify image quality screening flags tiny or blank images."""
        # 1. Blank solid black image
        black_img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        warnings_black = check_image_quality(black_img)
        self.assertTrue(any("visual contrast" in w for w in warnings_black))

        # 2. Extremely tiny image (<32x32)
        tiny_img = Image.new("RGB", (16, 16), color=(50, 120, 40))
        warnings_tiny = check_image_quality(tiny_img)
        self.assertTrue(any("extremely small" in w for w in warnings_tiny))

        # 3. Normal leaf-like image with rich contrast
        np_arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        normal_img = Image.fromarray(np_arr)
        warnings_normal = check_image_quality(normal_img)
        self.assertEqual(len(warnings_normal), 0)

    def test_preprocessing_tensor_dimensions(self):
        """Verify preprocessing yields standard (1, 3, 224, 224) float32 tensor."""
        img = Image.new("RGB", (300, 400), color=(80, 160, 90))
        tensor = preprocess_image(img, image_size=224)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_valid_inference_with_test_model(self):
        """Verify successful end-to-end inference using real weights and mapping."""
        # Create a synthetic leaf-like image with varying pixel intensities
        np_arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        leaf_img = Image.fromarray(np_arr)

        result = predict_image(
            leaf_img,
            model_path=self.dummy_model_path,
            mapping_path=self.dummy_mapping_path,
        )

        # Validate required contract keys
        for key in ["predicted_class", "crop", "disease", "confidence", "probabilities", "uncertain", "model_name", "warnings"]:
            self.assertIn(key, result)

        self.assertIn(result["predicted_class"], self.classes)
        self.assertIsInstance(result["confidence"], float)
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        self.assertEqual(len(result["probabilities"]), 3)

        # Check probabilities sum to approximately 1.0
        prob_sum = sum(result["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

        # Check all probabilities are finite
        for p in result["probabilities"].values():
            self.assertTrue(math.isfinite(p))

    def test_uncertainty_threshold_flagging(self):
        """Verify predictions below confidence threshold trigger uncertain=True and warning."""
        leaf_img = Image.new("RGB", (224, 224), color=(60, 140, 50))

        # Test with very high threshold (e.g. 0.99) -> must trigger uncertainty
        res_high_thresh = predict_image(
            leaf_img,
            model_path=self.dummy_model_path,
            mapping_path=self.dummy_mapping_path,
            confidence_threshold=0.999,
        )
        self.assertTrue(res_high_thresh["uncertain"])
        self.assertTrue(any("Low prediction confidence" in w for w in res_high_thresh["warnings"]))

        # Test with very low threshold (0.01) -> should not trigger low confidence warning
        res_low_thresh = predict_image(
            leaf_img,
            model_path=self.dummy_model_path,
            mapping_path=self.dummy_mapping_path,
            confidence_threshold=0.01,
        )
        self.assertFalse(res_low_thresh["uncertain"])


if __name__ == "__main__":
    unittest.main()
