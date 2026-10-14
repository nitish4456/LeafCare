"""
LeafCare Comprehensive Test Suite
Audits:
1. Model loading, weight shapes, class order verification
2. Image preprocessing (JPG, JPEG, PNG, Grayscale, RGBA) and invalid image handling
3. Prediction execution, confidence, uncertainty flags, and score distribution
4. NLP queries across all intents (symptoms, causes, prevention, management)
5. Custom questions, general overview, and unsupported queries
6. Error handling for missing files and corrupt input
"""

import os
import json
import torch
import numpy as np
from PIL import Image
import io

from model import load_model, preprocess_image, predict, load_metadata
from nlp_assistant import NLPAssistant


def run_audit():
    print("==================================================================")
    print("LEAFCARE AUDIT SUITE: COMPUTATIONAL & LOGICAL VERIFICATION")
    print("==================================================================")

    # -------------------------------------------------------------
    # Test 1: Metadata & Class Order Verification
    # -------------------------------------------------------------
    print("\n--- TEST 1: METADATA & CLASS ORDER ---")
    meta = load_metadata("metadata.json")
    expected_classes = [
        "Tomato___healthy",
        "Tomato___Early_blight",
        "Tomato___Late_blight"
    ]
    assert meta["classes"] == expected_classes, f"Class mismatch: {meta['classes']}"
    assert meta["image_size"] == 224
    assert meta["architecture"] == "ResNet18"
    assert meta["normalization_mean"] == [0.485, 0.456, 0.406]
    assert meta["normalization_std"] == [0.229, 0.224, 0.225]
    print("[PASS] metadata.json validated.")
    print("  Class index 0:", meta["classes"][0])
    print("  Class index 1:", meta["classes"][1])
    print("  Class index 2:", meta["classes"][2])

    # -------------------------------------------------------------
    # Test 2: Model Architecture & Weights Check
    # -------------------------------------------------------------
    print("\n--- TEST 2: MODEL ARCHITECTURE & WEIGHTS CHECK ---")
    weights_path = "leafcare_resnet18.pth"
    file_size_bytes = os.path.getsize(weights_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"  Weights file size: {file_size_bytes} bytes ({file_size_mb:.2f} MB)")
    assert file_size_mb < 50, f"Weights file exceeds 50MB: {file_size_mb:.2f} MB"

    model, loaded_meta = load_model(weights_path, "metadata.json", device="cpu")
    assert model.fc.out_features == 3, f"fc layer output features {model.fc.out_features} != 3"
    assert model.fc.in_features == 512, f"fc layer input features {model.fc.in_features} != 512"
    print(f"[PASS] ResNet-18 initialized and weights loaded successfully.")
    print("  fc.weight shape:", model.fc.weight.shape)
    print("  fc.bias shape:", model.fc.bias.shape)

    # -------------------------------------------------------------
    # Test 3: Preprocessing on Multiple Image Formats & Channels
    # -------------------------------------------------------------
    print("\n--- TEST 3: IMAGE PREPROCESSING ACROSS FORMATS & CHANNELS ---")
    # Test RGB (JPEG)
    rgb_arr = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
    rgb_img = Image.fromarray(rgb_arr)
    t_rgb, disp_rgb = preprocess_image(rgb_img, meta)
    assert t_rgb.shape == (1, 3, 224, 224), f"Unexpected tensor shape {t_rgb.shape}"
    print("  RGB Image -> Tensor shape:", t_rgb.shape, "[OK]")

    # Test RGBA (PNG with alpha channel)
    rgba_arr = (np.random.rand(300, 400, 4) * 255).astype(np.uint8)
    rgba_img = Image.fromarray(rgba_arr, mode="RGBA")
    t_rgba, disp_rgba = preprocess_image(rgba_img, meta)
    assert t_rgba.shape == (1, 3, 224, 224)
    print("  RGBA Image -> Tensor shape:", t_rgba.shape, "[OK]")

    # Test Grayscale
    gray_arr = (np.random.rand(300, 400) * 255).astype(np.uint8)
    gray_img = Image.fromarray(gray_arr, mode="L")
    t_gray, disp_gray = preprocess_image(gray_img, meta)
    assert t_gray.shape == (1, 3, 224, 224)
    print("  Grayscale Image -> Tensor shape:", t_gray.shape, "[OK]")

    # Test BytesIO input
    buf = io.BytesIO()
    rgb_img.save(buf, format="JPEG")
    buf.seek(0)
    t_buf, _ = preprocess_image(buf, meta)
    assert t_buf.shape == (1, 3, 224, 224)
    print("  BytesIO Stream -> Tensor shape:", t_buf.shape, "[OK]")
    print("[PASS] Preprocessing pipeline handles RGB, RGBA, Grayscale, and BytesIO.")

    # -------------------------------------------------------------
    # Test 4: Graceful Error Handling for Corrupt/Invalid Inputs
    # -------------------------------------------------------------
    print("\n--- TEST 4: ERROR HANDLING FOR INVALID / MISSING FILES ---")
    # Invalid image bytes
    corrupt_stream = io.BytesIO(b"This is not a valid JPEG/PNG file.")
    try:
        preprocess_image(corrupt_stream, meta)
        assert False, "Should have raised ValueError on corrupt file"
    except ValueError as ve:
        print("  Corrupt file error handled cleanly:", str(ve)[:60], "...")

    # Missing weights file
    try:
        load_model("non_existent_weights.pth", "metadata.json")
        assert False, "Should have raised FileNotFoundError on missing weights"
    except FileNotFoundError as fnf:
        print("  Missing weights file handled cleanly:", str(fnf)[:60], "...")

    # Missing metadata file
    try:
        load_metadata("non_existent_metadata.json")
        assert False, "Should have raised FileNotFoundError on missing metadata"
    except FileNotFoundError as fnf:
        print("  Missing metadata file handled cleanly:", str(fnf)[:60], "...")
    print("[PASS] Error handling for invalid inputs and missing assets verified.")

    # -------------------------------------------------------------
    # Test 5: Inference on Sample Test Images
    # -------------------------------------------------------------
    print("\n--- TEST 5: MEASURED INFERENCE ON SAMPLE IMAGES ---")
    sample_files = ["healthy_sample.jpg", "early_blight_sample.jpg", "late_blight_sample.jpg"]
    measured_results = {}
    for sf in sample_files:
        path = os.path.join("sample_images", sf)
        tensor, _ = preprocess_image(path, meta)
        res = predict(model, tensor, meta)
        measured_results[sf] = res
        print(f"  {sf}:")
        print(f"    Predicted: {res['display_name']} ({res['raw_class']})")
        print(f"    Confidence: {res['confidence_percentage']}")
        print(f"    Uncertainty Flag: {res['is_uncertain']}")
        for c, p in res["probabilities"].items():
            print(f"      {p['display_name']}: {p['percentage']}")
    print("[PASS] Inference executed cleanly on all sample images.")

    # -------------------------------------------------------------
    # Test 6: NLP Knowledge Base & Query Intent Identification
    # -------------------------------------------------------------
    print("\n--- TEST 6: NLP INTENT & GROUNDED RETRIEVAL AUDIT ---")
    assistant = NLPAssistant("knowledge_base.json")
    
    # 6a. All 4 Intents across all 3 classes
    classes_to_test = ["Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight"]
    intents_to_test = [
        ("What are the symptoms?", "symptoms"),
        ("What causes this disease?", "causes"),
        ("How can I prevent this disease?", "prevention"),
        ("What is the recommended management?", "management")
    ]
    for cls_name in classes_to_test:
        for query_str, expected_intent in intents_to_test:
            ans = assistant.query(query_str, cls_name)
            assert ans["status"] == "success", f"Failed on {query_str} for {cls_name}"
            assert ans["intent"] == expected_intent, f"Expected {expected_intent}, got {ans['intent']}"
            assert len(ans["response"]) > 20
    print("[PASS] All 4 primary intents verified across all 3 classes (12/12 combinations).")

    # 6b. Explicit disease override in query
    override_ans = assistant.query("Why does late blight occur?", "Tomato___healthy")
    assert override_ans["status"] == "success"
    assert override_ans["category"] == "late_blight"
    assert override_ans["intent"] == "causes"
    assert "Phytophthora infestans" in override_ans["response"]
    print("  Category Override Query -> Detected Category: late_blight [OK]")

    # 6c. General overview query
    overview_ans = assistant.query("Tell me about this disease", "Tomato___Early_blight")
    assert overview_ans["status"] == "success"
    assert overview_ans["intent"] == "general_overview"
    assert "Alternaria solani" in overview_ans["response"]
    print("  General Overview Query -> Detected Intent: general_overview [OK]")

    # 6d. Unsupported questions
    unsupported_ans = assistant.query("Can you write a poem about rain?", "Tomato___healthy")
    assert unsupported_ans["status"] == "unsupported"
    assert "could not identify" in unsupported_ans["response"].lower()
    print("  Out-of-Domain Query -> Status: unsupported [OK]")

    # 6e. Missing context
    no_ctx_ans = assistant.query("What are the symptoms?", None)
    assert no_ctx_ans["status"] == "missing_context"
    print("  No Context Query -> Status: missing_context [OK]")

    # 6f. Missing knowledge base file handling
    try:
        NLPAssistant("non_existent_kb.json")
        assert False, "Should have raised FileNotFoundError on missing KB"
    except FileNotFoundError as fnf:
        print("  Missing KB file handled cleanly:", str(fnf)[:60], "...")
    print("[PASS] NLP assistant intent routing, grounded retrieval, and fallbacks validated.")

    print("\n==================================================================")
    print("ALL AUDIT TESTS COMPLETED SUCCESSFULLY WITH ZERO ERRORS.")
    print("==================================================================")


if __name__ == "__main__":
    run_audit()
