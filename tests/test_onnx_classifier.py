import numpy as np

from src.onnx_classifier import IMAGENET_MEAN, IMAGENET_STD, preprocess_bgr, softmax, top_prediction


def test_preprocess_returns_nchw_float32() -> None:
    frame = np.zeros((20, 10, 3), dtype=np.uint8)
    output = preprocess_bgr(frame)
    assert output.shape == (1, 3, 224, 224)
    assert output.dtype == np.float32


def test_preprocess_converts_bgr_to_rgb_and_normalizes() -> None:
    # BGR blue becomes RGB [0, 0, 1].
    frame = np.array([[[255, 0, 0]]], dtype=np.uint8)
    output = preprocess_bgr(frame, 1, 1)[0, :, 0, 0]
    expected_rgb = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    expected = (expected_rgb - IMAGENET_MEAN) / IMAGENET_STD
    assert np.allclose(output, expected)


def test_softmax_is_stable_and_sums_to_one() -> None:
    probabilities = softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert np.isclose(probabilities.sum(), 1.0)
    assert int(np.argmax(probabilities)) == 2


def test_top_prediction_maps_index_to_label() -> None:
    label, confidence, index = top_prediction(
        np.array([0.1, 0.8, 0.1]), ["a", "b", "c"]
    )
    assert label == "b"
    assert index == 1
    assert confidence == 0.8
