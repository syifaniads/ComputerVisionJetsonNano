# Provenance

## Retained historical artifacts

- `ComputerVision_Riset_Robotik_JetsonNano.ipynb`
- the repository's 2025 README, which documented transfer of `mobilenet_v2.onnx` to Jetson Nano and an OpenCV + ONNX Runtime camera loop.

The historical README explicitly included 224×224 resizing, BGR→RGB conversion, NCHW layout, ImageNet mean/std normalization, ONNX Runtime inference, softmax, top-label selection, and per-frame inference-time overlay.

## Portfolio extension

The current `src/`, tests, CI, benchmark plan, runtime wrapper, environment inventory script, architecture visual, and limitations document are later engineering improvements. They make the idea reproducible and reviewable without rewriting the historical scope.
