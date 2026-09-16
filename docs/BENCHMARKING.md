# Jetson benchmark plan

A useful edge benchmark must be tied to an exact hardware/software environment.

Record:

- Jetson Nano model / power mode;
- JetPack and L4T releases;
- Python, OpenCV and ONNX Runtime versions;
- ONNX Runtime execution provider(s);
- model SHA-256, file size, input shape and dtype;
- camera resolution;
- warm-up count;
- at least 100 model-only inference samples;
- p50/p95/p99 inference latency;
- end-to-end capture/preprocess/inference/render FPS;
- CPU/GPU utilization, RAM, temperature and throttling state;
- evaluation accuracy on representative data.

The historical sample timer around `session.run(...)` is a good model-only starting point, but one observed frame is not a benchmark. This repository intentionally does not publish invented Jetson FPS values.
