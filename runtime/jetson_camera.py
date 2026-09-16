from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort

from src.onnx_classifier import preprocess_bgr, read_labels, softmax, top_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Jetson/OpenCV ONNX camera classifier")
    parser.add_argument("--model", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    labels = read_labels(args.labels)
    session = ort.InferenceSession(args.model)
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError(f"unable to open camera index {args.camera}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("camera opened but frame capture failed")

            input_data = preprocess_bgr(frame, args.width, args.height)
            started = time.perf_counter()
            outputs = session.run([model_output.name], {model_input.name: input_data})
            inference_ms = (time.perf_counter() - started) * 1000

            scores = np.asarray(outputs[0]).reshape(-1)
            probabilities = softmax(scores)
            label, confidence, _ = top_prediction(probabilities, labels)
            display = label if confidence >= args.threshold else "uncertain"

            cv2.putText(frame, f"{display}: {confidence:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"inference: {inference_ms:.1f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Jetson Nano ONNX Demo", frame)

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
