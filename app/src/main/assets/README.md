# Depth Estimation Models

## MiDaS 2.1:
Repo: <https://github.com/isl-org/MiDaS>
Download source: <https://aihub.qualcomm.com/models/midas>

### midas_v2_1_256x256.tflite
Input shape: float32[1, 256, 256, 3]

### midas_v2_1_256x256.onnx
(from experience for slower on mobile!)
Input shape: float32[1, 3, 256, 256]

### midas_v2_1_256x256_quantized.tflite
Input shape: uint8[1, 256, 256, 3]

### midas_v2_1_256x256_quantized.onnx
Input shape: uint8[1, 3, 256, 256]


## Depth Anything V2:
Repo: <https://github.com/DepthAnything/Depth-Anything-V2>

### depth_anything_v2_vits_210x210.onnx
Generated from: <https://github.com/fabio-sim/Depth-Anything-ONNX>

Input shape: float32[1, 3, 210, 210]