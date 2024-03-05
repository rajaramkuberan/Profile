# Traffic Flow Detection

## Install & Dependence

- python
- pytorch
- numpy
- TensorRT
- onnxruntime
- ultralytics - yolov8

## Use

- for Livestream Detection
  ```
  python camera_detection.py
  ```

## Pretrained model

| Model           | Precision | Inference Framework | Download                                                                                                                                                                                                                                     |
| --------------- | --------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| COCO_pretrained | FP16      | TensorRT            | [Engine File](https://github.com/rajaramkuberan/ProjectProfile/tree/main/AI_Projects/Computer_Vision/AI_Sensor_Fusion_Project/CameraSensor/engine/tensorrt) |
| Custom_Model    | FP16      | TensorRT            | [Engine File](https://github.com/rajaramkuberan/ProjectProfile/tree/main/AI_Projects/Computer_Vision/AI_Sensor_Fusion_Project/CameraSensor/engine/tensorrt) |
| Custom_Model    | FP32      | ONNX                | [ONNX File](https://github.com/rajaramkuberan/ProjectProfile/tree/main/AI_Projects/Computer_Vision/AI_Sensor_Fusion_Project/CameraSensor/engine/onnx)                                                                                        |

## Directory Hierarchy

```
|—— atcc_rhs_tp2_v3.py
|—— class.txt
|—— engine
|    |—— onnx
|        |—— best_coco_8n.onnx
|    |—— tensorrt
|        |—— best_8n_fp16.engine
|        |—— best_coco_8n_fp16.engine
|—— models
|    |—— detector.py
|    |—— utils.py
|    |—— __pycache__
|        |—— detector.cpython-38.pyc
|        |—— utils.cpython-38.pyc
|—— ntracker.py
|—— __pycache__
|    |—— ntracker.cpython-38.pyc
```

## Code Details

### Tested Platform

- software
  ```
  OS: Ubuntu 20.04 LTS
  Python: 3.8.5
  Jetpack: 5.1
  TensorRT : 8.5.5
  ```
- hardware
  ```
  Camera: 4MP IR Camera
  GPU: Nvidia Jetson Xavier
  ```

## References

<!--
- [paper-1]()
- [paper-2]()
- [code-1](https://github.com)
- [code-2](https://github.com)

## Citing
