Title
===
Abstract:xxx
## Papar Information
- Title:  `paper name`
- Authors:  `A`,`B`,`C`
- Preprint: [https://arxiv.org/abs/xx]()
- Full-preprint: [paper position]()
- Video: [video position]()

## Install & Dependence
- python
- pytorch
- numpy

## Dataset Preparation
| Dataset | Download |
| ---     | ---   |
| dataset-A | [download]() |
| dataset-B | [download]() |
| dataset-C | [download]() |

## Use
- for train
  ```
  python train.py
  ```
- for test
  ```
  python test.py
  ```
## Pretrained model
| Model | Download |
| ---     | ---   |
| Model-1 | [download]() |
| Model-2 | [download]() |
| Model-3 | [download]() |


## Directory Hierarchy
```
|—— cfg
|    |—— profiles_50.cfg
|    |—— profiles_50_1.cfg
|    |—— profiles_50_10.cfg
|    |—— profiles_50_15.cfg
|    |—— profiles_50_20.cfg
|    |—— profiles_50_5.cfg
|—— datasheet
|    |—— IWR6843ISK.pdf
|—— parsed_data
|    |—— data_20240224-231830.csv
|    |—— data_20240224-232330.csv
|—— radar_parsing
|    |—— mmwave_parse.py
|—— radar_processing
|    |—— bike_lane_model.py
|    |—— centre_lane_model.py
|    |—— dataframe_creation.py
|    |—— Fast_Lane_Final_Model.pkl
|    |—— fast_lane_model.py
|    |—— features_bike_lane.py
|    |—— features_centre_lane.py
|    |—— features_fast_lane.py
|    |—— features_slow_lane.py
|    |—— main.py
|    |—— Slow_Lane_Final_Model.pkl
|    |—— slow_lane_model.py
|—— README.md
```
## Code Details
### Tested Platform
- software
  ```
  OS: Debian unstable (May 2021), Ubuntu LTS
  Python: 3.8.5 (anaconda)
  PyTorch: 1.7.1, 1.8.1
  ```
- hardware
  ```
  CPU: Intel Xeon 6226R
  GPU: Nvidia RTX3090 (24GB)
  ```
### Hyper parameters
```
```
## References
- [paper-1]()
- [paper-2]()
- [code-1](https://github.com)
- [code-2](https://github.com)
  
## License

## Citing
If you use xxx,please use the following BibTeX entry.
```
```
