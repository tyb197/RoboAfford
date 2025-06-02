# RoboAfford: A Dataset and Benchmark for Enhancing Object and Spatial Affordance Learning in Robot Manipulation

## Environment Setup
```
git clone https://github.com/tyb197/RoboAfford.git
cd RoboAfford
conda create -n roboafford-eval python=3.10 -y
pip install -r requirements.txt
```

## Evaluation
### Closed-Source Models
```
bash eval_closed_source_models.sh
```
### Open-Source Models
```
bash eval_open_source_models.sh
```
### Visualization
```
python visualize.py
```
