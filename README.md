# SPAD for Multimodal ICL.
Idefics-3 is supported.

## Install

pip install -e .

pip install -r requirements.txt

## Data preparation
```python
cd ge_data/
python gen_data_coco2014.py

### Training
```python
cd train/
python train_ide3.py
```

## Evaluation
```
cd evaluation/
python eval.py
```
