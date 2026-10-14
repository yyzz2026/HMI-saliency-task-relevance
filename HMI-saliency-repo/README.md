# Task-Relevance Gap of Bottom-Up Saliency Models for Automotive HMI Screening

This repository contains the annotation files, scenario definitions, and evaluation scripts for the paper:

> **Quantifying the Task-Relevance Gap of Bottom-Up Saliency Models for Automotive Human-Machine Interface Screening**

## Contents
- `data/annotations.csv`: Bounding boxes and cognitive priority weights for all functional regions.
- `data/scenarios.csv`: 30 driving-task scenarios, each with a target region.
- `data/screenshot_sources.csv`: Sources of the 30 HMI screenshots (brand, model, interface type).
- `scripts/`: Python scripts for saliency detection, metric computation, statistical tests, and figure generation.
- `model_configs.md`: Parameter settings for SR, Itti-Koch, and GBVS.
- `random_seeds.txt`: Random seeds used for reproducibility.

## Data Availability
The original HMI screenshots are **not redistributed** due to copyright restrictions.
`screenshot_sources.csv` describes each screenshot's source so that interested readers can obtain them independently.
All annotations, scenario definitions, and evaluation scripts are provided for full reproducibility of the reported metrics.

## Requirements
Python 3.9, with packages listed in `requirements.txt`.

## Usage
1. Collect the 30 screenshots according to `screenshot_sources.csv` and place them in an `images/` folder.
2. Run `scripts/step4_batch_saliency.py` to generate SR saliency maps.
3. Run `scripts/step7_batch_itti.py` and `scripts/step8_batch_gbvs.py` for Itti-Koch and GBVS.
4. Run `scripts/step9_evaluate_models.py` to compute AUC, NSS, CC, and hit rates.
5. Further scripts reproduce the statistical tests and figures.

## Citation
If you use this repository, please cite the paper.

## Contact
For questions, contact [499634044@qq.com].
