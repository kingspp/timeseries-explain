<!-- Start of Badges -->
![version badge](https://img.shields.io/badge/rainbow--print%20version-0.0.0-green.svg)
![build](https://github.com/kingspp/rainbow-print/workflows/Release/badge.svg)
![coverage badge](https://img.shields.io/badge/coverage-0.00%25|%200.0k/0k%20lines-green.svg)
![test badge](https://img.shields.io/badge/tests-0%20total%7C0%20%E2%9C%93%7C0%20%E2%9C%98-green.svg)
![docs badge](https://img.shields.io/badge/docs-none-green.svg)
![commits badge](https://img.shields.io/badge/commits%20since%20v0.0.0-0-green.svg)
![footprint badge](https://img.shields.io/badge/mem%20footprint%20-0.00%20Mb-green.svg)
[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/kingspprathyush@gmail.com)
<!-- End of Badges -->

# PERT - ***PE***rturbation by Prioritized ***R***eplacemen***T***

Prathyush Parvatharaju, Ramesh Doddaiah, Tom Hartvigsen, Elke Rundensteiner

Paper: #insert link

## Usage
TS-Explain supports command line usage and provides python based API.
### Command Line Usage
```bash
pip install tsexp

# Explain a single instance and output to an image
tsexp -i 131 -f data.csv -m xyz.model -o saliency.png

# Explain a dataset
tsexp -a pert -f data.csv -m xyz.model -o saliency.csv
```

### Python API
```python
from tsexp import PERT

# Explain a single instance
saliency = PERT.explain_instance(...)

# Explain Dataset
saliencies = PERT.explain(...)
```

## API Documentation
[Timeseries Explain](https://kingspp.github.io/timeseries-explain)

## Abstract
Explainable classification is essential to high-impact settings where practitioners require evidence to support their decisions. However, state-of-the-art deep learning models suffer from a lack of trans- parency in how they derive their predictions. One common form of explainability, termed attribution-based explainability, identi- fies which input features are used by the classifier for its predic- tion. While such explainability for image classifiers has recently received focus, little work has been done to-date to address ex- plainability for deep time series classifiers. In this work, we thus propose PERT, a novel perturbation-based explainability method designed explicitly for time series that can explain any classifier. PERT adaptively learns to perform timestep-specific interpolation to perturb instances and explain a black-box model’s predictions for a given instance, learning which timesteps lead to different be- havior in the classifier’s predictions. For this, PERT pairs two novel complementary techniques into an integrated architecture: a Priori- tized Replacement Selector that learns to select the best replacement time series from the background dataset specific to the instance-of- interest with a novel and learnable Guided-Perturbation Function, that uses the replacement time series to carefully perturb an input instance’s timesteps and discover the impact of each timestep on a black-box classifier’s final prediction. Across our experiments recording three metrics on nine publicly-available datasets, we find that PERT consistently outperforms the state-of-the-art explain- ability methods. We also show a case study using the CricketX dataset that demonstrates PERT succeeds in finding the relevant regions of gesture recognition time series.


## Requirements
Python 3.7+

### Development
```bash
# Bare installation
git clone https://github.com/kingspp/pert

# With pre-trained models and datasets
git clone --recurse-submodules -j8 https://github.com/kingspp/pert

# Install requirements
cd pert && pip install -r requirements.txt
```

## Reproduction
```bash
python3 main.py --pname TEST --task_id10 \
--run_mode turing --jobs_per_task 20 \
--algo pert \
--dataset wafer \
--enable_dist False \
--enable_lr_decay False \
--grad_replacement random_instance \
--eval_replacement class_mean \
--background_data_perc 100 \
--run_eval True \
--enable_seed True \
--w_decay 0.00 \
--bbm dnn \
--max_itr 500
```

## Cite
```bash
@article{article,
  author  = {Peter Adams},
  title   = {The title of the work},
  journal = {The name of the journal},
  year    = 1993,
  number  = 2,
  pages   = {201-213},
  month   = 7,
  note    = {An optional note},
  volume  = 4
}
```
