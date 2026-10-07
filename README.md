# CoRT: Conformal Reliability-guided Triage

Reference implementation for **Reliability-Guided Conformal Triage for
Clinical Risk Prediction with Complementary Models**.

CoRT independently conformalizes probability outputs from a tabular predictor
and a longitudinal predictor. Their joint prediction sets define six triage
states. States A-D yield directional recommendations, while directional
conflict (state E) and joint uncertainty (state F) are routed for review.

## Triage States

| State | Tabular set | Longitudinal set | Routing |
|---|---|---|---|
| A | `{0}` | `{0}` | Consensus non-event direction |
| B | `{1}` | `{1}` | Consensus event direction |
| C | One predictor: `{0}` | Other predictor: `{0,1}` | Follow the non-event singleton |
| D | One predictor: `{1}` | Other predictor: `{0,1}` | Follow the event singleton |
| E | One predictor: `{0}` | Other predictor: `{1}` | Review: directional conflict |
| F | `{0,1}` | `{0,1}` | Review: joint uncertainty |

Empty prediction sets are routed for review and should trigger model
monitoring.

## Input Contract

The conformal modules operate on aligned calibration and test arrays:

- `label`: binary observed outcome for calibration or evaluation.
- `prob_tabular`: estimated probability of class 1 from the tabular model.
- `prob_longitudinal`: estimated probability of class 1 from the longitudinal
  model.

The same calibration observations and the same test observations must be used
for both predictors. Patient and encounter identifiers are not required by the
core implementation.

## Repository Layout

```text
src/cort/   conformal calibration, state assignment, routing, metrics, bootstrap
models/     reference tabular and longitudinal predictor components
configs/    generic configuration templates without site-specific fitted values
analysis/   reusable analysis functions
tests/      synthetic unit tests
```

## Environment

```bash
conda env create -f environment.yml
conda activate cort
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH}"
pytest -q
```

## Minimal Example

```python
import numpy as np

from cort.conformal import MondrianCalibrator
from cort.routing import route_prediction_sets

y_cal = np.array([0, 0, 1, 1])
p_cal = np.array([0.10, 0.20, 0.70, 0.90])
p_test = np.array([0.15, 0.55, 0.85])

calibrator = MondrianCalibrator(alpha=0.10).fit(y_cal, p_cal)
prediction_sets = calibrator.predict_sets(p_test)

decision = route_prediction_sets({0}, {0, 1})
print(decision.state, decision.direction, decision.review_flag)
```

## Data and Model Availability

Patient-level EHR data, encounter-level predictions, attribution records, and
fitted model checkpoints are not distributed because the multisite data are
governed by institutional data-use agreements. This repository provides the
method logic, model interfaces, configuration templates, analysis functions,
and synthetic tests needed to apply CoRT to locally authorized data.
