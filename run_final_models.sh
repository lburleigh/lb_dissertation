#!/bin/bash

set -e

./02_fit_models_OSG.py \
    --hypfile hypercfg/phase-A_trials-image_roi-whole_bin/ridge_TRUE_final_hypercfg.json \
    --phase A \
    --trial_types image \
    --model_type ridge \
    --single \
    --pickle \
    --job_id 0


./02_fit_models_OSG.py \
    --hypfile hypercfg/phase-A_trials-image_roi-whole_bin/ridge_FALSE_final_hypercfg.json \
    --phase A \
    --trial_types image \
    --model_type ridge \
    --pickle \
    --job_id 0


./02_fit_models_OSG.py \
    --hypfile hypercfg/phase-A_trials-image_roi-whole_bin/coirls_FALSE_final_hypercfg.json \
    --phase A \
    --trial_types image \
    --model_type coirls \
    --pickle \
    --job_id 0
