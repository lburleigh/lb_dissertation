#!/bin/bash

set -e

./02_fit_models_OSG.py \
    --hypfile ridge_TRUE_final_hypcfg.json \
    --phase B \
    --trial_types image view \
    --model_type ridge \
    --single \
    --pickle \
    --job_id 0


./02_fit_models_OSG.py \
    --hypfile ridge_FALSE_final_hypcfg.json \
    --phase B \
    --trial_types image view \
    --model_type ridge \
    --pickle \
    --job_id 0


./02_fit_models_OSG.py \
    --hypfile coirls_FALSE_final_hypcfg.json \
    --phase B \
    --trial_types image view \
    --model_type coirls \
    --pickle \
    --job_id 0