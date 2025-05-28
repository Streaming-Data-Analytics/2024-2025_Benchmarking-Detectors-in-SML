#!/bin/bash

cd "$(dirname "$0")"

streams=("Electricity" "Hyper100k" "Covtype" "Sensor")
classifiers=("HoeffdingTree")
detectors=("CUSUM" "PageHinkley" "DDM" "HDDMAverage" "HDDMWeighted" "ADWIN" "STEPD" "None")

for stream in "${streams[@]}"; do
    for classifier in "${classifiers[@]}"; do
        for detector in "${detectors[@]}"; do
            echo "Running with Stream: $stream, Classifier: $classifier, Detector: $detector"
            python evaluation.py --stream "$stream" --classifier "$classifier" --detector "$detector"
            echo "Done."
            echo "--------------------------------------"
        done
    done
done