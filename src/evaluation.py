import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os
import time
import psutil
from memory_profiler import memory_usage

from factories import ClassifierFactory, StreamFactory, DetectorFactory

### CAPYMOA

from capymoa.evaluation import ClassificationEvaluator, ClassificationWindowedEvaluator
from capymoa.evaluation.results import PrequentialResults

import argparse



WINDOW_SIZE = 100 # default value

def evaluate_detector(detector, stream, classifier):
    """Evaluate a drift detector with specified stream and classifier types."""

    
    WINDOW_SIZE = stream._length // 100  # 1% of dataset size
    i = 0
    cumulative_evaluator = ClassificationEvaluator(schema=stream.get_schema())
    windowed_evaluator = ClassificationWindowedEvaluator(schema=stream.get_schema(), window_size=WINDOW_SIZE)

    while stream.has_more_instances():
        i += 1
        instance = stream.next_instance()

        y = instance.y_index
        y_pred = classifier.predict(instance)

        cumulative_evaluator.update(y, y_pred)
        windowed_evaluator.update(y, y_pred)

        classifier.train(instance)

        if detector is not None:
            detector.add_element(y)
            if detector.detected_change(): 
                # print(f"Change detected at index: {i}")
                classifier = ClassifierFactory.create(classifier.__class__.__name__, stream.get_schema())

    results = PrequentialResults(
        learner=str(classifier),
        stream=stream,
        cumulative_evaluator=cumulative_evaluator,
        windowed_evaluator=windowed_evaluator
    )
    return results

def benchmark_detector(detector, stream, classifier, save_results=True, filename = "results.csv"):

    stream.restart()

    process = psutil.Process(os.getpid())

    start_time = time.time()
    mem_usage, results = memory_usage((evaluate_detector, (detector, stream, classifier)), retval=True)
    end_time = time.time()

    execution_time = end_time - start_time
    cpu_usage = process.cpu_percent(interval=1)
    memory_usage_max = max(mem_usage)

    results = pd.DataFrame([{
        "dataset": stream.__class__.__name__,
        "classifier": classifier.__class__.__name__,
        "detector": detector.__class__.__name__ if detector else "None",
        "cumulative_accuracy": results.cumulative.metrics_dict()["accuracy"],
        "cumulative_f1": results.cumulative.metrics_dict()["f1_score"],
        "windowed_accuracy": results.windowed.metrics_per_window()["accuracy"].tolist(),
        "windowed_f1": results.windowed.metrics_per_window()["f1_score"].tolist(),
        "execution_time": execution_time,
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage_max,
        "changes": detector.detection_index if detector != None else [],
    }])

    if save_results:
        results.to_csv(filename, mode="a", header=not pd.io.common.file_exists(filename), index=False)
        print(f"Results saved to {filename}")

    return results




def parse_args():

    parser = argparse.ArgumentParser(description="Select stream, classifier, and detector.")
    parser.add_argument("--stream", choices=StreamFactory.stream_classes.keys(), required=True, help="Select the data stream.")
    parser.add_argument("--classifier", choices=ClassifierFactory.classifier_classes.keys(), required=True, help="Select the classifier.")
    parser.add_argument("--detector", choices=DetectorFactory.detector_classes.keys(), required=True, help="Select the drift detector.")

    # Window size argument as percent of dataset size, optional and default to 1%
    parser.add_argument("--window_size", type=float, default=1.0, help="Window size as a percentage of dataset size (default: 1.0).")
    # Save results argument, optional and default to True
    parser.add_argument("--save_results", type=lambda x: x.lower() == 'true', default=True, help="Save results to a CSV file (default: True)")
    # Filename argument, optional and default to "results.csv"
    parser.add_argument("--filename", type=str, default="results.csv", help="Filename to save results (default: results.csv).")

    args = parser.parse_args()

    stream = StreamFactory.create(args.stream)
    classifier = ClassifierFactory.create(args.classifier, stream.get_schema())
    detector = DetectorFactory.create(args.detector)

    window_size = int(stream._length * (args.window_size / 100)) if args.window_size else None
    save_results = args.save_results
    filename = args.filename

    return stream, classifier, detector, window_size, save_results, filename




if __name__ == "__main__":

    stream, classifier, detector, window_size, save_results, filename = parse_args()

    WINDOW_SIZE = window_size

    benchmark_detector(detector, stream, classifier, save_results=save_results, filename=filename)
