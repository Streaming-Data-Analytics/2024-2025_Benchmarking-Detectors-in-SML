import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os
import time
import psutil
import threading
from memory_profiler import memory_usage

from factories import ClassifierFactory, StreamFactory, DetectorFactory

### CAPYMOA

from capymoa.evaluation import ClassificationEvaluator, ClassificationWindowedEvaluator
from capymoa.evaluation.results import PrequentialResults

import argparse



WINDOW_SIZE = 100 # default value

def evaluate_detector(detector, stream, classifier):
    i = 0
    cumulative_evaluator = ClassificationEvaluator(schema=stream.get_schema())
    windowed_evaluator = ClassificationWindowedEvaluator(schema=stream.get_schema(), window_size=WINDOW_SIZE)

    # [NOTE] in capymoa==0.9.0, the add_element() method of HDDM_Weighted behaves differently to the other detectors,
    # in this evaluation function the changes are managed autonomously
    changes = []

    while stream.has_more_instances():
        i += 1

        instance = stream.next_instance()

        y = instance.y_index
        y_pred = classifier.predict(instance)

        cumulative_evaluator.update(y, y_pred)
        windowed_evaluator.update(y, y_pred)

        classifier.train(instance)



        if detector != None:
            detector.add_element(y)
            if detector.detected_change():
              # print("Change detected at index: " + str(i))
              classifier = ClassifierFactory.create(classifier.__class__.__name__, stream.get_schema())
              if detector.__class__.__name__ == "HDDMWeighted":
                  changes.append(i)


    if detector.__class__.__name__ == "HDDMWeighted":
        detector.detection_index = changes



    results = PrequentialResults(learner=str(classifier),
                                 stream=stream,
                                 cumulative_evaluator=cumulative_evaluator,
                                 windowed_evaluator=windowed_evaluator)
    return results


def benchmark_detector(detector, stream, classifier, print_results=False, save_results=True, filename = "results.csv"):

    stream.restart()

    cpu_samples = []

    def monitor_cpu(process, interval=0.1):
        while not stop_event.is_set():
            cpu_samples.append(process.cpu_percent(interval=None))
            time.sleep(interval)

    process = psutil.Process(os.getpid())
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_cpu, args=(process,))
    monitor_thread.start()

    start_time = time.time()
    mem_usage, results = memory_usage((evaluate_detector, (detector, stream, classifier)), retval=True)
    end_time = time.time()

    stop_event.set()
    monitor_thread.join()

    cpu_usage = sum(cpu_samples) / len(cpu_samples) / psutil.cpu_count()    
    execution_time = end_time - start_time
    max_mem_usage = max(mem_usage)

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
        "memory_usage": max_mem_usage,
        "num_changes": len(detector.detection_index if detector != None else []),
    }])

    if save_results:
        results.to_csv(filename, mode="a", header=not pd.io.common.file_exists(filename), index=False)
        print(f"Results saved to {filename}")
    if print_results:
        print("Results:")
        print(results)

    return results




def parse_args():

    parser = argparse.ArgumentParser(description="Select stream, classifier, and detector.")
    parser.add_argument("--stream", choices=StreamFactory.stream_classes.keys(), required=True, help="Select the data stream.")
    parser.add_argument("--classifier", choices=ClassifierFactory.classifier_classes.keys(), required=True, help="Select the classifier.")
    parser.add_argument("--detector", choices=DetectorFactory.detector_classes.keys(), required=True, help="Select the drift detector.")

    parser.add_argument("--window_size", type=float, default=1.0, help="Window size as a percentage of dataset size (default: 1.0).")
    parser.add_argument("--print_results", type=lambda x: x.lower() == 'true', default=False, help="Print results to console (default: True)")
    parser.add_argument("--save_results", type=lambda x: x.lower() == 'true', default=True, help="Save results to a CSV file (default: True)")
    parser.add_argument("--filename", type=str, default="results.csv", help="Filename to save results (default: results.csv).")


    args = parser.parse_args()

    stream = StreamFactory.create(args.stream)
    classifier = ClassifierFactory.create(args.classifier, stream.get_schema())
    detector = DetectorFactory.create(args.detector)

    WINDOW_SIZE = int(stream._length * (args.window_size / 100)) if args.window_size else int(1.0 * (args.window_size / 100))
    print_results = args.print_results
    save_results = args.save_results
    filename = args.filename


    return stream, classifier, detector, print_results, save_results, filename




if __name__ == "__main__":

    stream, classifier, detector, print_results, save_results, filename = parse_args()

    benchmark_detector(detector, stream, classifier, print_results=print_results, save_results=save_results, filename=filename)
