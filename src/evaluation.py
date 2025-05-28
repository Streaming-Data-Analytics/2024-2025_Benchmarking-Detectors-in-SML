import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os
import time
import psutil
import argparse
import threading
from memory_profiler import memory_usage

from factories import ClassifierFactory, StreamFactory, DetectorFactory

### CAPYMOA

from capymoa.evaluation import ClassificationEvaluator, ClassificationWindowedEvaluator
from capymoa.evaluation.results import PrequentialResults




class Benchmarker:
    def __init__(self, stream=None, classifier=None, detector=None, window_size=1.0, cooldown_window=0, print_results=False, save_results=False, filename="results.csv"):
        if stream is None or classifier is None or detector is None:
            stream, classifier, detector, window_size, cooldown_window, print_results, save_results, filename = self.parse_args()
        
        self.stream = stream
        self.classifier = classifier
        self.detector = detector

        if window_size < 0 or window_size > 100:
            raise ValueError("Window size must be a value between 0 and 1.")
        self.window_size = int(stream._length * (window_size / 100)) if window_size else int(1.0 * (window_size / 100))
        if cooldown_window < 0:
            raise ValueError("Window size must be a non-negative integer.")
        self.cooldown_window = cooldown_window
        self.print_results = print_results if print_results is not None else False
        self.save_results = save_results if save_results is not None else True
        self.filename = filename if filename is not None else "results.csv"

    def evaluate_detector(self):
        i = 0
        cumulative_evaluator = ClassificationEvaluator(schema=self.stream.get_schema())
        windowed_evaluator = ClassificationWindowedEvaluator(schema=self.stream.get_schema(), window_size=self.window_size)

        changes = []
        last_detection_index = -self.cooldown_window

        while self.stream.has_more_instances():
            i += 1
            instance = self.stream.next_instance()
            y = instance.y_index
            y_pred = self.classifier.predict(instance)

            cumulative_evaluator.update(y, y_pred)
            windowed_evaluator.update(y, y_pred)

            self.classifier.train(instance)

            if self.detector is not None:
                self.detector.add_element(y)
                if self.detector.detected_change():
                    if i - last_detection_index >= self.cooldown_window:
                        self.classifier = ClassifierFactory.create(self.classifier.__class__.__name__, self.stream.get_schema())
                        last_detection_index = i
                        if self.detector.__class__.__name__ == "HDDMWeighted":
                            changes.append(i)


        # [NOTE] in capymoa==0.9.0, the add_element() method of HDDM_Weighted behaves differently to the other detectors,
        # in this evaluation function the changes are managed autonomously
        if self.detector.__class__.__name__ == "HDDMWeighted":
            self.detector.detection_index = changes

        results = PrequentialResults(
            learner=str(self.classifier),
            stream=self.stream,
            cumulative_evaluator=cumulative_evaluator,
            windowed_evaluator=windowed_evaluator
        )

        return results

    def benchmark_detector(self):
        self.stream.restart()
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
        mem_usage, results = memory_usage((self.evaluate_detector, ()), retval=True)
        end_time = time.time()

        stop_event.set()
        monitor_thread.join()

        cpu_usage = sum(cpu_samples) / len(cpu_samples) / psutil.cpu_count() if cpu_samples else 0
        execution_time = end_time - start_time
        max_mem_usage = max(mem_usage)

        results_df = pd.DataFrame([{
            "dataset": self.stream.__class__.__name__,
            "classifier": self.classifier.__class__.__name__,
            "detector": self.detector.__class__.__name__ if self.detector else "None",
            "cumulative_accuracy": results.cumulative.metrics_dict()["accuracy"],
            "cumulative_kappa": results.cumulative.metrics_dict()["kappa"],
            "windowed_accuracy": results.windowed.metrics_per_window()["accuracy"].tolist(),
            "windowed_kappa": results.windowed.metrics_per_window()["kappa"].tolist(),
            "execution_time": execution_time,
            "cpu_usage": cpu_usage,
            "memory_usage": max_mem_usage,
            "num_changes": len(self.detector.detection_index if self.detector != None else []),
        }])

        if self.save_results:
            results_df.to_csv(self.filename, mode="a", header=not pd.io.common.file_exists(self.filename), index=False)
            print(f"Results saved to {self.filename}")
        if self.print_results:
            print("Results:")
            print(results_df)

        return results_df

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description="Select stream, classifier, and detector.")
        parser.add_argument("--stream", choices=StreamFactory.stream_classes.keys(), required=True, help="Select the data stream.")
        parser.add_argument("--classifier", choices=ClassifierFactory.classifier_classes.keys(), required=True, help="Select the classifier.")
        parser.add_argument("--detector", choices=DetectorFactory.detector_classes.keys(), required=True, help="Select the drift detector.")

        parser.add_argument("--window_size", type=float, default=1.0, help="Window size as a percentage of dataset size (default: 1.0).")
        # cooldown window lengt
        parser.add_argument("--cooldown_window", type=int, default=0, help="Cooldown window length (default: 0).")
        
        parser.add_argument("--print_results", type=lambda x: x.lower() == 'true', default=False, help="Print results to console (default: True)")
        parser.add_argument("--save_results", type=lambda x: x.lower() == 'true', default=True, help="Save results to a CSV file (default: True)")
        parser.add_argument("--filename", type=str, default="results.csv", help="Filename to save results (default: results.csv).")

        args = parser.parse_args()

        stream = StreamFactory.create(args.stream)
        classifier = ClassifierFactory.create(args.classifier, stream.get_schema())
        detector = DetectorFactory.create(args.detector)

        window_size = args.window_size
        cooldown_window = args.cooldown_window
        print_results = args.print_results
        save_results = args.save_results
        filename = args.filename

        return stream, classifier, detector, window_size, cooldown_window, print_results, save_results, filename


if __name__ == "__main__":
    benchmark = Benchmarker()
    benchmark.benchmark_detector()
