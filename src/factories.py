### CAPYMOA

from capymoa.classifier import NaiveBayes, HoeffdingTree

from capymoa.datasets import Electricity, Covtype,  Hyper100k, Sensor
from capymoa.datasets import ElectricityTiny # For testing

from capymoa.drift.detectors import ADWIN, STEPD, CUSUM, PageHinkley, DDM, HDDMAverage, HDDMWeighted


class ClassifierFactory:

    classifier_classes = {
        "NaiveBayes": NaiveBayes,
        "HoeffdingTree": HoeffdingTree
    }

    @staticmethod
    def create(classifier_type, schema):
        """Create a new classifier instance based on type."""
        
        
        if classifier_type not in ClassifierFactory.classifier_classes:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
            
        return ClassifierFactory.classifier_classes[classifier_type](schema=schema)


class StreamFactory:

    stream_classes = {
        "Covtype": Covtype,
        "Electricity": Electricity,
        "Hyper100k": Hyper100k,
        "Sensor": Sensor,
        "ElectricityTiny": ElectricityTiny
    }

    @staticmethod
    def create(stream_type):
        """Create a new stream instance based on type."""
        
        
        if stream_type not in StreamFactory.stream_classes:
            raise ValueError(f"Unknown stream type: {stream_type}")
            
        # Return a new instance of the requested stream
        return StreamFactory.stream_classes[stream_type]()


class DetectorFactory:


    detector_classes = {
        "ADWIN": ADWIN,
        "STEPD": STEPD,
        "CUSUM": CUSUM,
        "PageHinkley": PageHinkley,
        "DDM": DDM,
        "HDDMAverage": HDDMAverage,
        "HDDMWeighted": HDDMWeighted
    }

    @staticmethod
    def create(detector_type):
        """Create a new detector instance based on type."""
        
        
        if detector_type == "None":
            return None
            
        if detector_type not in DetectorFactory.detector_classes:
            raise ValueError(f"Unknown detector type: {detector_type}")
            
        # Return a new instance of the requested detector
        return DetectorFactory.detector_classes[detector_type]()
