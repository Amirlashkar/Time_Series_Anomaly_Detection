from main import AnomalyDetector
import pickle

with open("timeseries.pkl", "rb") as f:
    timeseries = pickle.load(f)

anomaly_detector = AnomalyDetector.TrainingPipeline()
anomaly_detector(timeseries)