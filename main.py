import os
from typing import Any
import yaml
from yaml.loader import SafeLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from keras.models import load_model
from keras import Sequential
import keras

class AnomalyDetector:

    def __init__(self):
        with open('config.yml', 'r') as f:
                config = yaml.safe_load(f)["variables"]
        self.config = config

    class PreProcessPipeline:

        def __init__(self) -> None:
            config = AnomalyDetector().config
            self.mean_chunk_size = config["mean_chunk_size"]
            self.dataset_timestamps = config["dataset_timestamps"]

        def __call__(self, timeseries_df, stage) -> tuple:
            df = self.double_variance(timeseries_df, stage)
            data_tuple = self.dataset(df)
            return data_tuple
        
        def mean_chunk(self, seq, size) -> list:
            num = len(seq) // size
            avg = len(seq) / float(num)
            out = []
            last = 0.0
            while last < len(seq):
                out.append(seq[int(last):int(last + avg)])
                last += avg

            replicated = []
            for i in out:
                X = i.mean()
                for j in range(len(i)):
                    replicated.append(X)

            return replicated

        def windowing(self, X, Y, timesteps=1) -> np.ndarray:
            Xs, Ys = [], []
            for i in range(len(X) - timesteps):
                v = X.iloc[i:(i + timesteps)].value
                Xs.append(v)
                Ys.append(Y.iloc[i + timesteps])
            return np.array(Xs), np.array(Ys)

        def double_variance(self, df, stage) -> pd.DataFrame:
            """
            first component of the training pipeline ; it gets raw dataframe and drive another dataframe from it 
            with desired feature which is called double variance ; double variance is good enough for magnifying
            samples away from partial mean or that chunk mean.
            """
            df["mean"] = self.mean_chunk(df["value"], self.mean_chunk_size)
            df["variance"] = (df["value"] - df["mean"]) ** 2
            df["variance_mean"] = self.mean_chunk(df["variance"], self.mean_chunk_size)
            df["double_variance"] = (df["variance"] - df["variance_mean"]) ** 2
            df.drop(["mean", "value", "variance", "variance_mean"], axis=1, inplace=True)
            if stage == "train":
                df.loc[df["label"] == 1, "value"] = 0
            df = df.rename(columns={"double_variance": "value"})
            return df

        def dataset(self, df) -> tuple:
            data = df.iloc[:]
            # scaler = StandardScaler()
            # pca = PCA()
            # train["value"] = pca.fit_transform(train[["value"]])
            # test["value"] = pca.fit_transform(test[["value"]])
            # train["value"] = scaler.fit_transform(train[["value"]])
            # test["value"] = scaler.fit_transform(test[["value"]])
            X_data, Y_data = self.windowing(data[["value"]], data.value, self.dataset_timestamps)
            X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)
            return (X_data, Y_data)
            

    class TrainingPipeline:

        def __init__(self) -> None:
            config = AnomalyDetector().config
            self.epochs = config["train_epochs"]

        def __str__(self) -> str:
            return "Training Pipeline"

        def __call__(self, timeseries) -> None:
            preprocess = AnomalyDetector.PreProcessPipeline()
            data_tuple = preprocess(timeseries, "train")
            model = self.create_model(data_tuple)
            self.model_train(data_tuple, model)
        
        def create_model(self, data_tuple) -> keras.Model:
            X_train, Y_train = data_tuple
            model = Sequential()
            model.add(keras.layers.LSTM(
                units=64,
                input_shape=(X_train.shape[1], X_train.shape[2])
            ))
            model.add(keras.layers.Dropout(rate=0.2))
            model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
            model.add(keras.layers.LSTM(units=64, return_sequences=True))
            model.add(keras.layers.Dropout(rate=0.2))
            model.add(keras.layers.TimeDistributed(keras.layers.Dense(
                units=X_train.shape[2])))
            model.compile(optimizer="adam", loss="mae")
            return model

        def model_train(self, data_tuple, model) -> None:
            X_train, Y_train = data_tuple
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath="weights.h5",
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True
            )
            history = model.fit(
                X_train, Y_train,
                epochs=self.epochs,
                batch_size=1024,
                validation_split=0.2,
                shuffle=False,
                callbacks=[model_checkpoint_callback]
            )
            model.save("model.h5")
            X_train_pred = model.predict(X_train)
            train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
            A = np.mean(train_mae_loss)
            STD = np.std(train_mae_loss)
            threshold = A + STD

            self.config["threshold"] = threshold
            with open("config.yml", "w") as f:
                yaml.safe_dump(self.config, f, default_flow_style=False)

    class PredictPipeline:

        def __init__(self):
            config = AnomalyDetector().config
            self.threshold = config["threshold"]
            if self.threshold == None:
                raise ValueError("Threshold value is not valid and is None by now ; train a model with \
                                 'AnomalyDetector.TrainingPipeline' to change it")
            
            if not os.path.exists("model.h5"):
                raise FileNotFoundError("Download model.h5 from 'https://github.com/Amirlashkar/Time_Series_Anomaly_Detection' \
                                        or if you have data to train, use Training_Pipeline.run_pipeline to create model.h5")
            else:
                self.model = load_model("model.h5")
                self.model.load_weights("weights.h5")
        
        def __str__(self) -> str:
            return "Predict Pipeline"
        
        def __call__(self, timeseries) -> pd.DataFrame:
            data_tuple = AnomalyDetector.PreProcessPipeline(timeseries, stage="test")
            test_score_df = self.test_predict(data_tuple, timeseries)
            indexes = self.anomaly_detection(test_score_df)
            output = self.create_output(timeseries, indexes)
            return output
        
        def test_predict(self, data_tuple, timeseries):
            X_test, _ = data_tuple
            X_test_pred = self.model.predict(X_test)
            test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
            test_score_df = pd.DataFrame(index=timeseries[self.dataset_timestamps:].index)
            test_score_df['loss'] = test_mae_loss
            test_score_df['threshold'] = self.threshold
            test_score_df['above_anomaly'] = test_score_df.loss > test_score_df.threshold
            test_score_df['value'] = timeseries[self.dataset_timestamps:].value
            return test_score_df
        
        def anomaly_detection(self, test_score_df) -> Any:
            model_data = self.model_data
            above_anomalies = test_score_df[test_score_df.above_anomaly == True]
            above_index = above_anomalies.index.tolist()
            anomalies_indexes = above_index

            a = []
            for index in anomalies_indexes:
                b = []
                for margin in range(-50, 50):
                    b.append(index + margin)
                a.append(b)

            stds = []
            for s in a:
                std = float("-inf")
                for d in s:
                    t = model_data[model_data.index == d]["value"]
                    if len(t) == 1:
                        t = t.item()
                        if t > std:
                            std = t
                stds.append(std)

            stds = list(np.unique(stds))
            detected_anomalies = model_data.query(f"value == {stds}")
            indexes = detected_anomalies.index
            return indexes

        def create_output(self, timeseries, anomaly_indexes) ->pd.DataFrame:
            out = timeseries.copy()
            out["create_label"] = 0
            out.loc[anomaly_indexes, "created_label"] = 1
            out = out.rename(columns={"timestamp": "time"})
            out.set_index("time", inplace=True)
            return out