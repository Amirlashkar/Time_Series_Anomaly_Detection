import os
from typing import Any
import yaml
from yaml.loader import SafeLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow import keras

# Main Object
class AnomalyDetector:

    def __init__(self):
        # reading yml file for constant values calling (whenever you wanted to change some variable 
        # its just easier to change it on config.yml)
        with open('config.yml', 'r') as f:
                config = yaml.safe_load(f)["variables"]
        # making cofig main feature of object cause its needed to be called on other subclasses also
        self.config = config

    class PreProcessPipeline:
        """
        description:
        PreProcess Pipeline is needed either for Training and Predicting ; but there is just slight change and it can be deployed
        by changing stage argument on __call__ function
        """

        def __init__(self) -> None:
            # calling config dictionary and variables from it
            config = AnomalyDetector().config
            self.mean_chunk_size = config["mean_chunk_size"]
            self.window_size = config["window_size"]

        def __call__(self, timeseries_df, stage) -> tuple:
            """
            description:
            whole implementation of pipeline and components are created on __call__ function of each Pipeline to make class act as a function (
            we give it an input and expect an specific output)

            args:
            timeseries_df -> a dataframe with columns like timestamp, value, and label
            stage -> which stage you want this pipeline for? train or pred
            """
            df = self.double_variance(timeseries_df, stage)
            data_tuple = self.dataset(df)
            return data_tuple
        
        def mean_chunk(self, seq, size) -> list:
            """
            description:
            gets a series of data like python list or pandas Series as input and chunks it to same sizes ;
            measures mean of every chunk and place it as main value with same order of main data value

            args:
            seq -> sequence of data
            size -> size of chunks
            """
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

        def windowing(self, X, Y, window_size=1) -> np.ndarray:
            """
            description:
            implements sliding window on data, every window differs just one value from previous window thus every (n, ) shape data would
            be converted to (n, window_size, 1) shape data

            args:
            X -> data with model input shape
            Y -> data with model output shape
            with respect to AutoEncoder Type model, both X and Y have same shape to each other
            window_size -> obvious
            """
            Xs, Ys = [], []
            for i in range(len(X) - window_size):
                v = X.iloc[i:(i + window_size)].value
                Xs.append(v)
                Ys.append(Y.iloc[i + window_size])
            return np.array(Xs), np.array(Ys)

        def double_variance(self, df, stage) -> pd.DataFrame:
            """
            first component of the training pipeline ; it gets raw dataframe and drive another dataframe from it 
            with desired feature which is called double variance ; double variance is good enough for magnifying
            samples away from partial mean or that chunk mean.
            """
            # measuring partial mean
            df["mean"] = self.mean_chunk(df["value"], self.mean_chunk_size)
            # measuring partial variance
            df["variance"] = (df["value"] - df["mean"]) ** 2
            # repeating last process on variance
            df["variance_mean"] = self.mean_chunk(df["variance"], self.mean_chunk_size)
            df["double_variance"] = (df["variance"] - df["variance_mean"]) ** 2
            # removing unwanted columns
            df.drop(["mean", "value", "variance", "variance_mean"], axis=1, inplace=True)
            # renaming double_variance to value (main data for model is double_variance)
            df = df.rename(columns={"double_variance": "value"})
            # we'll normalize data for training if stage is declared as train ; cause due to decided architecture for this problem solution
            # every anomalous data data should be brought to normal distribution ; so we change double_variance to 0
            if stage == "train":
                df.loc[df["label"] == 1, "value"] = 0
            return df

        def dataset(self, df) -> tuple:
            data = df.iloc[:]
            # scaler = StandardScaler()
            # pca = PCA()
            # train["value"] = pca.fit_transform(train[["value"]])
            # test["value"] = pca.fit_transform(test[["value"]])
            # train["value"] = scaler.fit_transform(train[["value"]])
            # test["value"] = scaler.fit_transform(test[["value"]])
            X_data, Y_data = self.windowing(data[["value"]], data.value, self.window_size)
            X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)
            return (X_data, Y_data)
            

    class TrainingPipeline:

        def __init__(self) -> None:
            config = AnomalyDetector().config
            self.epochs = config["train_epochs"]

        def __str__(self) -> str:
            return "Training Pipeline"

        def __call__(self, timeseries, gpu=False) -> None:
            preprocess = AnomalyDetector.PreProcessPipeline()
            data_tuple = preprocess(timeseries, "train")
            model = self.create_model(data_tuple, gpu)
            self.model_train(data_tuple, model)
        
        def create_model(self, data_tuple, gpu) -> keras.Model:
            if gpu:
                tf.config.set_visible_devices([], 'GPU')
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
            model.save("model.h5")
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
            self.window_size = config["window_size"]
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
            test_score_df = pd.DataFrame(index=timeseries[self.window_size:].index)
            test_score_df['loss'] = test_mae_loss
            test_score_df['threshold'] = self.threshold
            test_score_df['above_anomaly'] = test_score_df.loss > test_score_df.threshold
            test_score_df['value'] = timeseries[self.window_size:].value
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