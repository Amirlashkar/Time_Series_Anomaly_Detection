import os
import yaml
from yaml.loader import SafeLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras import Sequential
import keras


input_path = os.path.join(os.getcwd(), "time-series")
output_path = "./output"
if not os.path.exists(output_path):
    os.mkdir(output_path)

class TimeSeriesAnomalyDetector:

    def __init__(self, timeseries):
        with open('config.yml', 'r') as f:
            config = list(yaml.load_all(f, Loader=SafeLoader))[0]["variables"]
        self.df = pd.DataFrame(timeseries)
        self.model_data = self.create_model_data()
        self.mean_chunk_size = config["mean_chunk_size"]
        self.dataset_timestamps = config["dataset_timestamps"]

    def chunk_and_replicate(self, seq, size):
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

    def create_model_data(self):
        model_data = self.df.copy()
        model_data["mean"] = self.chunk_and_replicate(model_data["value"], self.mean_chunk_size)
        model_data["variance"] = (model_data["value"] - model_data["mean"]) ** 2
        model_data["variance_mean"] = self.chunk_and_replicate(model_data["variance"], self.mean_chunk_size)
        model_data["double_variance"] = (model_data["variance"] - model_data["variance_mean"]) ** 2
        model_data.drop(["mean", "value", "variance", "variance_mean"], axis=1, inplace=True)
        model_data = model_data.rename(columns={"double_variance": "value"})
        return model_data

    def create_dataset(self, X, Y, timesteps=1):
        Xs, Ys = [], []
        for i in range(len(X) - timesteps):
            v = X.iloc[i:(i + timesteps)].value
            Xs.append(v)
            Ys.append(Y.iloc[i + timesteps])
        return np.array(Xs), np.array(Ys)

    def dataset(self):
        model_data = self.model_data
        train_size = int(len(self.df) * .3)
        train, test = model_data.iloc[:], model_data.iloc[:]
        scaler = StandardScaler()
        pca = PCA()
        train["value"] = pca.fit_transform(train[["value"]])
        test["value"] = pca.fit_transform(test[["value"]])
        train["value"] = scaler.fit_transform(train[["value"]])
        test["value"] = scaler.fit_transform(test[["value"]])
        X_train, Y_train = self.create_dataset(train[["value"]], train.value, self.dataset_timestamps)
        X_test, Y_test = self.create_dataset(test[["value"]], test.value, self.dataset_timestamps)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        return X_train, Y_train, X_test, Y_test, train, test

    def model(self):
        X_train, Y_train, X_test, Y_test, train, test = self.dataset()
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
        history = model.fit(
            X_train, Y_train,
            epochs=1,
            batch_size=1024,
            validation_split=0.2,
            shuffle=False
        )
        X_train_pred = model.predict(X_train)
        train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
        A = np.mean(train_mae_loss)
        STD = np.std(train_mae_loss)
        threshold1 = A + (2 * STD)
        threshold2 = A - (2 * STD)
        X_test_pred = model.predict(X_test)
        test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
        test_score_df = pd.DataFrame(index=test[self.dataset_timestamps:].index)
        test_score_df['loss'] = test_mae_loss
        test_score_df['threshold2'] = threshold2
        test_score_df['threshold1'] = threshold1
        test_score_df['above_anomaly'] = test_score_df.loss > test_score_df.threshold1
        test_score_df['below_anomaly'] = test_score_df.loss < test_score_df.threshold2
        test_score_df['value'] = test[self.dataset_timestamps:].value
        return test_score_df

    def anomaly_detection(self):
        test_score_df = self.model()
        model_data = self.model_data
        above_anomalies = test_score_df[test_score_df.above_anomaly == True]
        below_anomalies = test_score_df[test_score_df.below_anomaly == True]
        above_index = above_anomalies.index.tolist()
        below_index = below_anomalies.index.tolist()
        anomalies_indexes = above_index + below_index

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

    def create_output(self):
        df = self.df
        indexes = self.anomaly_detection()
        out = df.copy()
        out["label"] = 0
        out.loc[indexes, "label"] = 1
        out = out.rename(columns={"timestamp": "time"})
        out.set_index("time", inplace=True)
        return out


if __name__ == '__main__':
    for filename in os.listdir(input_path):
        df = pd.read_csv(os.path.join(input_path, filename))
        print(filename, len(df))
        anomaly_detector = TimeSeriesAnomalyDetector(df)
        result = anomaly_detector.create_output()
        result.to_csv(os.path.join(output_path, filename))
        print(f'item {filename} processed.')