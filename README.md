# Time_Series_Anomaly_Detection

## Usage
This is a model which in more complex shape can detect invalid financial transactions, DDoS Attacks on servers, and etc.
somehow this code is core of some bigger system to prevent waste of money and resources.

## Architecture
### These Three Pipeline are doing main jobs:
* PreProcess: implements all needed processes on data to be prepared for next step which may be train or predict; you should notice pipeline if you want data for training or predicting cause there is a slight difference between preprocess of these two
* Training
* Predicting

## Pipelines in Detail
### PreProcess:
Every timesery becomes chunked and of that chunk will be measured; then we take variance in that specific chunk using measred mean; this process would be repeated once again on variance itself; a feature comes out of it which we call it double variance which is very useful for bolding anomalies; with this procedure your model would not be bound to overall changes through whole timesery
Obviously real world tiemseries would not be the same in length and in opposite, our model needs inputs with same length then we'll use windowing to create desired inputs.
### Training:
Training model is only occured on normal data to become defferentiator of anomalous data; AutoEncoder model is created in this pipeline and input will be passed to it; model architecture would be saved at `model.h5` and also weights would be saved on `weights.h5` in checkpoints with with comparision of loss.
There is one other output beside model weights and thats threshold based on normal data destribution; whenever some predicted data point places higher than threshold then it would be acted as an anomaly.
### Predict
Saved model and weights will be called to reconstruct unseen data; if we pass some anomalous data to model, the reconstructed data would be without anomaly, then diffeneces of data points on input and output (losses) would be compared to threshold; now we can decide if a data point is an anomaly or normal.
