# import keras
from keras.models import load_model
# from keras import Sequential

# model = Sequential()
# model.add(keras.layers.LSTM(
#     units=64,
#     input_shape=(40, 1)
# ))
# model.add(keras.layers.Dropout(rate=0.2))
# model.add(keras.layers.RepeatVector(n=40))
# model.add(keras.layers.LSTM(units=64, return_sequences=True))
# model.add(keras.layers.Dropout(rate=0.2))
# model.add(keras.layers.TimeDistributed(keras.layers.Dense(
#     units=1)))
# model.compile(optimizer="adam", loss="mae")

# model.save('gfgModel.h5')
# print('Model Saved!')
 
# load model
savedModel=load_model('gfgModel.h5')
m = savedModel.load_weights()
savedModel.summary()