import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, LeakyReLU, BatchNormalization, ReLU
from sklearn.preprocessing import StandardScaler

#------- Load training data, scale it --------
print("Loading training data...")
train_df = pd.read_csv('data/train_clean.csv',index_col=False)
train_df.dropna(inplace=True)
np.random.seed(7)

print("Scaling data for model...")
scaler = StandardScaler()
X = scaler.fit_transform(train_df.drop(labels=['is_q','Sentence','Unnamed: 0'],axis=1))
Y = train_df['is_q']
# X.shape should be (49998,41), Y.shape should be (49998,)
print("Training model....")
#------- model architecture -----------

# try adding an LSTM, may have to reformat data for that though

model = Sequential()

model.add(Dense(2048,input_shape=(41,)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(rate=0.1))

model.add(Dense(1024))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(rate=0.5))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(rate=0.5))

model.add(Dense(256)) # hidden layer
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(rate=0.5)) # dropout regularization

model.add(Dense(1,activation='sigmoid'))

adam = Adam(lr=0.0001) # 1 order of magnitude less than Adam default learning rate
model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy']) 

#------- fit to training data ---------

model.fit(X,Y,epochs=3,batch_size=64) 
model.save('tag_count_nn.h5')
print("Model trained!")

#------- load test data, make predictions---
print("Loading test data...")
test_df = pd.read_csv('data/test_clean.csv',index_col=False)
X_pred = scaler.fit_transform(test_df.drop(labels=['Sentence','Unnamed: 0'],axis=1))
print("Making predictions....")
predictions = model.predict(X_pred,verbose=1)

# floor/ceiling sigmoids, predictions are all single value arrays
# for some reason
bin_pred = []
for p in predictions:
    if p[0] > 0.5:
        bin_pred.append(1)
    else:
        bin_pred.append(0)

test_df['Predictions'] = bin_pred

test_df[['Sentence','Predictions']].to_csv('model_output.csv')

with open('model_predictions.txt','w') as f:
	for p in bin_pred:
		f.write(str(p))
		f.write('\n')


print("Predictions done, check directory for model output.")
