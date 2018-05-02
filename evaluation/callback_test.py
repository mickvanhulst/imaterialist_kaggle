from keras.layers import Dense, LSTM, Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.datasets import imdb
from evaluation import f1_score

max_features =200
max_len = 20
batch_size = 32

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)


model = Sequential()
model.add(Embedding(max_features, 20))
model.add(LSTM(20, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=2,
          callbacks=[f1_score.micro_averaged_f1((X_test, y_test))],
          validation_data=(X_test, y_test))
