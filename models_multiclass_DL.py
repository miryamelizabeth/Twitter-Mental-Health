### enviroment: deep_learning

import pandas as pd
import numpy as np

from os import path

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence, text

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Convolution1D, GRU
from tensorflow.keras.layers import Dropout, SpatialDropout1D, GlobalMaxPool1D, MaxPooling1D

from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import Adam, SGD, RMSprop
# from tensorflow.keras.regularizers import l1, l2

from tensorflow.keras import backend as K 

from pycm import ConfusionMatrix
# https://www.pycm.ir/doc/index.html#Full

import warnings
warnings.filterwarnings('ignore')



# ---------------------------
SEED = 42
FOLDS = 5

VOCAB_SIZE = 10000
MAX_LEN = 5000

EMBEDDING_DIMS = 300

FILTERS = 100 # size of convolved matrix
KERNEL_SIZE = 3 # filter convolved matrix size
UNITS = 100 # size of lstm o gru
HIDDEN_DIMS = 64 # in full-connected network

BATCH_SIZE = 4
EPOCHS = 20

ONLY_DISORDERS = True


def load_dataset(trainFile, testFile):
	"""
	Read datasets in csv format using pandas.
	
	Parameters
	------------
	trainFile: string
		Name of training file including path
	testFile: string
		Name of training file including path

	Return
	------------
	X_train: numpy array or dataframe
		Data selected as predictors for our ML model (training phase)
	X_test: numpy array or dataframe
		Data selected as predictors for our ML model (testing phase)
	y_train,: numpy array
		Column with the data used as the class (training phase)
	y_test: numpy array
		Column with the data used as the class (testing phase)
	encoder: sklearn.preprocessing.LabelEncoder
		Transformer used for encoding target values
	"""

	print(f'Reading files...')
	train = pd.read_csv(trainFile)
	test = pd.read_csv(testFile)


	if ONLY_DISORDERS:
		train = train[train['class'] != 'CONTROL']
		test = test[test['class'] != 'CONTROL']

	train = train.sample(frac=1).reset_index(drop=True)

	X_train = train['tweets_user'].values
	X_test = test['tweets_user'].values


	y_train = train['class'].values
	y_test = test['class'].values

	encoder = LabelEncoder()
	encoder.fit(y_train)

	y_train = encoder.transform(y_train)
	y_test = encoder.transform(y_test)

	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)


	return X_train, X_test, y_train, y_test, encoder



def preprocess_text_format(train, test):
	"""
	Convert text to sequence of tokens and pad them to ensure equal length vectors.
	
	Parameters
	------------
	train: numpy array or dataframe
		Data selected as predictors for our ML model (training phase)
	test: numpy array or dataframe
		Data selected as predictors for our ML model (testing phase)
	
	Return
	------------
	X_train: 
		Sequence of tokens
	X_test:
		Sequence of tokens
	tokenizer: tensorflow.keras.preprocessing.text.Tokenizer
		Tokenizer fit on text
	"""

	print(f'Pre-process text...')

	tokenizer = text.Tokenizer(num_words=VOCAB_SIZE)
	tokenizer.fit_on_texts(train)

	X_train = tokenizer.texts_to_sequences(train)
	X_test = tokenizer.texts_to_sequences(test)

	# convert text to sequence of tokens and pad them to ensure equal length vectors 
	X_train = sequence.pad_sequences(X_train, padding='post', maxlen=MAX_LEN)
	X_test = sequence.pad_sequences(X_test, padding='post', maxlen=MAX_LEN)


	return X_train, X_test, tokenizer



def load_embeddings_file(embeddingsFile):
	"""
	Read embeddings file
	
	Parameters
	------------
	embeddingsFile: string
		Name of the embbedding file
	
	Return
	------------
	embeddings_index: dict
		Dictionary with the embeddings
	"""

	print(f'Load embeddings...')

	embeddings_index = dict()
	f = open(embeddingsFile, encoding='UTF-8')

	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()


	print(f'Found {len(embeddings_index)} word vectors.')

	return embeddings_index



def create_embedding_matrix(tokenizer, embeddings_index):
	"""
	Create the embedding matrix for the neural network
	
	Parameters
	------------
	tokenizer: tensorflow.keras.preprocessing.text.Tokenizer
		Tokenizer fit on text
	
	embeddings_index: dict
		Dictionary with the embeddings
	
	Return
	------------
	embedding_matrix: numpy array
		Embedding matrix
	"""

	print(f'Create layer...')

	hits = 0
	misses = 0

	embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIMS))
	for word, index in tokenizer.word_index.items():
		if index > VOCAB_SIZE - 1:
			break
		else:
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[index] = embedding_vector
				hits += 1
			else:
				misses += 1

	print(f'Converted {hits} words ({misses} misses).\nShape: {embedding_matrix.shape}')

	return embedding_matrix



def train_test_model(X_train, X_test, y_train, model, historyFilename):
	"""
	Train and test the model using the given data.
	
	Parameters
	------------
	X_train:
		Data selected as predictors for our ML model (training phase)
	X_test: numpy array or dataframe
		Data selected as predictors for our ML model (testing phase)
	y_train: numpy array
		Column with the data used as the class (training phase)
	model:
		Neural network
	historyFilename: string
		Name to save the history values
	
	Return
	------------
	y_pred: array
		Array with the predicted classes
	"""

	print(f'Training...')

	# simple early stopping
	es = EarlyStopping(monitor='val_auc', mode='auto', verbose=1, patience=3)

	history = model.fit(X_train, y_train,
					epochs=EPOCHS,
					batch_size=BATCH_SIZE,
					validation_split=0.20,
					callbacks=[es]
					)


	# Save history
	auc = history.history['auc']
	val_auc = history.history['val_auc']

	loss = history.history['loss']
	val_loss = history.history['val_loss']
  
	total_epochs = list(range(1, len(auc) + 1))

	historyDf = pd.DataFrame(data={'epochs': total_epochs,
									'auc': auc,
									'val_auc': val_auc,
									'loss': loss,
									'val_loss': val_loss})
	historyDf.to_csv(historyFilename, index=False)


	print(f'Testing...')
	
	# get class predictions with the model
	y_pred = np.argmax(model.predict(X_test), axis=-1)


	return y_pred



def save_results(y_test, y_pred, encoder, metricsFilename, confusionFilename, normalizedConfusionFilename):
	"""
	Evaluates the trained model using different metrics from the Pycm confusion matrix.
	
	Parameters
	------------	
		y_test: numpy array
			Column with the data used as the class (training phase)
		y_pred: numpy array
			Array with the predicted classes
		encoder: sklearn.preprocessing.LabelEncoder
			Transformer used for decoding target values
		metricsFilename: string
			Name to save the evaluation results including the path
		confusionFilename: string
			Name to save the confusion matrix including the path
		normalizedConfusionFilename: string
			Name to save the normalized confusion matrix including the path

	Return
	------------
		None
	"""

	print(f'Saving results...')

	# convert one-hot-encoder classes to number value
	y_true = np.argmax(y_test, axis=1)


	cm = ConfusionMatrix(actual_vector=encoder.inverse_transform(y_true),
						predict_vector=encoder.inverse_transform(y_pred),
						digit=5)

	# Métricas de c/clase
	# GM - Geometric mean of specificity and sensitivity
	# MCC - Matthews correlation coefficient
	# PPV - Positive predictive value: precision
	# TPR - True positive rate: sensitivity/recall (e.g. the percentage of sick people who are correctly identified as having the condition)
	summary = pd.DataFrame(cm.class_stat).T
	summary.replace('None', 0, inplace=True)
	summary = summary.reset_index()
	summary = summary[summary['index'].isin(['AUC', 'F1', 'GM', 'MCC', 'PPV', 'TPR', 'ACC'])]

	if ONLY_DISORDERS:
		summary['Macro_Average'] = (summary['ADHD'] + summary['ANXIETY'] + summary['ASD'] + summary['BIPOLAR'] + summary['DEPRESSION'] + summary['EATING'] + summary['OCD'] + summary['PTSD'] + summary['SQUIZOPHRENIA']) / 9.0
	else:
		summary['Macro_Average'] = (summary['ADHD'] + summary['ANXIETY'] + summary['ASD'] + summary['BIPOLAR'] + summary['CONTROL'] + summary['DEPRESSION'] + summary['EATING'] + summary['OCD'] + summary['PTSD'] + summary['SQUIZOPHRENIA']) / 10.0
	
	summary.to_csv(metricsFilename, index=False)


	# Matriz de confusion
	# En la documentación parece ser que columns = predict, rows = actual
	# por eso en el dataframe trasponemos la matriz
	cmDf = pd.DataFrame(cm.matrix).T.reset_index()
	cmDf.to_csv(confusionFilename, index=False)


	cmDf = pd.DataFrame(cm.normalized_matrix).T.reset_index()
	cmDf.to_csv(normalizedConfusionFilename, index=False)


	# Imprimir en pantalla métricas
	print(f'Report metrics...')

	f1_macro = summary[summary['index'] == 'F1']['Macro_Average'].values[0]
	auc_macro = summary[summary['index'] == 'AUC']['Macro_Average'].values[0]

	print(f'F1 = {f1_macro:.3f}\tAUC = {auc_macro:.3f}\n')



def CNN_model(embedding_matrix, num_classes):

	# Building the CNN Model
	model = Sequential()

	# Created Embedding Layer
	model.add(Embedding(VOCAB_SIZE,
						EMBEDDING_DIMS,
						weights=[embedding_matrix],
						trainable=True,
						input_length=MAX_LEN))

	model.add(SpatialDropout1D(0.3))

	# Create the convolutional layer
	model.add(Convolution1D(FILTERS,
							KERNEL_SIZE,
							activation='relu'))

	# Create the pooling layer
	model.add(GlobalMaxPool1D())

	# Create the fully connected layer
	model.add(Dense(HIDDEN_DIMS,
					activation='relu'))
	# Mask some values
	model.add(Dropout(0.25))

	# Create the output layer (num_classes porque es una salida multiclase)
	model.add(Dense(num_classes, activation='softmax'))

	# print(model.summary())

	# compile the keras model
	model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['AUC'])
	
	return model


def LSTM_model(embedding_matrix, num_classes):

	# Building the LSTM Model
	model = Sequential()

	# Created Embedding Layer
	model.add(Embedding(VOCAB_SIZE,
						EMBEDDING_DIMS,
						weights=[embedding_matrix],
						trainable=True,
						input_length=MAX_LEN))

	model.add(SpatialDropout1D(0.3))

	# Create the LSTM layer
	model.add(LSTM(UNITS,
				activation='tanh'))

	# Create the fully connected layer
	model.add(Dense(HIDDEN_DIMS,
					activation='relu'))
	# Mask some values
	model.add(Dropout(0.25))

	# Create the output layer (num_classes porque es una salida multiclase)
	model.add(Dense(num_classes, activation='softmax'))

	# print(model.summary())

	# compile the keras model
	model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['AUC'])
	
	return model


def GRU_model(embedding_matrix, num_classes):

	# Building the RNN Model
	model = Sequential()

	# Created Embedding Layer
	model.add(Embedding(VOCAB_SIZE,
						EMBEDDING_DIMS,
						weights=[embedding_matrix],
						trainable=True,
						input_length=MAX_LEN))

	model.add(SpatialDropout1D(0.3))

	# Create the GRU layer
	model.add(GRU(UNITS,
				activation='tanh'))

	# Create the fully connected layer
	model.add(Dense(HIDDEN_DIMS,
					activation='relu'))
	# Mask some values
	model.add(Dropout(0.25))

	# Create the output layer (num_classes porque es una salida multiclase)
	model.add(Dense(num_classes, activation='softmax'))

	# print(model.summary())

	# compile the keras model
	model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['AUC'])
	
	return model


def CNN_LSTM_model(embedding_matrix, num_classes):

	# Building the CNN Model
	model = Sequential()

	# Created Embedding Layer
	model.add(Embedding(VOCAB_SIZE,
						EMBEDDING_DIMS,
						weights=[embedding_matrix],
						trainable=True,
						input_length=MAX_LEN))

	model.add(SpatialDropout1D(0.3))

	# Create the convolutional layer
	model.add(Convolution1D(FILTERS,
							KERNEL_SIZE,
							activation='relu'))

	# Create the pooling layer
	model.add(MaxPooling1D(3))

	# Create the LSTM layer
	model.add(LSTM(UNITS,
				activation='tanh'))

	# Create the fully connected layer
	model.add(Dense(HIDDEN_DIMS,
					activation='relu'))
	# Mask some values
	model.add(Dropout(0.25))

	# Create the output layer (num_classes porque es una salida multiclase)
	model.add(Dense(num_classes, activation='softmax'))

	# print(model.summary())

	# compile the keras model
	model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['AUC'])
	
	return model



def run_model(dataDirectory, embeddingsDirectory, saveDirectory, nameResults, language, method):
	"""
	Read the train and test files for each fold, train and test the algorithm.

	Parameters
	------------
	dataDirectory: string
		Path of the directory where the folder with the data is located
	embeddingsDirectory: string
		Path of the directory where the folder with the embeddings are located
	saveDirectory: string
		Path of the directory where the folders to save the results are located
	nameResults: string
		Base name of result files
	language: string, {"eng", "esp"}
		Option "eng" for english, option "esp" for spanish experiments
	method: string, {"GLOVE", "FASTTEXT"}
		Supported methods are "GLOVE" for glove embeddings, "FASTTEXT" for fasttext embeddings

	Return
	------------
		None
	"""

	text = 'sbwc' if language == 'esp' else 'crawl'
	ext = 'txt' if method == 'GLOVE' and language == 'eng' else 'vec'

	embeddingsFilename = path.join(embeddingsDirectory, f'{method.lower()}-{text}-{EMBEDDING_DIMS}d.{ext}')	
	embeddings_index = load_embeddings_file(embeddingsFilename)


	print('\n---------------------')
	print(f'\n\n\t{nameResults.upper()} - {language}\n')
	print('\n---------------------')


	for i in range(1, FOLDS + 1):

		print(f'++ Fold > {i}')

		trainFilename = path.join(dataDirectory, f'fold{i}_train_{language}.csv')
		testFilename = path.join(dataDirectory, f'fold{i}_test_{language}.csv')

		X_train, X_test, y_train, y_test, encoder = load_dataset(trainFilename, testFilename)
		X_train, X_test, tokenizer = preprocess_text_format(X_train, X_test)

		embedding_matrix = create_embedding_matrix(tokenizer, embeddings_index)

		num_classes = y_train.shape[1] # Number of classes


		classifiers_list = {'CNN': CNN_model(embedding_matrix, num_classes),
							'LSTM': LSTM_model(embedding_matrix, num_classes),
							'GRU': GRU_model(embedding_matrix, num_classes),
							'CNN-LSTM': CNN_LSTM_model(embedding_matrix, num_classes)
						}


		for nameClf, clf in classifiers_list.items():

			print(f'\n*** {nameClf} ***')

			fold = 'Fold' + str(i)

			fileResults = path.join(saveDirectory, fold, f'fold{i}_{nameClf}_{nameResults}_metrics_{language}.csv')
			fileMatrix = path.join(saveDirectory, fold, f'fold{i}_{nameClf}_{nameResults}_matrix_{language}.txt')
			fileMatrixNorm = path.join(saveDirectory, fold, f'fold{i}_{nameClf}_{nameResults}_normalized_matrix_{language}.txt')
			fileHistory = path.join(saveDirectory, fold, f'fold{i}_{nameClf}_{nameResults}_history_{language}.csv')

			pipe = clf

			try:
				y_pred = train_test_model(X_train, X_test, y_train, pipe, fileHistory)
				save_results(y_test, y_pred, encoder, fileResults, fileMatrix, fileMatrixNorm)
			except Exception as e:
				print(f'An exception occurred: {e}')
			# Clear session
			K.clear_session()
	
	print('END!')


def main(dataDirectory, embeddingsDirectory, resultsDirectory, lang):
	"""
	Run all experiments.

	Parameters
	------------
	partitionDirText: string
		Directory with data having only clear text
	embeddingsDirectory: string
		Directory with embeddings files
	resultsDirectory: string
		Root path of the directory where the folders to save the results are located
	lang: string, {"eng", "esp"}
		Option "eng" for english, option "esp" for spanish experiments

	Return
	------------
		None
	"""

	run_model(dataDirectory, embeddingsDirectory, path.join(resultsDirectory, f'EMBEDDINGS_FASTTEXT_300d_{lang.upper()}'), 'embeddings_fasttext_300d', lang, 'FASTTEXT')
	run_model(dataDirectory, embeddingsDirectory, path.join(resultsDirectory, f'EMBEDDINGS_GLOVE_300d_{lang.upper()}'), 'embeddings_glove_300d', lang, 'GLOVE')



# --------------------------------
if __name__ == '__main__':

	language = 'eng'

	partitionDirText = r'5FCV_Txt_ENG'
	embeddingsDirectory = r'Embeddings_ENG'

	directoryResults = r'/content/drive/MyDrive/MyExperiments/Results'
	main(partitionDirText, embeddingsDirectory, directoryResults, language)




# # ESPAÑOL
# !wget -q http://dcc.uchile.cl/~jperez/word-embeddings/fasttext-sbwc.vec.gz
# !gzip -d -q fasttext-sbwc.vec.gz
# !rm fasttext-sbwc.vec.gz
# !mv fasttext-sbwc.vec fasttext-sbwc-300d.vec

# !wget -q http://dcc.uchile.cl/~jperez/word-embeddings/glove-sbwc.i25.vec.gz
# !gzip -d -q glove-sbwc.i25.vec.gz
# !rm glove-sbwc.i25.vec.gz
# !mv glove-sbwc.i25.vec glove-sbwc-300d.vec


# # INGLES
# !wget -q https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
# !unzip crawl*.zip
# !rm crawl-300d-2M.vec.zip
# !mv crawl-300d-2M.vec fasttext-crawl-300d.vec

# !wget -q https://nlp.stanford.edu/data/glove.42B.300d.zip
# !unzip glove*.zip
# !mv glove.42B.300d.txt glove-crawl-300d.txt
