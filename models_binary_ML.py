### enviroment: experiments_env

import pandas as pd
import time

from joblib import dump, load
from os import path

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

from sklearn.pipeline import Pipeline
from mlxtend.preprocessing import DenseTransformer

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier

from pycm import ConfusionMatrix
# https://www.pycm.ir/doc/index.html#Full

import warnings
warnings.filterwarnings('ignore')



# ---------------------------
n_jobs = 8
seed = 42
FOLDS = 5

disorderTag = 'DIAGNOSED'
binaryDvsC = True
# ---------------------------


def load_dataset(trainFile, testFile, method):
	"""
	Read datasets in csv format using pandas.
	
	Parameters
	------------
	trainFile: string
		Name of training file including path
	testFile: string
		Name of training file including path
	method: string, {"NGRAM", "NMF", "LDA", "LIWC"}
		Supported methods are "NGRAM" for n-grams and POS n-grams,
		"NMF" and "LDA" for topic modeling, and "LIWC"

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


	if binaryDvsC:
		train['class'].replace({'ADHD': 'DIAGNOSED', 'ANXIETY': 'DIAGNOSED',
								'ASD': 'DIAGNOSED', 'BIPOLAR': 'DIAGNOSED',
								'DEPRESSION': 'DIAGNOSED', 'EATING': 'DIAGNOSED',
								'OCD': 'DIAGNOSED', 'PTSD': 'DIAGNOSED',
								'SQUIZOPHRENIA': 'DIAGNOSED'}, inplace=True)
		test['class'].replace({'ADHD': 'DIAGNOSED', 'ANXIETY': 'DIAGNOSED',
								'ASD': 'DIAGNOSED', 'BIPOLAR': 'DIAGNOSED',
								'DEPRESSION': 'DIAGNOSED', 'EATING': 'DIAGNOSED',
								'OCD': 'DIAGNOSED', 'PTSD': 'DIAGNOSED',
								'SQUIZOPHRENIA': 'DIAGNOSED'}, inplace=True)

	
	train = train[train['class'].isin(['CONTROL', disorderTag])]
	test = test[test['class'].isin(['CONTROL', disorderTag])]


	if method == 'NGRAM' or method == 'NMF' or method == 'LDA':
		X_train = train['tweets_user'].values
		X_test = test['tweets_user'].values

	elif method == 'LIWC':
		X_train = train.drop(['class'], axis=1)
		X_test = test.drop(['class'], axis=1)

		X_train.fillna(0, inplace=True)
		X_test.fillna(0, inplace=True)

	else:
		raise Exception('Unknown method, it must be "NGRAM", "NMF", "LDA" or "LIWC".')


	y_train = train['class'].values
	y_test = test['class'].values

	encoder = LabelEncoder()
	encoder.fit(y_train)

	y_train = encoder.transform(y_train)
	y_test = encoder.transform(y_test)

	return X_train, X_test, y_train, y_test, encoder



def train_test_model(X_train, X_test, y_train, model, saveModel=False, filename=None):
	"""
	Train and test the model using the given data.
	
	Parameters
	------------
	X_train: numpy array or dataframe
		Data selected as predictors for our ML model (training phase)
	X_test: numpy array or dataframe
		Data selected as predictors for our ML model (testing phase)
	y_train: numpy array
		Column with the data used as the class (training phase)
	model: sklearn.pipeline.Pipeline
		Pipeline of transforms with a final estimator.
		Sequentially apply a list of transforms and a final classifier
	saveModel: boolean, default=False
		True to save the models with pickle, otherwise False
	filename: string, default=None
		Name to save the trained model including the path
	
	Return
	------------
	y_pred: numpy array
		Array with the predicted classes
	"""

	print(f'Training...')
	model.fit(X_train, y_train)


	if saveModel:
		print(f'Saving...')
		dump(model, filename)


	print(f'Testing...')
	y_pred = model.predict(X_test)


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

	cm = ConfusionMatrix(actual_vector=encoder.inverse_transform(y_test),
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
	summary['Macro_Average'] = (summary['CONTROL'] + summary[disorderTag]) / 2.0
	
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

	F1_macro = summary[summary['index'] == 'F1']['Macro_Average'].values[0]
	AUC = summary[summary['index'] == 'AUC']['Macro_Average'].values[0]

	print(f'F1 = {F1_macro:.3f}\tAUC = {AUC:.3f}\n')



def get_objects(method, ngram_range, ngram_type, n_features, n_topics):
	"""
	Train and test the model using the given data.
	
	Parameters
	------------
	method: string, {"NGRAM", "NMF", "LDA", "LIWC"}
		Supported methods are "NGRAM" for n-grams and POS n-grams,
		"NMF" and "LDA" for topic modeling, and "LIWC"
	ngram_range: tuple
		The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted.
		All values of n such such that min_n <= n <= max_n will be used.
		For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams.
	ngram_type: string, {"word", "char", "char_wb"}
		Whether the feature should be made of word n-gram or character n-grams.
		Option 'char_wb' creates character n-grams only from text inside word boundaries;
		n-grams at the edges of words are padded with space.
	n_features: int
		Size of the vocabulary
	n_topics: int
		Number of topics to extract
	
	Return
	------------
	vectorizer: sklearn.feature_extraction.text.CountVectorizer, or sklearn.feature_extraction.text.TfidfVectorizer
		Vectorizer object
	topic: sklearn.decomposition.LatentDirichletAllocation, or sklearn.decomposition.NMF
		Topic object
	"""

	if method == 'LDA':
		vectorizer = CountVectorizer(ngram_range=ngram_range, analyzer=ngram_type, min_df=15, max_df=0.75, max_features=n_features)
	elif method == 'NGRAM' or method == 'NMF' or method == 'MIX2':
		vectorizer = TfidfVectorizer(ngram_range=ngram_range, analyzer=ngram_type, min_df=15, max_df=0.75, max_features=n_features)
	else:
		vectorizer = None
	

	if method == 'LDA':
		topic = LatentDirichletAllocation(n_components=n_topics, learning_method='online', random_state=seed, n_jobs=n_jobs)
	elif method == 'NMF':
		topic = NMF(n_components=n_topics, random_state=seed) # alpha=1
	else:
		topic = None


	return vectorizer, topic



def run_model(dataDirectory, saveDirectory, nameResults, language, ngram_range, ngram_type, n_features, n_topics, method):
	"""
	Read the train and test files for each fold, train and test the algorithm.

	Parameters
	------------
	dataDirectory: string
		Path of the directory where the folder with the data is located
	saveDirectory: string
		Path of the directory where the folders to save the results are located
	nameResults: string
		Base name of result files
	language: string, {"eng", "esp"}
		Option "eng" for english, option "esp" for spanish experiments
	ngram_range: tuple
		The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted.
		All values of n such such that min_n <= n <= max_n will be used.
		For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams.
	ngram_type: string, {"word", "char", "char_wb"}
		Whether the feature should be made of word n-gram or character n-grams.
		Option 'char_wb' creates character n-grams only from text inside word boundaries;
		n-grams at the edges of words are padded with space.
	n_features: int
		Size of the vocabulary
	n_topics: int
		Number of topics to extract
	method: string, {"NGRAM", "NMF", "LDA", "LIWC"}
		Supported methods are "NGRAM" for n-grams and POS n-grams,
		"NMF" and "LDA" for topic modeling, and "LIWC"

	Return
	------------
		None
	"""

	print('\n---------------------')
	print(f'\n\n\t{nameResults.upper()} - {language}\n')
	print('\n---------------------')

	classifiers_list = {'DecisionTree': DecisionTreeClassifier(random_state=seed),
						'NaiveBayes': GaussianNB(),
						'BaggingDT': BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=seed, n_jobs=n_jobs),
						'LinearSVM': SVC(kernel='linear', probability=True, random_state=seed),
						'XGBoost': XGBClassifier()
					}

	vectorizer, topic = get_objects(method, ngram_range, ngram_type, n_features, n_topics)



	for i in range(1, FOLDS + 1):

		print(f'++ Fold > {i}')

		trainFilename = path.join(dataDirectory, f'fold{i}_train_{language}.csv')
		testFilename = path.join(dataDirectory, f'fold{i}_test_{language}.csv')

		X_train, X_test, y_train, y_test, encoder = load_dataset(trainFilename, testFilename, method)


		for nameClf, clf in classifiers_list.items():

			print(f'\n*** {nameClf} ***')

			fold = f'Fold{i}'

			fileResults = path.join(saveDirectory, fold, f'fold{i}_{nameClf}_{nameResults}_metrics_{language}.csv')
			fileMatrix = path.join(saveDirectory, fold, f'fold{i}_{nameClf}_{nameResults}_matrix_{language}.txt')
			fileMatrixNorm = path.join(saveDirectory, fold, f'fold{i}_{nameClf}_{nameResults}_normalized_matrix_{language}.txt')
			fileModel = path.join(saveDirectory, fold, f'fold{i}_{nameClf}_{nameResults}_model_{language}.joblib')


			if method == 'NGRAM':
				pipe = Pipeline([('vec', vectorizer), ('to_dense', DenseTransformer()), ('clf', clf)])

			elif method == 'NMF' or method == 'LDA':
				pipe = Pipeline([('vec', vectorizer), ('to_dense', DenseTransformer()), ('topic', topic), ('clf', clf)])

			elif method == 'LIWC':
				pipe = Pipeline([('preprocessor', MinMaxScaler()), ('clf', clf)])

			else:
				pipe = Pipeline([('clf', clf)])


			try:
				print(pipe)
				y_pred = train_test_model(X_train, X_test, y_train, pipe)
				save_results(y_test, y_pred, encoder, fileResults, fileMatrix, fileMatrixNorm)
			except Exception as e:
				print(f'An exception occurred: {e}')
	
	print('END!')


def main(partitionDirText, partitionDirPos, partitionDirLiwc, partitionDirOtros, resultsDirectory, lang):
	"""
	Run all experiments.

	Parameters
	------------
	partitionDirText: string
		Directory with data having only clear text
	partitionDirPos: string
		Directory with data having only clear text and the attached POS tag
	partitionDirLiwc: string
		Directory with data from the LIWC tool
	partitionDirOtros: list of strings
		List of directories with other data types
	resultsDirectory: string
		Root path of the directory where the folders to save the results are located
	lang: string, {"eng", "esp"}
		Option "eng" for english, option "esp" for spanish experiments

	Return
	------------
		None
	"""

	max_features = 10000

	ngram_type = 'word'
	unigram_range = (1, 1)
	bigram_range = (2, 2)
	trigram_range = (3, 3)
	unibitri_range = (1, 3)

	qgram_type = 'char_wb'
	trigram_range = (3, 3)
	quingram_range = (5, 5)
	sepgram_range = (7, 7)

	n_topics = 50


	# --------------------------
	# TF-IDF Tokenization - WORDS
	# --------------------------	
	run_model(partitionDirText, path.join(resultsDirectory, f'WORD_UNIGRAM_10K_{lang.upper()}'), 'word_unigram_10k', lang, unigram_range, ngram_type, max_features, None, 'NGRAM')
	run_model(partitionDirText, path.join(resultsDirectory, f'WORD_BIGRAM_10K_{lang.upper()}'), 'word_bigram_10k', lang, bigram_range, ngram_type, max_features, None, 'NGRAM')
	run_model(partitionDirText, path.join(resultsDirectory, f'WORD_TRIGRAM_10K_{lang.upper()}'), 'word_trigram_10k', lang, trigram_range, ngram_type, max_features, None, 'NGRAM')
	run_model(partitionDirText, path.join(resultsDirectory, f'WORD_UNIBITRIGRAM_10K_{lang.upper()}'), 'word_unibitrigram_10k', lang, unibitri_range, ngram_type, max_features, None, 'NGRAM')
	

	# --------------------------
	# Part-of-speech tags
	# --------------------------
	run_model(partitionDirPos, path.join(resultsDirectory, f'POSTAG_UNIGRAM_10K_{lang.upper()}'), 'postag_unigram_10k', lang, unigram_range, ngram_type, max_features, None, 'NGRAM')
	run_model(partitionDirPos, path.join(resultsDirectory, f'POSTAG_BIGRAM_10K_{lang.upper()}'), 'postag_bigram_10k', lang, bigram_range, ngram_type, max_features, None, 'NGRAM')
	run_model(partitionDirPos, path.join(resultsDirectory, f'POSTAG_TRIGRAM_10K_{lang.upper()}'), 'postag_trigram_10k', lang, trigram_range, ngram_type, max_features, None, 'NGRAM')
	run_model(partitionDirPos, path.join(resultsDirectory, f'POSTAG_UNIBITRIGRAM_10K_{lang.upper()}'), 'postag_unibitrigram_10k', lang, unibitri_range, ngram_type, max_features, None, 'NGRAM')


	# --------------------------
	# TF-IDF Tokenization - CHARACTERS
	# --------------------------
	run_model(partitionDirText, path.join(resultsDirectory, f'CHARACTER-WB_3GRAM_10K_{lang.upper()}'), 'character-wb_3gram_10k', lang, trigram_range, qgram_type, max_features, None, 'NGRAM')
	run_model(partitionDirText, path.join(resultsDirectory, f'CHARACTER-WB_5GRAM_10K_{lang.upper()}'), 'character-wb_5gram_10k', lang, quingram_range, qgram_type, max_features, None, 'NGRAM')
	run_model(partitionDirText, path.join(resultsDirectory, f'CHARACTER-WB_7GRAM_10K_{lang.upper()}'), 'character-wb_7gram_10k', lang, sepgram_range, qgram_type, max_features, None, 'NGRAM')


	# --------------------------
	# Topic modeling
	# --------------------------
	run_model(partitionDirText, path.join(resultsDirectory, f'TOPIC_NMF{n_topics}_10K_{lang.upper()}'), f'topic_nmf{n_topics}_10k', lang, unigram_range, ngram_type, max_features, n_topics, 'NMF')
	run_model(partitionDirText, path.join(resultsDirectory, f'TOPIC_LDA{n_topics}_10K_{lang.upper()}'), f'topic_lda{n_topics}_10k', lang, unigram_range, ngram_type, max_features, n_topics, 'LDA')


	# --------------------------
	# LIWC
	# --------------------------
	run_model(partitionDirLiwc, path.join(resultsDirectory, f'STYLE_LF_LIWC_{lang.upper()}'), 'style_lf_liwc', lang, None, None, None, None, 'LIWC')
	

	# Others...
	# run_model(partitionDirOtros[0], path.join(resultsDirectory, f'STYLE_LF_ALL_{lang.upper()}'), 'style_lf_all', lang, None, None, None, None, 'ALL')



# --------------------------------
if __name__ == '__main__':

	language = 'eng'

	partitionDirText = r'Partitions\5FCV_Txt_ENG'
	partitionDirPos = r'Partitions\5FCV_Pos_ENG'
	partitionDirLiwc = r'Partitions\5FCV_LfLiwc_ENG'
	partitionDirOtros = [r'Partitions\5FCV_Other1_ENG', r'Partitions\5FCV_Other2_ENG'] # ...

	directoryResults = r'C:\Users\JaneDoe\RESULTS'
	main(partitionDirText, partitionDirPos, partitionDirLiwc, partitionDirOtros, directoryResults, language)
