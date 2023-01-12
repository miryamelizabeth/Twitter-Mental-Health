### enviroment: experiments_env

import pandas as pd
import os


## -----------------------------------------
## PARTE 1
## Extract all results into a single manageable csv file
## -----------------------------------------

## JOIN THE 5 FOLDS RESULTS in one results folder
## fodlsResultsDirectory = WORD_UNIGRAM_10K_ESP 
def joinResultsByFolds(foldsResultsDirectory, folds=5):

	final = pd.DataFrame()

	for i in range(1, folds + 1):

		foldPath = os.path.join(foldsResultsDirectory, f'Fold{i}')

		files = [f for f in os.listdir(foldPath) if ('metrics' in f)]
	
		for name in files: # name = fold1_BaggingDT_word_unigram_10k_metrics_eng.csv

			df = pd.read_csv(os.path.join(foldPath, name))
			df['classifier'] = name.split('_')[1] # BaggingDT
			df['fold'] = f'Fold{i}'
			df['file'] = '-'.join(name.split('_')[2:5]) # word-unigram-10k	style-lf-liwc	embeddings-fasttext-300d

			final = pd.concat([final, df])
	
	# Cambiar nombre de index a metric
	final.rename(columns={'index': 'metric'}, inplace=True)

	return final


## Ignore this function, it is used when you do train-test instead of 5FCV.
## JOIN ALL RESULTS OF THE classifiers in a single results folder
## resultsDirectory = WORD_UNIGRAM_10K_ESP 
def joinResultsByPartition(resultsDirectory):

	final = pd.DataFrame()

	files = [f for f in os.listdir(resultsDirectory) if 'metrics' in f]
	
	for name in files: # name = BaggingDT_word_unigram_10k_metrics_eng.csv

		df = pd.read_csv(os.path.join(resultsDirectory, name))
		df['classifier'] = name.split('_')[0] # BaggingDT
		df['file'] = '-'.join(name.split('_')[1:4]) # word-unigram-10k	style-lf-liwc	embeddings-fasttext-300d
		
		final = pd.concat([final, df])
	
	# Cambiar nombre de index a metric
	final.rename(columns={'index': 'metric'}, inplace=True)

	return final


## SAVES THE RESULTS OF THE 5 FOLDS (OR PARTITION) IN A FILE, FOR EACH DIRECTORY
## resultsDirectory = "C:\Results".
def saveResultsInPartitionFold(resultsDirectory, saveDirectory, cv=True, folds=5):

	directories = [d for d in os.listdir(resultsDirectory) if not d.endswith('.csv')]

	for d in directories:

		print(f'*** {d} ***')
		if cv:
			results = joinResultsByFolds(os.path.join(resultsDirectory, d), folds)
		else:
			results = joinResultsByPartition(os.path.join(resultsDirectory, d))

		# GUARDAR
		name = 'folds' if cv else 'partitions'
		results.to_csv(os.path.join(saveDirectory, f'{name}Results_{d.lower()}.csv'), index=False)


## Group results into a single raw file
def concatResultsALL(saveDirectory, finalFilename, multiclass=True, tag=''):
	
	todos = pd.DataFrame()

	archivos = [f for f in os.listdir(saveDirectory) if f.startswith('foldsResults_') or f.startswith('partitionsResults_')]

	for f in archivos:
		
		print(f'*** {f} ***')
		
		results = pd.read_csv(os.path.join(saveDirectory, f))

		if multiclass:
			params = {'ADHD': ['mean', 'std'],
					'ANXIETY': ['mean', 'std'],
					'ASD': ['mean', 'std'],
					'BIPOLAR': ['mean', 'std'],
					# 'CONTROL': ['mean', 'std'],
					'DEPRESSION': ['mean', 'std'],
					'EATING': ['mean', 'std'],
					'OCD': ['mean', 'std'],
					'PTSD': ['mean', 'std'],
					'SQUIZOPHRENIA': ['mean', 'std'],
					'Macro_Average': ['mean', 'std'],
					'file': ['first']
					}
		else:
			params = {tag: ['mean', 'std'],
					'CONTROL': ['mean', 'std'],
					'Macro_Average': ['mean', 'std'],
					'file': ['first']
					}

		holi = results.groupby(by=['classifier', 'metric']).agg(params).reset_index()
		todos = pd.concat([todos, holi])

	todos.columns = ['_'.join(x) for x in todos.columns.ravel()]

	# GUARDAR
	todos.to_csv(os.path.join(saveDirectory, finalFilename), index=False)




## -----------------------------------------
## PART 2
## Format results for Microsoft Excel
## -----------------------------------------
## Support function
def sorter(column):
	
	reorder = ['NaiveBayes (NB)', 'Linear SVM',
				'DecisionTree (DT)', 'Bagging with DT (BDT)',
				'XGBoost (XGB)']
	
	cat = pd.Categorical(column, categories=reorder, ordered=True)

	return pd.Series(cat)

## Formatea el archivo todo junto
def formatResultsDisorders(sourceDirectory, destinyDirectory, filenameResults):

	print(f'\n*** {filenameResults.upper()} ***')


	# Leer datos
	df = pd.read_csv(os.path.join(sourceDirectory, filenameResults))
	print(f'\nDatos: {df.shape}')


	# Eliminar datos no necesarios
	# (estas métricas no las presento, pero están en el archivo crudo por si acaso)
	print(f'\nEliminando datos...')
	df = df[~df['metric_'].isin(['GM', 'MCC'])]
	print(df.shape)


	# Seleccionar y Renombrar columnas
	# por el momento la desviación estandar (std) no la presento jeje
	print(f'\nSeleccionar y Renombrar columnas')
	cols = [c for c in df.columns if not c.endswith('_std')]
	df = df[cols]
	print(df.shape)

	colsNewNames = dict()
	for c in cols:

		old = c
		c = c.replace('_mean', '').replace('_first', '')

		if c in ['ANXIETY', 'EATING']: colsNewNames[old] = c[:3]
		elif c in ['BIPOLAR', 'DEPRESSION']: colsNewNames[old] = c[:4]
		elif c == 'SQUIZOPHRENIA': colsNewNames[old] = 'SCHIZO'
		elif c == 'CONTROL': colsNewNames[old] = 'NORMAL'
		elif c == 'ASD': colsNewNames[old] = 'AUTS'
		elif c == 'file': colsNewNames[old] = 'Feature'
		elif c in ['classifier_', 'metric_']: colsNewNames[old] = c[:-1].capitalize()
		elif c.isupper(): colsNewNames[old] = c # Otros desordenes que faltan
		else: colsNewNames[old] = c.title()

	df.rename(columns=colsNewNames, inplace=True)


	# Renombrar datos: clasificadores, métricas y features
	print(f'\nRenombrar datos')

	classifiers = {'BaggingDT': 'Bagging with DT (BDT)',
					'DecisionTree': 'DecisionTree (DT)',
					'XGBoost': 'XGBoost (XGB)',
					'LinearSVM': 'Linear SVM',
					'NaiveBayes': 'NaiveBayes (NB)',
					}
	
	metrics = {'TPR': 'Recall',
				'PPV': 'Precision',
				'ACC': 'Accuracy',
				}

	features = {'character-wb-3gram-10k': 'Char 3gram (C3)',
				'character-wb-5gram-10k': 'Char 5gram (C5)',
				'character-wb-7gram-10k': 'Char 7gram (C7)',
				'postag-bigram-10k': 'POS Bigram (POS_B)',
				'postag-trigram-10k': 'POS Trigram (POS_T)',
				'postag-unibitrigram-10k': 'POS 1,2,3gram (POS_UBT)',
				'postag-unigram-10k': 'POS Unigram (POS_U)',
				'topic-lda50-10k': 'Topic LDA',
				'topic-nmf50-10k': 'Topic NMF',
				'word-bigram-10k': 'Bigram (B)',
				'word-trigram-10k': 'Trigram (T)',
				'word-unibitrigram-10k': 'Word 1,2,3gram (UBT)',
				'word-unigram-10k': 'Unigram (U)',
				'embeddings-fasttext-300d': 'Embeddings fastText',
				'embeddings-glove-300d': 'Embeddings GloVe',
				'style-lf-liwc': 'LIWC'
				}

	df['Classifier'].replace(classifiers, inplace=True)
	df['Metric'].replace(metrics, inplace=True)
	df['Feature'].replace(features, inplace=True)


	# Separar por métrica, ordenar por features y clasificador
	print(f'\nOrdenar datos')

	ordenFeatures = ['Unigram (U)', 'Bigram (B)', 'Trigram (T)', 'Word 1,2,3gram (UBT)',
					'Char 3gram (C3)', 'Char 5gram (C5)', 'Char 7gram (C7)',
					'POS Unigram (POS_U)', 'POS Bigram (POS_B)', 'POS Trigram (POS_T)', 'POS 1,2,3gram (POS_UBT)',
					'Topic LDA', 'Topic NMF',
					'LIWC',
					'Embeddings fastText', 'Embeddings GloVe'
					]

	general = pd.DataFrame()
	for metric in ['AUC', 'F1', 'Precision', 'Recall', 'Accuracy']:
		print(metric)
		
		subsetMetric = df[df['Metric'] == metric]

		temp = pd.DataFrame()
		for method in ordenFeatures:

			if 'Embeddings' not in method:
				subsetMethod = subsetMetric[subsetMetric['Feature'] == method].sort_values(by='Classifier', key=sorter)
			else:
				subsetMethod = subsetMetric[subsetMetric['Feature'] == method].sort_values(by='Classifier')

			temp = pd.concat([temp, subsetMethod])

		general = pd.concat([general, temp])


	# Checar...
	# print(general)


	# ORDENAR finalmente las columnas
	print(f'\nOrdenar columnas')
	lista = general.columns.tolist()
	lista.remove('Feature')
	lista.remove('Classifier')
	lista.remove('Metric')
	lista.remove('Macro_Average')

	if len(lista) == 2:
		lista.remove('NORMAL')
		ordenColumnas = ['Feature', 'Classifier', 'Metric', lista[0], 'NORMAL', 'Macro_Average']
	else:
		lista.sort()
		ordenColumnas = ['Feature', 'Classifier', 'Metric'] + lista + ['Macro_Average']

	# Guardar
	print(f'Guardar TABLA formateada <3')
	conjuntoFinal = general[ordenColumnas]
	conjuntoFinal.to_csv(os.path.join(destinyDirectory, f'FORMAT_{filenameResults}'), float_format='%.3f', index=False)





# -------------------------------------------
# PARTE 1
resultsDirectory = r'RESULTS'
saveDirectory = r'Fold files'
finalFilename = f'ALL_Results.csv'

saveResultsInPartitionFold(resultsDirectory, saveDirectory)
# For multiclasss
concatResultsALL(saveDirectory, finalFilename)
# # For binary
# concatResultsALL(saveDirectory, finalFilename, False, 'DIAGNOSED')


# -------------------------------------------
# PARTE 2
formatResultsDisorders(saveDirectory, saveDirectory, finalFilename)
