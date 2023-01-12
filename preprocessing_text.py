### enviroment: experiments_env

## -------------------------------------
# ANTES DE EJECUTAR...
# Especificar --> idioma ingles/español
# Especificar --> n_partitions
# Cambiar las rutas de los directorios en el main
## -------------------------------------

import dask.dataframe as ddf
import pandas as pd
import numpy as np
import spacy as sp
import time, re, os

from datetime import datetime

from pysentimiento.preprocessing import preprocess_tweet


# ---------------------------------------
# CAMBIAR por lo que queramos...
languageName = 'english'
languageCode = 'en'

n_partitions = 6
# ---------------------------------------

nlp = sp.load('es_core_news_sm', disable=['parser', 'ner', 'textcat']) if languageName == 'spanish' else sp.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
nlp.add_pipe('sentencizer')


STOPWORDS_ESP = set(['a', 'ah', 'ahi', 'ahí', 'al', 'algo', 'algunas', 'algunos', 'alli', 'allí', 'ante', 'antes',
						'aqui', 'aquí', 'asi', 'así', 'b', 'c', 'cada', 'cómo', 'con', 'contra', 'cual', 'cuál', 'cuando',
						'cuándo', 'd', 'da', 'de', 'del', 'desde', 'donde', 'dónde', 'durante', 'e', 'el', 'en', 'entre',
						'era', 'erais', 'éramos', 'eramos', 'eran', 'eras', 'eres', 'es', 'esa', 'esas', 'ese', 'eso', 'esos',
						'esta', 'estaba', 'estabais', 'estaban', 'estabas', 'estad', 'estada', 'estadas', 'estado', 'estados',
						'estamos', 'estando', 'estar', 'estaremos', 'estará', 'estara', 'estarán', 'estaran', 'estarás',
						'estaras', 'estaré', 'estare', 'estaréis', 'estareis', 'estaría', 'estaria', 'estaríais', 'estariais',
						'estaríamos', 'estariamos', 'estarían', 'estarian', 'estarías', 'estarias', 'estas', 'este',
						'estemos', 'esto', 'estos', 'estoy', 'estuve', 'estuviera', 'estuvierais', 'estuvieran', 'estuvieras',
						'estuvieron', 'estuviese', 'estuvieseis', 'estuviesen', 'estuvieses', 'estuvimos', 'estuviste',
						'estuvisteis', 'estuviéramos', 'estuviésemos', 'estuvo', 'está', 'estábamos', 'estabamos',
						'estáis', 'estais', 'están', 'estan', 'estás', 'estas', 'esté', 'este', 'estéis', 'esteis',
						'estén', 'esten', 'estés', 'estes', 'f', 'for', 'ft', 'fue', 'fuera', 'fuerais', 'fueran', 'fueras',
						'fueron', 'fuese', 'fueseis', 'fuesen', 'fueses', 'fui', 'fuimos', 'fuiste', 'fuisteis',
						'fuéramos', 'fueramos', 'fuésemos', 'fuesemos', 'g', 'go', 'gt', 'h', 'ha', 'ha', 'habida',
						'habidas', 'habido', 'habidos', 'habiendo', 'habremos', 'habrá', 'habra', 'habrán', 'habran',
						'habrás', 'habras', 'habré', 'habre', 'habréis', 'habreis', 'habría', 'habria', 'habríais', 'habriais',
						'habríamos', 'habriamos', 'habrían', 'habrian', 'habrías', 'habrias', 'habéis', 'habeis',
						'había', 'habia', 'habíais', 'habiais', 'habíamos', 'habiamos', 'habían', 'habian', 'habías', 'habias',
						'han', 'has', 'hasta', 'hay', 'haya', 'hayamos', 'hayan', 'hayas', 'hayáis', 'hayais', 'he',
						'hemos', 'hube', 'hubiera', 
						'hubierais', 'hubieran', 'hubieras', 'hubieron', 'hubiese', 'hubieseis', 'hubiesen', 'hubieses',
						'hubimos', 'hubiste', 'hubisteis', 'hubiéramos', 'hubieramos', 'hubiésemos', 'hubiesemos', 'hubo',
						'i', 'j', 'k', 'l', 'lt', 'la', 'las', 'los', 'm', 'más', 'mas', 'mi', 'mis', 'mucho', 'muchos', 'muy', 'n', 'nada',
						'ni', 'o', 'oh', 'otra',
						'otras', 'otro', 'otros', 'p', 'para', 'pero', 'poco', 'por', 'pork', 'porq', 'porque', 'q',
						'que', 'quien', 'quienes', 'qué', 'que', 'r', 'rt', 's', 'sea', 'seamos', 'sean', 'seas', 'sentid',
						'sentida', 'sentidas', 'sentido', 'sentidos', 'ser', 'seremos', 'será', 'serán', 'serás', 'seré',
						'seréis', 'sería', 'seríais', 'seríamos', 'serían', 'serías', 'seáis',
						'sera', 'seran', 'seras', 'sere', 'sereis', 'seria', 'seriais', 'seriamos', 'serian', 'serias', 'seais',
						'siente', 'sin',
						'sintiendo', 'sobre', 'sois', 'somos', 'son', 'soy', 'su', 'sus', 't', 'también', 'tanto',
						'tendremos', 'tendrá', 'tendrán', 'tendrás', 'tendré', 'tendréis', 'tendría', 'tendríais',
						'tendríamos', 'tendrían', 'tendrías',
						'tendra', 'tendran', 'tendras', 'tendre', 'tendreis', 'tendria', 'tendriais',
						'tendriamos', 'tendrian', 'tendrias',
						'tened', 'tenemos', 'tener', 'tenga', 'tengamos', 'tengan',
						'tengas', 'tengo', 'tengáis', 'tengais', 'tenida', 'tenidas', 'tenido', 'tenidos', 'teniendo', 'tenéis', 'teneis',
						'tenía', 'teníais', 'teníamos', 'tenían', 'tenías',
						'tenia', 'teniais', 'teniamos', 'tenian', 'tenias', 'tiene', 'tienen', 'tienes', 'todo', 'todos',
						'tu', 'tus', 'tuve', 'tuviera', 'tuvierais', 'tuvieran', 'tuvieras', 'tuvieron', 'tuviese',
						'tuvieseis', 'tuviesen', 'tuvieses', 'tuvimos', 'tuviste', 'tuvisteis', 'tuviéramos',
						'tuviésemos', 'tuvo', 'u', 'un', 'una', 'uno', 'unos', 'v', 'va', 'vez', 'w', 'x', 'xk', 'xq',
						'y', 'ya', 'z'])

STOPWORDS_ENG = set(['a', 'about', 'above', 'after', 'again', 'against', 'ah', 'ai', 'ain', 'ain"t', 'aint', 'all', 'am',
						'amp', 'an', 'and', 'any', 'are', 'aren', 'aren"t', 'arent', 'as', 'at', 'b', 'bc', 'be', 'because',
						'been',	'before', 'being', 'below', 'between', 'both', 'but', 'by', 'c', 'can', 'couldn', 'couldn"t',
						'couldnt', 'd', 'did', 'didn', 'didn"t', 'didnt', 'do', 'does', 'doesn', 'doesn"t', 'doesnt',
						'doing', 'don', 'don"t', 'dont', 'down',
						'during', 'e', 'each', 'f', 'few', 'for', 'from', 'ft', 'further', 'g', 'get', 'getta', 'gon',
						'gonna', 'h', 'had', 'hadn', 'hadn"t', 'hadnt', 'has', 'hasn', 'hasn"t', 'hasnt', 'have', 'haven',
						'haven"t', 'havent',
						'having', 'here', 'how', 'if', 'in', 'into', 'is',
						'isn', 'isn"t', 'isnt', 'it"s', 'j', 'just', 'k', 'l', 'll', 'lt', 'm', 'ma',
						'mightn', 'mightn"t', 'mightnt', 'more', 'most', 'mustn', 'mustn"t', 'n',
						'na', 'needn', 'needn"t', 'neednt',
						'nor', 'now', 'o', 'of', 'off', 'oh', 'on', 'once', 'only', 'or', 'other', 'out', 'over', 'own',
						'p', 'q', 'r', 're', 'rn', 'rt', 's', 'same', 'shan', 'shan"t', 'shant', 'she"s', 'shes',
						'should', 'should"ve', 'shouldve', 'shouldn', 'shouldn"t', 'shouldnt', 'so', 'some', 'such',
						't', 'ta', 'than', 'that', 'that"ll', 'thatll',
						'the', 'then', 'there', 'these', 'this', 'those', 'through', 'to', 'too', 'under', 'until',
						'u', 'up', 'ur', 'v', 've', 'very', 'vs', 'w', 'was', 'wasn', 'wasn"t', 'wasnt', 'were',
						'weren', 'weren"t', 'werent', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why',
						'will', 'with', 'won', 'won"t', 'wont', 'wouldn', 'wouldn"t', 'wouldnt', 'x', 'y', 'yall',
						'you"d', 'youd', 'you"ll', 'youll', 'you"re', 'youre', 'you"ve', 'youve' 'z'])


# -------------------------------------
# Remove stopwords
def removeStopwords(text, languageCode):
	""" Elimina stopwords usando la lista personalizada de arriba """

	stopwords = STOPWORDS_ESP if languageCode == 'es' else STOPWORDS_ENG

	text = ' '.join([word for word in text.split() if word not in stopwords])

	return re.sub(' +', ' ', text).strip()


# Remove punctuation
def removePunctuation(text):
	""" Elimina signos de puntuación incluidos ¿! (usados en español) """

	punctuationStr = r'[_━🇧.▪"\[!"#\$%&\(\)\*\+,-\./:;<=>\?\^`{\|}~¿¡¬‘’£¥€¢₩°«»“”— ´¨¸•¤‹›–…·\]]'

	text = re.sub(punctuationStr, ' ', text) 

	return re.sub(' +', ' ', text).strip()


# Preprocess tweet
def helper_preprocess(text, demojiFlag):
	""" Utiliza la función de pysentimiento para pre-procesar tweets, basicamente quita el nombre del usuario y coloca la etiqueta @usuario,
	para el url deja la palabra URL nada más, quita el símbolo de hashtag, no quita emojis ni los convierte a palabras , quita letras repetidas en las palabras,
	normaliza risas jajajja --> jajaja """

	return preprocess_tweet(text, lang=languageCode, user_token="@usuario",
									url_token="url", preprocess_hashtags=True,
									hashtag_token=None, demoji=demojiFlag,
									shorten=3, normalize_laughter=True,
									emoji_wrapper="emoji")


# -------------------------------------
# Lemmatization and POS tag
# -------------------------------------
def lemmatizeAndPOStagText(text):
	""" Se utiliza spacy para la lemmatizacion de las palabras y obtener su POS tag """

	# nlp es parte de spacy
	doc = nlp(text)

	lista_lemmatized = []
	lista_postags_text = []

	for token in doc:
		lista_lemmatized.append(token.lemma_)
		lista_postags_text.append(f'{token.lemma_}_{token.pos_}')


	text1 = ' '.join(lista_lemmatized).strip()
	text2 = ' '.join(lista_postags_text).strip()
	
	return text1, text2



# -------------------------------------
def processText(allText, demojiFlag):
	""" Limpia toda la basura del texto y devuelve el texto limpio """

	# Remove extra newlines
	allText = [re.sub(r'[\r|\n|\r\n]+', ' ', t) for t in allText]

	# Remove extra whitespace
	allText = [re.sub(' +', ' ', t).strip() for t in allText]

	# Replace symbols (eg. I’m --> I'm   that´s --> that's)
	allText = [re.sub('’', '\'', t) for t in allText]
	allText = [re.sub('”', '\'', t) for t in allText]
	allText = [re.sub('´', '\'', t) for t in allText]
	allText = [re.sub('"', '\'', t) for t in allText]

	allText = [re.sub('‑', '-', t) for t in allText]
	allText = [re.sub('—', '-', t) for t in allText]

	# Preprocess tweet using pysentimiento
	allText = [helper_preprocess(t, demojiFlag) for t in allText]

	allText = [removePunctuation(t) for t in allText]

	# Lowercase
	allText = [t.lower() for t in allText]

	return allText




def cleanProcessDataframe(df):
	""" Procesar el texto y aplicar todas las técnicas programadas anteriormente, también divide la fecha... """

	df[['day','time']] = df['tweet_created_at'].str.split(' ', expand=True,)


	df['clean_tweet'] = processText(df['tweet_text'].values, demojiFlag=False)

	result1 = []
	result2 = []
	for t in df['clean_tweet'].values:
		lst1, lst2 = lemmatizeAndPOStagText(t)
		result1.append(lst1)
		result2.append(lst2)

	df['clean_tweet_lemma'] = result1
	df['clean_tweet_lemma_postags'] = result2


	df['clean_tweet_nostop'] = df['clean_tweet'].apply(lambda text: removeStopwords(text, languageCode))

	result1 = []
	result2 = []
	for t in df['clean_tweet_nostop'].values:
		lst1, lst2 = lemmatizeAndPOStagText(t)
		result1.append(lst1)
		result2.append(lst2)


	df['clean_tweet_nostop_lemma'] = result1
	df['clean_tweet_nostop_lemma_postags'] = result2


	df['clean_description'] = processText(df['user_description'].values, demojiFlag=False)

	return df


# -------------------------------------
def convertNum(value):
	return 0 if (value == np.nan) else value

def convertText(value):
	return '' if (value == np.nan) else value



# -----------------------------------------------------------------
if __name__ == '__main__':

	# timelineDirectory tiene todos los archivos descargados por usuario
	timelineDirectory = r'Timelines'
	# aquí se guardan los archivos limpios
	cleanUsersDirectory = r'Clean_users'


	print(f'*** {timelineDirectory} ***')

	usuarios = os.listdir(timelineDirectory)
	print(f'Tamaño lista: {len(usuarios)}')


	# iterar sobre c/usuario
	for user in usuarios[:len(usuarios)]:

		print(f'\n*** {user}***\n')
		print(f'{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}')
		
		print('Start...')
		count = 0

		start_time = time.time()

		df = pd.read_csv(os.path.join(timelineDirectory, user), low_memory=True,
						usecols=['tweet_id', 'tweet_created_at', 'tweet_lang', 'tweet_text', 'tweet_source',
						'tweet_favorited', 'tweet_retweeted', 'tweet_favorite_count', 'tweet_retweet_count',
						'tweet_hashtags', 'tweet_user_mentions',
						'user_id', 'user_name', 'user_screen_name', 'user_description', 'user_location',
						'user_followers_count', 'user_friends_count', 'user_listed_count', 'user_favourites_count', 'user_statuses_count'],
						converters={'tweet_text': convertText, 'tweet_favorite_count': convertNum,
						'tweet_retweet_count': convertNum, 'user_description': convertText},
						dtype={'tweet_id': str, 'user_id': str}
						)


		# The empty DataFrame that you provide doesn't have the correct column names.
		# You don't provide any columns in your metadata, but your output does have them.
		# This is the source of your error.
		# The meta value should match the column names and dtypes of your expected output.
		# ... es por eso que se crean columnas vacías

		df[['day','time']] = ''
		
		df['clean_tweet'] = ''
		df['clean_tweet_lemma'] = ''
		df['clean_tweet_lemma_postags'] = ''

		df['clean_tweet_nostop'] = ''
		df['clean_tweet_nostop_lemma'] = ''
		df['clean_tweet_nostop_lemma_postags'] = ''

		df['clean_description'] = ''


		dask_dataframe = ddf.from_pandas(df, npartitions=n_partitions)

		print(df.shape)
		result = dask_dataframe.map_partitions(cleanProcessDataframe, meta=df)
		df = result.compute()

		# Organizar columnas
		cleanData = df[['tweet_id', 'day','time', 'tweet_lang',
						'tweet_text', 'clean_tweet',
						'clean_tweet_lemma', 'clean_tweet_lemma_postags',
 						'clean_tweet_nostop', 'clean_tweet_nostop_lemma', 'clean_tweet_nostop_lemma_postags',
						'tweet_favorited', 'tweet_retweeted',
						'tweet_favorite_count', 'tweet_retweet_count', 'tweet_hashtags',
						'tweet_user_mentions', 'tweet_source',
						'user_id', 'user_name', 'user_screen_name',
						'user_description', 'clean_description',
						'user_location',
						'user_followers_count', 'user_friends_count', 'user_listed_count', 'user_favourites_count', 'user_statuses_count']]


		cleanData = cleanData[cleanData['clean_tweet'] != ''] # más rapido que remove

		print(cleanData.shape)

		# guardar en la nueva ubicacion
		cleanData.to_csv(os.path.join(cleanUsersDirectory, f'user_{user}'), index=False)

		end_time = time.time()
		print(f'{(end_time - start_time) / 60.0}')

		count += 1
		if count % 20 == 0:
			print(f'Processing {count}, {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}')
