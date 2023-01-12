### enviroment: experiments_env

import tweepy
import pandas as pd
import os

import time
from datetime import datetime

# ---------------------------------------------------------------
# Mental health keys
consumer_key = 'KEY'
consumer_secret = 'KEY SECRET'
access_token = 'TOKEN'
access_token_secret = 'TOKEN SECRET'

# ---------------------------------------------------------------
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True,
						wait_on_rate_limit_notify=True)


# ---------------------------------------------------------------
def getTweets(destinyDirectory, userID, numberTweets):

	try:
		cols_names = ['tweet_id', 'tweet_created_at', 'tweet_text', 'tweet_source', 'tweet_truncated', 'tweet_in_reply_to_status_id',
				'tweet_in_reply_to_user_id', 'tweet_in_reply_to_screen_name', 'tweet_retweet_count', 'tweet_favorite_count',
				'tweet_favorited', 'tweet_retweeted', 'tweet_lang', 'tweet_hashtags', 'tweet_symbols', 'tweet_user_mentions',
				'tweet_urls', 'user_id', 'user_name', 'user_screen_name', 'user_location', 'user_url', 'user_description',
				'user_verified', 'user_followers_count', 'user_friends_count', 'user_listed_count', 'user_favourites_count',
				'user_statuses_count', 'user_created_at', 'user_utc_offset', 'user_time_zone', 'user_geo_enabled', 'user_default_profile',
				'user_default_profile_image']

		# Obtener los primeros 3200 tweets sin retweets (o sea son menos de 3,200)
		tweets = tweepy.Cursor(api.user_timeline, id=userID, include_rts=False, tweet_mode='extended').items(numberTweets)

		# Iterar sobre los tweets encontrados
		tweet_list = [[tweet.id_str, tweet.created_at, tweet.full_text, tweet.source, tweet.truncated, tweet.in_reply_to_status_id_str,
						tweet.in_reply_to_user_id_str, tweet.in_reply_to_screen_name, tweet.retweet_count, tweet.favorite_count, tweet.favorited,
						tweet.retweeted, tweet.lang, tweet.entities['hashtags'], tweet.entities['symbols'], tweet.entities['user_mentions'],
						tweet.entities['urls'], tweet.user.id_str, tweet.user.name, tweet.user.screen_name, tweet.user.location, tweet.user.url,
						tweet.user.description, tweet.user.verified, tweet.user.followers_count, tweet.user.friends_count, tweet.user.listed_count,
						tweet.user.favourites_count, tweet.user.statuses_count, tweet.user.created_at, tweet.user.utc_offset, tweet.user.time_zone,
						tweet.user.geo_enabled, tweet.user.default_profile, tweet.user.default_profile_image] for tweet in tweets]

		df = pd.DataFrame(data=tweet_list, columns=cols_names)

		total_tweets = df.shape[0]
		if total_tweets > 0:
			user_name = df['user_screen_name'][0]
			print(f'User: {user_name}\t{userID}\t{total_tweets}')
			filename = f'{user_name}_{userID}.csv'
			df.to_csv(os.path.join(destinyDirectory, filename), index=False)
		else:
			print(f'User: {userID}\t{total_tweets}')
	
	except tweepy.error.TweepError as e:
		print(e) # Ej. Usuario no encontrado



# ---------------------------------------------------------------
print('Start...')
print(f'\n{datetime.today().strftime("%d-%m-%Y %H:%M:%S")}\n')

directory = r'Timeline_users'
numberTweets = 3200
users_retrieve_list = pd.read_csv(r'ids_search_timelines.csv', dtype={'user_id': str})

print(f'Total de usuarios = {users_retrieve_list.shape[0]}\n')

contador = 0
for user in users_retrieve_list['user_id'].values:

	getTweets(directory, user, numberTweets)

	contador += 1

	if contador % 10 == 0:
		print(f'\nProcesados... {contador}\t{datetime.today().strftime("%d-%m-%Y %H:%M:%S")}\n')
