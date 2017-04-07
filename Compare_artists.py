import pandas as pd
import os
import nltk
from nltk import sent_tokenize, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

db = pd.read_csv('lyrics.csv')
db = db.drop(['genre'], axis=1)
db = db.drop(['index'], axis=1)
db = db[db.lyrics.notnull()]

#ask user for two artists
artist_1 = input('Enter the name of the first artist: ')
songs_1 = db[db.artist == artist_1.lower()]
while songs_1.empty:
    artist_1 = input('No artist found, please enter another name for the first artist: ')
    songs_1 = db[db.artist == artist_1.lower()]
artist_2 = input('Enter the name of the second artist: ')
songs_2 = db[db.artist == artist_2.lower()]
while songs_2.empty:
    artist_2 = input('No artist found, please enter another name for the second artist: ')
    songs_2 = db[db.artist == artist_2.lower()]

#compute sentiment for every song of artist 1
sentiment_1 = []
pos_songs_1 = ''
neg_songs_1 = ''
artist_sentiment_1 = 0
for index, row in songs_1.iterrows():
    song = row['song']
    year = row['year']
    lyrics = row['lyrics']
    lines = lyrics.splitlines()
    song_sentiment = 0
    i = 0
    for sent in lines:
        sent_score = analyzer.polarity_scores(sent)
        song_sentiment += sent_score.get('compound')
        i+=1
    sentiment_1.append(song_sentiment/i)
    artist_sentiment_1 += song_sentiment
    if song_sentiment > 0:
        pos_songs_1 += lyrics
    if song_sentiment < 0:
        neg_songs_1 += lyrics
songs_1['sentiment'] = sentiment_1
songs = songs_1.sort('sentiment')

#compute sentiment for every song of artist 2
sentiment_2 = []
pos_songs_2 = ''
neg_songs_2 = ''
artist_sentiment_2 = 0
for index, row in songs_2.iterrows():
    song = row['song']
    year = row['year']
    lyrics = row['lyrics']
    lines = lyrics.splitlines()
    song_sentiment = 0
    i = 0
    for sent in lines:
        sent_score = analyzer.polarity_scores(sent)
        song_sentiment += sent_score.get('compound')
        i += 1
    sentiment_2.append(song_sentiment/i)
    artist_sentiment_2 += song_sentiment
    if song_sentiment > 0:
        pos_songs_2 += lyrics
    if song_sentiment < 0:
        neg_songs_2 += lyrics
songs_2['sentiment'] = sentiment_2
songs = songs_2.sort('sentiment')

pos_lines_1 = str(pos_songs_1).splitlines()
pos_text_1 = "".join(pos_lines_1)

neg_lines_1 = str(neg_songs_1).splitlines()
neg_text_1 = "".join(neg_lines_1)

if artist_sentiment_1 >= artist_sentiment_2:
    pos_artist = artist_1
    neg_artist = artist_2
elif artist_sentiment_1 < artist_sentiment_2:
    pos_artist = artist_2
    neg_artist = artist_1

print('{0} songs are more positive than {1} songs.'.format(pos_artist, neg_artist))

d = path.dirname('Project_wordclouds.py')
mask = np.array(Image.open(path.join(d, "music.png")))
image_colors = ImageColorGenerator(mask)
stopwords = set(STOPWORDS)
stopwords.add("ain't")
stopwords.add("got")
stopwords.add("Chorus")
wc = WordCloud(background_color="white", mask=mask, stopwords=stopwords)
wc.generate(pos_text_1)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")

d = path.dirname('Project_wordclouds.py')
mask = np.array(Image.open(path.join(d, "music.png")))
image_colors = ImageColorGenerator(mask)
stopwords = set(STOPWORDS)
stopwords.add("ain't")
stopwords.add("got")
stopwords.add("Chorus")
wc = WordCloud(background_color="white", mask=mask, stopwords=stopwords)
wc.generate(neg_text_1)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
