import pandas as pd
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

arts_list = db['artist'].unique()

artist = input('Enter the name of the artist: ')
songs = db[db.artist == artist.lower()]
while songs.empty:
    artist = input('No artist found, please enter another name: ')
    songs = db[db.artist == artist.lower()]

sentiment = []
pos_songs = ''
neg_songs = ''
for index, row in songs.iterrows():
    song = row['song']
    year = row['year']
    lyrics = row['lyrics']
    lines = lyrics.splitlines()
    song_sentiment = 0
    for sent in lines:
        sent_score = analyzer.polarity_scores(sent)
        song_sentiment += sent_score.get('compound')
    sentiment.append(song_sentiment)
    if song_sentiment > 0:
        pos_songs += lyrics
    if song_sentiment < 0:
        neg_songs += lyrics
songs['sentiment'] = sentiment

songs = songs.sort_values('sentiment', ascending=False)
pos_lyrics = ''
i = 0
for index, row in songs[:15].iterrows():
    if i == 0:
        print("The most positive song of {0} is {1}".format(artist, row['song']))
    pos_lyrics = row['lyrics']
    i += 1
pos_lines = str(pos_lyrics).splitlines()
pos_text = "".join(pos_lines)
d = path.dirname('Wordclouds.py')
mask = np.array(Image.open(path.join(d, "music.png")))
image_colors = ImageColorGenerator(mask)
stopwords = set(STOPWORDS)
stopwords.add("ain't")
stopwords.add("got")
stopwords.add("Chorus")
wc = WordCloud(background_color="white", mask=mask, stopwords=stopwords)
wc.generate(pos_text)
plt.imshow(wc, interpolation='bilinear')
plt.title('Positive songs of ' + artist_2)
plt.axis("off")
wc.to_file('positive.png')

songs = songs.sort_values('sentiment', ascending=True)
neg_lyrics = ''
i = 0
for index, row in songs[:15].iterrows():
    if i == 0:
        print("The most negative song of {0} is {1}".format(artist, row['song']))
    neg_lyrics = row['lyrics']
    i += 1
neg_lines = str(neg_lyrics).splitlines()
neg_text = "".join(neg_lines)
d = path.dirname('Wordclouds.py')
mask = np.array(Image.open(path.join(d, "music.png")))
image_colors = ImageColorGenerator(mask)
stopwords = set(STOPWORDS)
stopwords.add("ain't")
stopwords.add("got")
stopwords.add("Chorus")
wc = WordCloud(background_color="white", mask=mask, stopwords=stopwords)
wc.generate(neg_text)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title('Negative songs of ' + artist_2)
wc.to_file('negative.png')

print("The wordclouds are printed of the most frequent words of the 15 most positive and negative songs.")
