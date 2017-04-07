import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

db = pd.read_csv('lyrics.csv')
db = db.drop(['genre'], axis=1)
db = db.drop(['index'], axis=1)
db = db[db.lyrics.notnull()]

#ask user for an artist
artist = input('Enter the name of the artist: ')
songs = db[db.artist == artist]
while songs.empty:
    artist = input('No artist found, please enter another name: ')
    songs = db[db.artist == artist.lower()]

#calculate sentiment for each artist's song
sentiment = []
for index, row in songs.iterrows():
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
    sentiment.append(song_sentiment/i)

songs['sentiment'] = sentiment #add column in database for sentiment
songs2 = songs.drop(['lyrics'], axis=1) #drop lyrics column to reduce size

#plot sentiment of songs over the years
years = list(songs2['year'].unique())
years.sort()
sp = songs.plot(x='year', y='sentiment', style='o')
sp.set_xlim([(years[0]-1), (years[-1]+1)])

#calculate average sentiments per year
new_table = []
average_sentiments = []
highest_sentiment = -100
lowest_sentiment = 100
for year in years:
    total_sentiment = 0
    counter = 0
    for index, row in songs2.iterrows():
        sentiment = row['sentiment']
        song_year = row['year']
        song = row['song']
        if song_year == year:
            total_sentiment += sentiment
            if sentiment > highest_sentiment: #retrieve most positive and negative song of the artist
                highest_sentiment = sentiment
                pos_song = song
            if sentiment < lowest_sentiment:
                lowest_sentiment = sentiment
                neg_song = song
            counter += 1
    average_sentiment = total_sentiment / counter
    average_sentiments.append(average_sentiment)
    new_table.append((year, average_sentiment))

x = years
y = average_sentiments
plt.plot(x, y)
plt.ylabel('Sentiment')
plt.title(artist)
plt.show()
plt.legend()
axes = plt.gca()
axes.set_xlim([(years[0]-1), (years[-1]+1)])
plt.to_file('graph.png')

print("{0}'s song sentiments are displayed in the plot. {0}'s song '{1}' is the most positive song and '{2}' the most negative one.".format(artist, pos_song, neg_song))
