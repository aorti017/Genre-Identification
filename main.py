import os, string, codecs
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

genreTrain = ["./res/train/hiphop", "./res/train/country"]

lyrics = []
genre = []

stemmer = SnowballStemmer("english")
for genrePath in genreTrain:
	for subdir, dirs, files in os.walk(genrePath):
		for f in files:
			temp = ""
			pth = os.path.join(subdir, f)
			fileLyrics = open(pth).read().decode('utf-8')
			fileLyrics = fileLyrics.replace(string.punctuation, "")
			for i in string.punctuation:
				fileLyrics = fileLyrics.replace(i, '')
			for i in fileLyrics.split(" "):
				temp += stemmer.stem(i) + " "
			lyrics.append(temp)
			if genrePath == genreTrain[0]:
				genre.append("HH")
			else:
				genre.append("CN")
				

sw = stopwords.words("english")

countVec = CountVectorizer(stop_words=sw)
trainCount = countVec.fit_transform(lyrics)

classifier = MultinomialNB(alpha=.5).fit(trainCount, genre)

#testCount = countVec.transform(testSet)
#predicted = classifier.predict(testCount)
#print "Accuracy: " + str(accuracy_score(predicted, predTest))
