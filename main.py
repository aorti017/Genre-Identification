#!/usr/bin/env

import os, string, codecs
from sklearn import svm, tree
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


def getPreAcc(trainCount, testCount, predTest, genre):
	clfMNB = MultinomialNB(alpha=.3).fit(trainCount, genre)
	predicted = clfMNB.predict(testCount)
	print "Multinomial Naive Bayes Accuracy: " + str(round(accuracy_score(predicted, predTest), 3))
	
	clfGNB = GaussianNB().fit(trainCount.toarray(), genre)
	predicted = clfGNB.predict(testCount.toarray())
	print "Gaussian Naive Bayes Accuracy: " + str(round(accuracy_score(predicted, predTest), 3))
	
	clfSVM = svm.SVC(kernel="linear").fit(trainCount, genre)
	predicted = clfSVM.predict(testCount)
	print "Support Vector Classifier Accuracy: " + str(round(accuracy_score(predicted, predTest), 3))
	
	clfDT = tree.DecisionTreeClassifier().fit(trainCount.toarray(), genre)
	predicted = clfDT.predict(testCount.toarray())
	print "Decision Tree Accuracy: " + str(round(accuracy_score(predicted, predTest), 3))

def main():
	genreTrain = ["./res/train/hiphop", "./res/train/country"]
	genreTest = ["./res/test/hiphop", "./res/test/country"]
	lyrics = []
	genre = []
	testSet = []
	predTest = []
	
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
	
	for genrePath in genreTest:
		for subdir, dirs, files in os.walk(genrePath):
			for f in files:
				#print f
				temp = ""
				pth = os.path.join(subdir, f)
				fileLyrics = open(pth).read().decode('utf-8')
				fileLyrics = fileLyrics.replace(string.punctuation, "")
				for i in string.punctuation:
					fileLyrics = fileLyrics.replace(i, '')
				for i in fileLyrics.split(" "):
					temp += stemmer.stem(i) + " "
				testSet.append(temp)
				if genrePath == genreTest[0]:
					predTest.append("HH")
				else:
					predTest.append("CN")
	
	sw = stopwords.words("english")
	
	countVec = CountVectorizer(stop_words=sw)
	trainCount = countVec.fit_transform(lyrics)
	testCount = countVec.transform(testSet)
	print "Using Count Vectorizer\n----------"
	getPreAcc(trainCount, testCount, predTest, genre)
	
	tfidfVec = TfidfTransformer().fit(trainCount)
	trainCount = tfidfVec.transform(trainCount)
	testCount = tfidfVec.transform(testCount)
	print "\nUsing TF-IDF Vectorizer\n----------"
	getPreAcc(trainCount, testCount, predTest, genre)



if __name__ == "__main__":
	main()
