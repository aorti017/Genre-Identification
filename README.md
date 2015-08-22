Genre Identification
==========

Overview
---------
This is a personal project I have been working on to help facilitate learning how to use the [scikit-learn](http://scikit-learn.org/stable/) library as well as some machine learning algorithms and principles. 
The program uses various classification methods, along with some preprocessing, to determine if a song belongs to either the Hip Hop or Country genre based on the song's lyrics.
The training data goes through preprocessing, stemming and then the removal of stop words, before it is used to train the classifier. 
After the classifier is trained, the testing data goes through the same preprocessing and is then used to obtain predictions from the various classifiers.
The training and testing data can be found in ```./res/```.   


Notes
---------
* Before tuning the classifiers, the bag of words model gave a higher accuracy score then the tf-idf model. After tuning both models gave approximately the same accuracy score.
* I found that the Support Vector Classifier with a linear kernel gave the highest accuracy score. 
* Any parameter not passed to the classifier did not substantially affect the classifiers accuracy score. 


To Do
---------
* Add remaining testing and training data.
* Introduce another music genre to the training and testing dataset. 
