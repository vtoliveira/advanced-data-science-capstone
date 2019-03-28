# Advanced Data Science Capstone

This repository has a series of notebooks used in the course Advanced Data Science Capstone project where I developed models to work with sentiment analysis from Twitter US Airline Sentiment.

# Content

 - Initial Data Exploration: Notebook for exploratory analysis, checking features, missing values, graphics and insights for feature extraction.
 - Feature Creation (Bag of Words): Here I extract features based on classical approach for text mining. Polarity scores, negation, POS count, emoticons, etc. The data was saved as data_preprocessed.csv
 - Feature Creation (Word Embedding): Another approach is to create embeddings based on the given tweets. Here I use gensim, a python library to train embeddings from pure texts.
 - Model Definition (Classical Algorithms): Here I develop the pipeline to train classical models such as SVMs, Naive Bayes, and Random Forests. I also implement more text transformations such as CountVectorizer and TF-IDF and cross-validate the models with a variety of features.
 - Model Definition (Deep Learning): In this notebook I define a LSTM Network for text classification using pytorch. We also import GloVe pre-trained embeddings from twitter data found on kaggle. You can download it here: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/50350
 - Model Evaluation: Here I explore some metrics to check how our models are perform and do a brief discussion about the best models.
 - Sentiment Analysis: Here we have a demo where a few tweets are classified and we see our model for a non-technical public.
 
 # Summary
 
 My major contribution with this project was being able to produce diffent models in order to do a systematic comparison between classical machine learning techniques and feature extraction vs deep learning recurrent neural networks with twitter word embeddings.

Below we have the confustion matrix for the three best classical approaches (Random Forest, SVM and Naive Bayes). I used a tf-idf approach with SVM and Random Forest and a count vector with NB. The results were obtained after cross-validation for hyperparameter tuning. It is clear that SVM with bi-grams achieved the best results in this case.

![Sentiment Analysis](https://github.com/vtoliveira/advanced-data-science-capstone/blob/master/classical.png)

Next, we used word embeddings trained on twitter data, for 50, 100, and 200 dimensions and used a LSTM network to classify the tweets sentiments.

![Sentiment Analysis](https://github.com/vtoliveira/advanced-data-science-capstone/blob/master/lstm.png)

To finish, we had prediction, accuracy, f1 and recall scores for each model.

![Sentiment Analysis](https://github.com/vtoliveira/advanced-data-science-capstone/blob/master/lstm_scores.png)
![Sentiment Analysis](https://github.com/vtoliveira/advanced-data-science-capstone/blob/master/classical_scores.png)

We can see the LSTM performs better overall and that was our choosed model.

**Observation**: I would like to include a comment on training the LSTM Network. Even though it achieved a better result, as our test set is small and we have a considerable small sample regarding text classification, the best results were achieved after intense training and tuning, which can be impractical sometimes.

 

 # Video Presentation
 
 Also, one of the tasks was to produce a video presentation about my project. Here you can see all the details about the product developed, models definition, evaluation, etc.
 
 ![Sentiment Analysis](https://github.com/vtoliveira/advanced-data-science-capstone/blob/master/video.JPG)
 (https://www.youtube.com/watch?v=oLX73CMh8Fc&t=43s)
