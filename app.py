"""

    Simple Streamlit webserver application for serving developed classification
	models.
    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.
	For further help with the Streamlit framework, see:
	https://docs.streamlit.io/en/latest/
"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image

# Streamlit dependencies
import streamlit as st
import joblib,os

# general
import numpy as np 
import pandas as pd
import dill as pickle

# text preprocessing
import re
from string import punctuation
import nltk
nltk.download(['stopwords','punkt'])
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

# models
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# metrics
from sklearn.metrics import classification_report

# metrics
from sklearn.metrics import classification_report

# function to preprocess tweets
def preprocess(tweet):
  tweet = tweet.lower() # converting everything to lower case
  tweet = re.sub(r"\W", " ", tweet)
  tweet = re.sub(r'#([^\s]+)', r'\1', tweet) 
  tweet = word_tokenize(tweet) # tokenizing the tweets so that they may be used in modeling and/or NLP
  stopwords_list = set(stopwords.words('english') + list(punctuation)) # cresting a list containing punctuation and stop words
  tweets = [word for word in tweet if word not in stopwords_list] # iterating over the list and saving the output into a list 
  return " ".join(tweet)

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.write('-----------------------------------------------')
	st.title("Tweet Classifer")
	st.subheader("Classifing tweets towards their belief in Climate Change")
	from PIL import Image
	logo = Image.open('resources/imgs/twitter_logo.jpg')
	st.image(logo, width = 100)

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home", "About","Data Cleaning","Modelling"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the predication page
	if selection == "Home":
		# pickle preprocessing function
		process_path = "resources/process.pkl"
		with open(process_path,'rb') as file:
			process = pickle.load(file)

		# loading linear SVC model
		model_load_path = "resources/svc.pkl"    
		with open(model_load_path,'rb') as file:
			svc = pickle.load(file)

		# loading logistic regression model
		model_load_path = "resources/LR.pkl"    
		with open(model_load_path,'rb') as file:
			lr = pickle.load(file)

		# loading grid search best fit model
		model_load_path = "resources/linear_SVC.pkl"    
		with open(model_load_path,'rb') as file:
			grid = pickle.load(file)

		result = {-1:'Anti Climate Change',
				0: 'Neutral towards Climate Change',
				1: 'Pro Climate Change',
				2: 'News'}
		st.write('-----------------------------------------------')
		st.info("See the **About** page for more information and data analysis")

		# Creating a text box for user input
		st.markdown("## **Enter Tweet Below**")
		tweet_text = st.text_area("","Type Here")
		tweet_processed = process(tweet_text)

		if st.button("Classify Linear SVC model"):
			tweet_pred = svc.predict([tweet_processed])
			print("predicted",result[int(tweet_pred)])
			st.success("Tweet classified as: {}".format(result[int(tweet_pred)]))

		if st.button("Classify SVC (gridsearch) model"):
			tweet_pred = grid.predict([tweet_processed])
			print("predicted",result[int(tweet_pred)])
			st.success("Tweet classified as: {}".format(result[int(tweet_pred)]))

		if st.button("Classify LR model"):
			tweet_pred = lr.predict([tweet_processed])
			print("predicted",result[int(tweet_pred)])
			st.success("Tweet classified as: {}".format(result[int(tweet_pred)]))

	# Building out the "Information" page
	if selection == "About":
		# Title
		st.title('About')
		st.write('-----------------------------------------------')
		# st.info("General Information")

		# Intro
		st.markdown('## Introduction')
		st.info("""Climate change has been a trending topic ever since
				 Al Gore received a Nobel Peace Prize for his campaign in 2007.
				The topic has become a controversial subject on twitter where some 
				twitter users feel very strongly that climate change is not real 
				and is part of a conspiracy theory. To add fire to the situation, 
				American President, Donald Trump, claimed that climate change is a 
				Chinese-funded conspiracy. As a result, some twitter users
				started tweeting that Climate Change is not real and trying to
				follow tweets about climate change suddenly required a degree in politics.""")

		# EDA
		st.markdown('## Exploratory Data Analysis')
		st.subheader('Most tweeted hashtag')
		st.info("""\n anti : #MAGA (11) 
				\n neutral : #climate (16)
				\n pro : #climate : (130)
				\n news : #climate : (130)
				""")
		st.subheader('Most tweeted username')
		st.info("""\n anti : @realDonaldTrump (71)
				\n neutral : @StephenSchlegel (307)
				\n pro : @realDonaldTrump (31)
				\n news : @thehill (77)
				""")

		# Adding word clouds
		st.markdown('### Word Clouds')
		st.info("Words Clouds can give an indication of the frequency of words in the data.")

		if st.button("Word Cloud for Anti Tweets"):
			from PIL import Image
			anti = Image.open('resources/imgs/wordcloud_anti.PNG')
			st.image(anti, width = 650)

		if st.button("Word Cloud for Pro Tweets"):
			from PIL import Image
			pro = Image.open('resources/imgs/wordcloud_pro.PNG')
			st.image(pro, width = 650)

		if st.button("Word Cloud for Neutral Tweets"):
			from PIL import Image
			pro = Image.open('resources/imgs/wordcloud_neutral.PNG')
			st.image(pro, width = 650)

		if st.button("Word Cloud for News Tweets"):
			from PIL import Image
			pro = Image.open('resources/imgs/wordcloud_fact.PNG')
			st.image(pro, width = 650)

		st.subheader('Interesting insights about the training data: ')
		st.markdown("""For Anti-climate change tweets:""")
		st.info("""\n - The word **science** occurs alot as many people against climate change
				use scientific facts to express their view.
				\n - **Steves Goddard** pops up and he is a social media influencer of
				 scientific background but denies science about climate change. 
				He would often produce a graphic that shows that climate change is not real.
				\n - Politicians referenced include **Al Gore**, **Obama** and **Donald Trump**.
				\n - The **#MAGA** is a hasthtag that means Make America Great Again and
				 is associated with Americans who side with Trump.""")
		st.markdown("""For Pro-climate change tweets:""")
		st.info("""\n - **Steven Schlegel** occurs often as there is a science report
				from 2006 detailing the predictions of climate change in the future published by Schlegel.
				\n - The hashtag **#ImVotingBecause** occured 62 times which might imply that climate change
				can influence voters decision who to vote for.
				\n - The word husband pops up for some reason.
				\n - Politicians referenced include **Sen Sanders** and **Donald Trump**""")
		st.markdown("""For Neutral tweets:""")
		st.info("""\n - The word **journalist** pops up.
	\n - Places referrenced are **America** and **Paris**. The paris word might reference to *The Paris Agreement* 
	\n - **Chelsea Clinton** is referrenced, she is an American author and global health advocate.
	\n - Politicians referenced include **Sen Sanders** and **Donald Trump**
	\n - Celebrities referenced include **Leonardo Dicaprio**.
	\n - Strong emotional words include please, action, fuck and responsible""")
		st.markdown("""For News tweets:""")
		st.info("""\n - The word **EPA** pops up, this is the United States Environmental Protection Agency.
	\n - News outlets referenced include **CNN**, **Guardian**, **Time**.
	\n - **Scott Prutt** is mentioned, he is the Administrator of the United States Environmental Protection Agency.
	\n - The word **independent study** pops up.
	\n - Che **white house** and **Trump** is mentioned.
	\n - Countries that pop up include **US** and **China**.""")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'):
			# Load your raw data
			raw = pd.read_csv("climate-change-belief-analysis/train.csv")
			st.write(raw[['sentiment', 'message']])

	if selection == "Data Cleaning":
		st.write('-----------------------------------------------')
		st.write("""Cleaning tweets is vitally important for an accurate model. 
				\n Try our tweet cleaner below that will show you how we cleaned our tweets.""")
		st.markdown("## **Enter Tweet Below**")
		# pickle preprocessing function
		process_path = "resources/process.pkl"
		with open(process_path,'rb') as file:
			process = pickle.load(file)
		tweet_text = st.text_area("","Type Here")
		tweet_processed = process(tweet_text)
		if st.button("Clean Tweet"):
			st.success("Tweet cleaned as: {}".format(tweet_processed))
		st.subheader("Before preprocessing we need to ask ourseleves the following questions about this data:")
		st.write("Does URL have impact on the tweet sentiment?")
		st.info("""So many twitter users retweet URL's to substantiate their view, therefore by removing 
				the URL the sentiment value might be reduced.""")
		st.write("Does retweet have any impact on tweet sentiment?")
		st.info("""\n Number of Original Tweets: 6133
				\n Number of Retweets: 9687
				\n Ratio of Orignal Tweets to retweets: 0.63
				\n Because the retweet ratio is more than 0.5, it would be better to keep the retweet as RT.""")
		st.write("Does removing hashtags remove sentiment?")
		st.info("""Hashtags can link one tweet to another, therefore it would be better to keep the hashtags.""")
		st.write("Contractions are a problem. how will removing them effect our model?")
		st.info("""Twitter users use slang to communicate their views and many tweets contain contractions. 
				Contractions does make the modelling process more challenging as [don't] needs to mean the same
				as [do not]. 
				\n Using the TwitterTokenizer module helps to keep contractions in their own form.""")
		st.write("What about emojis?")
		st.info("""Twitter users love using emojis as a way of expressing their emotions. The training data did not
				have a large sample of emojis so keeping the emojis in their raw form would improve the sentiment
				of our models.""")

	if selection == "Modelling":
		st.write('-----------------------------------------------')
		from PIL import Image
		st.subheader("Data used in our models")
		st.write("""The data we used in our models was unbalanced.
					\n This posed a challenge to find a accurate model.""")
		st.info("""-1 : Anti
				\n 0 : Neutral
				\n 1 : Pro
				\n 2 : News """)
		i = Image.open("resources/imgs/i.PNG")
		st.image(i, caption="Distribution of data", use_column_width=True)
		st.text("     ")
		st.write("Logistic Regression Model")
		st.info("""Logistic regression is a supervised learning classification algorithm 
				used to predict the probability of a target variable.
				This model works best on binary data classification but almost performs well in our data
				even though it badly predicts some classes such as the recall of 0 and -1. 
				\n The overall accuracy is decent and it also does quite well on unseen data.""")
		from PIL import Image
		image1 = Image.open('resources/imgs/logistic.PNG')
		st.image(image1, caption="Logistic Regression")
		st.text("                                                                        ")
		st.write("Linear SVC Model")
		st.info("""Linear Support Vector Machine is machine learning algorithm for solving 
				multiclass classification problems. 
				\n It gives a better score than Logistics regression""")
		st.text("                                             ")
		from PIL import Image
		image2 = Image.open('resources/imgs/linear svc.PNG')
		st.image(image2, caption="Linear SVC")
		st.write("Grid search on SVC")
		image3 = Image.open("resources/imgs/grid.PNG")
		st.info("""Grid search is the process of scanning the models to configure optimal parameters
				 for a model. In our Support Vector Machine model, we searched for the best parameters
				bewteen C and gamma parameters to get the past fit. Our grid search gave these parameters
				as the best parameters: Using C=1 and gamma=0.01.
				\n Grid search is computationally expensive but gave an accuracy 0.75""")
		st.text("     						")
		from PIL import Image
		st.image(image3, caption="Grid search")
		st.subheader("Insights on how models perform on unseen data")
		st.info("""Having unbalanced data made it difficult to find the most accurate model. 
		Even using methods like upsampling and downsampling did not improve models. We ended 
		up having overfit models which perform very poorly on unseen data. 
		Logistic model performs well on binary labels which is why it did not, perform very well
		in our data which has 4 labels. \n  Grid search ultimately gave the best score.
		It uses all possible combinations and outputs the results for each combination to give the best
		accuracy.""")


		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    print('loading pickle file')
    model_load_path = "resources/linear_SVC.pkl"
    
    with open(model_load_path,'rb') as file:
        unpickled_model = pickle.load(file)

    print("model successfully pickled")
    main()
