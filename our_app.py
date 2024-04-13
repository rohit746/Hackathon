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

# Data dependencies
import pandas as pd
# cleaning text
import re
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 


# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "Modelling"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
		    tweet = tweet.lower()
		    tweet = re.sub(r"\W", " ", tweet) # remove usernames
		    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
		    tweet = word_tokenize(tweet)
		    stopwords_list = set(stopwords.words('english') + list(punctuation))
		    tweets = [word for word in tweet if word not in stopwords_list]
		    tweet_text = " ".join(tweet)
		    predictor = joblib.load(open(os.path.join("resources/svc.pkl"),"rb"))
		    prediction = predictor.predict(tweets)		    
		    st.success("Text Categorized as: {}".format(prediction))

		if st.button("Button"):
			st.success("We can add text here")
		if st.button("Random new button"):
			st.success("Yeah! it worked!")


	# add EDA
	# add real world research
	# add choose a model button
	
	if selection == "Modelling":
		st.info("Model 1")
		st.subheader("Models")
		pic1 = {"Logistic regression": "https://drive.google.com/file/d/1wgWgT9wribP8Oa2Vxs_hkTROzGAIrbQF/view?usp=sharing"
		, "Linear SVC": "https://drive.google.com/file/d/144fovoeaSTs9Q-4hT_44j_qv3XzvsF6m/view?usp=sharing"}
		pic = st.selectbox("model choices",list(pic1.keys()), 0)
		st.image(pic1[pic], use_column_width=True,caption=pic1[pic])
		


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
