# Streamlit dependencies
import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
tweet_cv = joblib.load(news_vectorizer)  # Loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# Load your test data
test_data = pd.read_csv("resources/train.csv")

# Split the test data into features and labels
X_test = test_data['tweets']  # Replace 'text_column' with the name of your text column
y_test = test_data['sentiment']  # Replace 'label_column' with the name of your label column

# Vectorize the test features
X_test_vectorized = tweet_cv.transform(X_test)

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Tweet Classifer")
    st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # Data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']])  # Will write the df to the page

    # Building out the predication page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

        # Load your .pkl file with the model of your choice
        predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"), "rb"))

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))

    # Print the accuracy of the model
    accuracy = predictor.score(X_test_vectorized, y_test)
    st.write(f"Accuracy of the model: {accuracy:.2f}")

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()