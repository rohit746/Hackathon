import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Load the dataset
df = pd.read_csv('./resources/train.csv')

# Identify climate change denial tweets based on the label
denial_tweets = df[df['sentiment'] == -1]

# Preprocess the text data
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)      # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters except spaces
    text = re.sub(r'\d+', '', text)          # Remove digits
    text = text.lower()                      # Convert to lowercase
    return text

denial_tweets['clean_text'] = denial_tweets['tweets'].apply(preprocess_text)

# Analyze common themes and arguments
words = ' '.join(denial_tweets['clean_text']).split()
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

word_freq = Counter(filtered_words)
print(word_freq.most_common(20))

# Explore language and rhetoric
# You can further analyze sentiment or specific language used in denial tweets