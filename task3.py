import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a pandas DataFrame
data = pd.read_csv('./resources/train.csv')

# Create a function to extract hashtags from the tweet text
def extract_hashtags(text):
    import re
    hashtags = re.findall(r'#\w+', text)
    return hashtags

# Apply the extract_hashtags function to the tweet text column
data['hashtags'] = data['tweets'].apply(extract_hashtags)

# Flatten the list of lists in the 'hashtags' column
flat_hashtags = [item for sublist in data['hashtags'] for item in sublist]

# Create a new DataFrame with sentiment categories and hashtags
hashtags_by_sentiment = pd.DataFrame({'sentiment': data['sentiment'].repeat(data['hashtags'].str.len()),
                                      'hashtags': flat_hashtags})

# Count the occurrences of hashtags for each sentiment category
hashtag_counts = hashtags_by_sentiment.groupby(['sentiment', 'hashtags']).size().reset_index(name='count')

# Function to visualize and identify the top 3 hashtags for each sentiment category
def visualize_top_hashtags(sentiment_category):
    category_hashtags = hashtag_counts[hashtag_counts['sentiment'] == sentiment_category]
    top_hashtags = category_hashtags.nlargest(3, 'count')
    
    plt.figure(figsize=(8, 6))
    plt.bar(top_hashtags['hashtags'], top_hashtags['count'])
    plt.xticks(rotation=45)
    plt.xlabel('Hashtags')
    plt.ylabel('Count')
    plt.title(f'Top 3 Hashtags for Sentiment Category: {sentiment_category}')
    plt.show()

# Call the visualize_top_hashtags function for each sentiment category
visualize_top_hashtags(2)  # News
visualize_top_hashtags(1)  # Pro
visualize_top_hashtags(0)  # Neutral
visualize_top_hashtags(-1) # Anti