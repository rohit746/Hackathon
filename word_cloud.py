import wordcloud
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('./resources/train.csv')

# Extract the text column from the CSV file
text = ' '.join(data['tweets'].astype(str))

# Define the words to be ignored
stop_words = ['â','¢','‚','¬','Â','¦','’',"It's",'Ã','..','Å'] 

# Load the image to be used as the mask
twitter_mask = np.array(Image.open("images.png").convert("RGB"))

# Create the word cloud object
wc = wordcloud.WordCloud(background_color="white", mask=twitter_mask, contour_width=3, contour_color='steelblue', stopwords=stop_words)

# Generate the word cloud from the text
wc.generate(text)

# Display the word cloud
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()