# 1. Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 2. Load your social media dataset (CSV with a 'text' column)
df = pd.read_csv("twitter_training.csv")  # Replace with your file path
print(df.head())

# 3. Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# 4. Define function to get sentiment category
def get_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# 5. Analyze sentiment scores
df['sentiment_score'] = df['text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
df['sentiment'] = df['sentiment_score'].apply(get_sentiment)

# 6. Visualize sentiment distribution
plt.figure(figsize=(8,6))
sns.countplot(data=df, x='sentiment', order=['Positive', 'Neutral', 'Negative'])
plt.title('Sentiment Distribution')
plt.show()

# 7. (Optional) Time-based sentiment trend if you have timestamp column
# Convert timestamp to datetime if necessary
# df['timestamp'] = pd.to_datetime(df['timestamp'])
# df.set_index('timestamp', inplace=True)
# df.resample('D')['sentiment_score'].mean().plot()
# plt.title('Sentiment Trend Over Time')
# plt.show()
