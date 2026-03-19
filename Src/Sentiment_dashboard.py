import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob  # TextBlob is a simple sentiment analysis tool

# Load the Data
data = pd.read_csv(r"C:/Users/dp439_ykr3dmm/OneDrive/Desktop/New Data.csv") #Paste the file path here

# Display the dataset
print(data)

# Function to analyze sentiment
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Apply sentiment analysis to the dataset
data['Sentiment'] = data['text'].apply(analyze_sentiment)

# Categorize sentiment
data['Sentiment_Label'] = np.where(data['Sentiment'] > 0, 'Positive',
                                   np.where(data['Sentiment'] < 0, 'Negative', 'Neutral'))
sentiment_summary = data['Sentiment_Label'].value_counts()

print(sentiment_summary)

# Pie chart for sentiment distribution
plt.figure(figsize=(8, 6))
colors = ['lightblue', 'lightcoral', 'lightgreen']
sentiment_summary.plot(kind='pie', autopct='%1.1f%%', colors=colors)
plt.title('Sentiment Distribution')
plt.ylabel('')  # Hides the y-label
plt.show()

# Histogram of sentiment scores
plt.figure(figsize=(8, 6))
plt.hist(data['Sentiment'], bins=20, color='purple', edgecolor='black')
plt.title('Sentiment Polarity Scores')
plt.xlabel('Polarity Score')
plt.ylabel('Frequency')
plt.show()

# Function for user input sentiment analysis
def analyze_user_input():
    print("\nSentiment Analysis for User Input")
    user_text = input("Enter text for sentiment analysis: ")
    if not user_text.strip():
        print("No input provided. Please try again.")
        return
    polarity = analyze_sentiment(user_text)
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    print(f"\nAnalyzed Sentiment:\nText: {user_text}\nPolarity Score: {polarity:.2f}\nSentiment: {sentiment}")
    
    # Ask user if they want to visualize the sentiment results
    visualize_choice = input("\nWould you like to visualize the sentiment results? (yes/no): ").strip().lower()
    if visualize_choice == 'yes':
        visualize_sentiment([polarity])

# Function to visualize sentiment results (Pie chart or Histogram)
def visualize_sentiment(polarities):
    # Categorize sentiment based on polarity
    sentiment_labels = ['Positive' if p > 0 else 'Negative' if p < 0 else 'Neutral' for p in polarities]

    # Pie chart for sentiment distribution
    plt.figure(figsize=(8, 6))
    sentiment_summary = pd.Series(sentiment_labels).value_counts()
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    sentiment_summary.plot(kind='pie', autopct='%1.1f%%', colors=colors)
    plt.title('Sentiment Distribution')
    plt.ylabel('')  # Hides the y-label
    plt.show()

    # Histogram of sentiment scores
    plt.figure(figsize=(8, 6))
    plt.hist(polarities, bins=20, color='purple', edgecolor='black')
    plt.title('Sentiment Polarity Scores')
    plt.xlabel('Polarity Score')
    plt.ylabel('Frequency')
    plt.show()

# Prompt user for sentiment analysis
while True:
    analyze_user_input()
    again = input("\nDo you want to analyze another text? (yes/no): ").strip().lower()
    if again != 'yes':
        print("Exiting sentiment analysis tool. Goodbye!")
        break
