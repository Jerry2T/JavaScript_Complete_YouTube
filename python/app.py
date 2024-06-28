import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import emoji
import warnings
from transformers import pipeline
from langdetect import detect
from googletrans import Translator
import fitz  # PyMuPDF
warnings.simplefilter('ignore')

# Load emoji sentiment data
df_emoji = pd.read_csv("Emoji_Sentiment_Data.csv", usecols=['Emoji', 'Negative', 'Neutral', 'Positive'])

# Process emoji sentiment data
polarity_list = []
for index, row in df_emoji.iterrows():
    polarity = 0
    arg_1 = row['Positive'] > row['Negative']
    arg_2 = row['Positive'] == row['Negative'] and row['Neutral'] % 2 != 0
    if arg_1 or arg_2:
        polarity = 1
    polarity_list.append(polarity)

new_df_emoji = pd.DataFrame(polarity_list, columns=['sentiment'])
new_df_emoji['emoji'] = df_emoji['Emoji'].values

# Function to extract text and emojis from input
def extract_text_and_emoji(text):
    remove_keys = ('@', 'http://', 'https://', '&', '#')
    clean_text = ' '.join(txt for txt in text.split() if not txt.startswith(remove_keys))
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.EMOJI_DATA]
    clean_text = ' '.join([str for str in clean_text.split() if not any(i in str for i in emoji_list)])
    clean_emoji = ''.join([str for str in text.split() if any(i in str for i in emoji_list)])
    return clean_text, clean_emoji

# Function to detect language and translate text to English
def detect_and_translate2(text):
    translator = Translator()
    lang = translator.detect(text).lang
    trans = translator.translate(text, src=lang, dest='en')
    return trans.text

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

# Function to get sentiment of text
def get_sentiment(s_input):
    results = sentiment_pipeline(s_input)
    pred_senti = results[0]['label']
    return 1 if pred_senti == 'POSITIVE' else 0

# Function to get sentiment of emojis
def get_emoji_sentiment(emoji_ls, emoji_df=new_df_emoji):
    emoji_val_ls = []
    for e in emoji_ls:
        get_emo_senti = [row['sentiment'] for index, row in emoji_df.iterrows() if row['emoji'] == e]
        emoji_val_ls.append(get_emo_senti[0] if get_emo_senti else 0)
    return emoji_val_ls

# Function to analyze emotion of text
def analyze_emotion(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    if sentiment.polarity > 0:
        if sentiment.subjectivity > 0.5:
            return "joy"
        else:
            return "surprise"
    elif sentiment.polarity < 0:
        if sentiment.subjectivity > 0.5:
            return "anger"
        else:
            return "sadness"
    else:
        return "neutral"

# Function to get sentiment and emotion of text and emojis
def get_text_emoji_sentiment(input_text):
    ext_text, ext_emoji = extract_text_and_emoji(input_text)
    
    if ext_text.strip() == "" and ext_emoji.strip() != "":
        # Only emojis are present
        senti_emoji_values = get_emoji_sentiment(ext_emoji, new_df_emoji)
        senti_emoji_value_sum = sum(senti_emoji_values)
        print_emo_val_avg = 0 if len(ext_emoji) == 0 else senti_emoji_value_sum / len(ext_emoji)
        senti_truth = "Positive" if print_emo_val_avg >= 0.5 else "Negative"
        emoji_sentiments = dict(zip(ext_emoji, senti_emoji_values))
        return senti_truth, "N/A", "N/A", emoji_sentiments
    else:
        lang = detect(ext_text)
        if lang != "en":
            ext_text = detect_and_translate2(ext_text)
        
        senti_text = get_sentiment(ext_text)
        senti_emoji_value = sum(get_emoji_sentiment(ext_emoji, new_df_emoji))
        print_emo_val_avg = 0 if len(ext_emoji) == 0 else senti_emoji_value / len(ext_emoji)
        senti_avg = (senti_emoji_value + senti_text) / (len(ext_emoji) + 1)
        senti_truth = "Positive" if senti_avg >= 0.5 else "Negative"
        emtext = analyze_emotion(ext_text)
        return senti_truth, emtext, ext_text, {}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Streamlit frontend
st.title('Text and Emoji Sentiment Analysis')
user_input = st.text_input("Enter tweet with emojis in your language:")

uploaded_pdf = st.file_uploader("Or upload a PDF file containing tweets:", type="pdf")
uploaded_csv = st.file_uploader("Or upload a CSV file containing tweets:", type="csv")

if user_input:
    sentiment, emotion, tran, emoji_sentiments = get_text_emoji_sentiment(user_input)
    
    if tran == "N/A":
        st.write(f"Only emojis detected: {user_input}")
        st.write(f"Emoji Sentiment: {sentiment}")
        st.write("Detailed Emoji Sentiments:")
        for emoji_char, senti_val in emoji_sentiments.items():
            senti_text = "Positive" if senti_val == 1 else "Negative"
            st.write(f"{emoji_char}: {senti_text}")
    else:
        st.write(f"Translated tweet: {tran}")
        st.write(f"Overall Sentiment: {sentiment}")
        st.write(f"Text Emotion: {emotion}")

elif uploaded_pdf:
    pdf_text = extract_text_from_pdf(uploaded_pdf)
    tweets = pdf_text.split('\n')
    
    for tweet in tweets:
        if tweet.strip():
            sentiment, emotion, tran, emoji_sentiments = get_text_emoji_sentiment(tweet)
            st.write(f"Tweet: {tweet}")
            
            if tran == "N/A":
                st.write(f"Only emojis detected")
                st.write(f"Emoji Sentiment: {sentiment}")
                st.write("Detailed Emoji Sentiments:")
                for emoji_char, senti_val in emoji_sentiments.items():
                    senti_text = "Positive" if senti_val == 1 else "Negative"
                    st.write(f"{emoji_char}: {senti_text}")
            else:
                st.write(f"Translated tweet: {tran}")
                st.write(f"Overall Sentiment: {sentiment}")
                st.write(f"Text Emotion: {emotion}")
            st.write("\n")

elif uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    if 'tweet' in df.columns:
        for tweet in df['tweet']:
            if tweet.strip():
                sentiment, emotion, tran, emoji_sentiments = get_text_emoji_sentiment(tweet)
                st.write(f"Tweet: {tweet}")
                
                if tran == "N/A":
                    st.write(f"Only emojis detected")
                    st.write(f"Emoji Sentiment: {sentiment}")
                    st.write("Detailed Emoji Sentiments:")
                    for emoji_char, senti_val in emoji_sentiments.items():
                        senti_text = "Positive" if senti_val == 1 else "Negative"
                        st.write(f"{emoji_char}: {senti_text}")
                else:
                    st.write(f"Translated tweet: {tran}")
                    st.write(f"Overall Sentiment: {sentiment}")
                    st.write(f"Text Emotion: {emotion}")
                st.write("\n")
    else:
        st.write("CSV file does not contain 'tweet' column.")
