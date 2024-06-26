import pandas as pd
import numpy as np
import nltk
from textblob import TextBlob
import emoji
import warnings; warnings.simplefilter('ignore')
from transformers import pipeline
from langdetect import detect
from googletrans import Translator

df_emoji = pd.read_csv("Emoji_Sentiment_Data.csv",usecols = ['Emoji', 'Negative', 'Neutral', 'Positive'])
df_emoji.head()

df_emoji.Emoji.values

polarity_list = []
for index, row in df_emoji.iterrows():
    # initial polarity is negative
    polarity = 0
    # positive if positive value is greater than negative value
    arg_1 = row['Positive'] > row['Negative']
    # positive if neutral value is odd and positive and negative value are equal
    arg_2 = row['Positive'] == row['Negative'] and row['Neutral'] % 2 != 0

    # positive if either of the two arguments are true
    if arg_1 or arg_2:
        polarity = 1
    polarity_list.append(polarity)

# create new emoji dataset
new_df_emoji = pd.DataFrame(polarity_list, columns=['sentiment'])
new_df_emoji['emoji'] = df_emoji['Emoji'].values
new_df_emoji.head()

# print out the emoticons and sentiment values
e_c, p = 0, 0
for index, row in new_df_emoji.iterrows():
    print(f"{row['emoji']} = {row['sentiment']}")
    p += 1 if row['sentiment'] else 0
    e_c += 1

print(f'Total Positive Emojis are ({p}:{e_c}) or {round(p / e_c * 100)}%')

text = "i ❤ sentiments #goodlife @today"
def extract_text_and_emoji(text = text):
    global allchars, emoji_list
    # remove all tagging and links, not need for sentiments
    remove_keys = ('@', 'http://','https://', '&', '#')
    clean_text = ' '.join(txt for txt in text.split() if not txt.startswith(remove_keys))
    print(clean_text)

    # setup the input, get the characters and the emoji lists
    allchars = [str for str in text]
    # Use emoji.EMOJI_DATA
    emoji_list = [c for c in allchars if c in emoji.EMOJI_DATA]    
    clean_text = ' '.join([str for str in clean_text.split() if not any(i in str for i in emoji_list)])    
    clean_emoji = ''.join([str for str in text.split() if any(i in str for i in emoji_list)])
    return (clean_text, clean_emoji)

allchars, emoji_list = 0, 0
(ct, ce) = extract_text_and_emoji()
print('\nAll Char:', allchars)
print('\nAll Emoji:',emoji_list)
print('\n', ct)
print('\n',ce)


def detect_and_translate2(text):
  translator = Translator()
  lang = translator.detect(text).lang
  trans = translator.translate(text, src=lang, dest='en')
  return trans.text

# Load pre-trained sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

def get_sentiment(s_input='i hate sentiment analysis'):
    # Perform sentiment analysis
    results = sentiment_pipeline(s_input)
    # Get the sentiment prediction
    pred_senti = results[0]['label']
    if pred_senti == 'POSITIVE':
        pc = 1
    else:
        pc = 0
    return pc
# Test the model function
print(get_sentiment())

def get_emoji_sentiment(emoji_ls = '❤❤', emoji_df = new_df_emoji):
    emoji_val_ls = []
    for e in emoji_ls:
        get_emo_senti = [row['sentiment'] for index, row in emoji_df.iterrows() if row['emoji'] == e]
        if get_emo_senti:  # Check if the list is not empty
            emoji_val_ls.append(get_emo_senti[0])
        else:
            emoji_val_ls.append(0)  # Or any default value you prefer
    return emoji_val_ls
ges = get_emoji_sentiment()
print('Sentiment value of each emoji:',ges)

"""### Building the sentiment analysis"""

def analyze_emotion(text):
    # Create a TextBlob object
    blob = TextBlob(text)
    # Get sentiment scores
    sentiment = blob.sentiment
    # Determine emotion based on sentiment scores
    if sentiment.polarity > 0:
        if sentiment.subjectivity > 0.5:
            return "joy"  # Happy
        else:
            return "surprise"  # Surprise
    elif sentiment.polarity < 0:
        if sentiment.subjectivity > 0.5:
            return "anger"  # Anger
        else:
            return "sadness"  # Sad
    else:
        return "neutral"  # Neutral

tweet = "Had a terrible experience at the restaurant tonight. Never going back."
emotion = analyze_emotion(tweet)
print(f"Emotion: {emotion}")

def get_text_emoji_sentiment(input_test = 'i hate 😒 sentiment analysis'):
    # separate text and emoji
    (ext_text, ext_emoji) = extract_text_and_emoji(input_test)
    print(f'Extracted: "{ext_text}" , {ext_emoji}')
    #translate
    ttext=""
    lang = detect(ext_text)
    if lang!="en":
      ext_text= detect_and_translate2(ext_text)
      print(f'Translated: "{ext_text}"')
    # get text sentiment
    senti_text = get_sentiment(ext_text)
    print(f'Text value: {senti_text}')
    # get emoji sentiment
    senti_emoji_value = sum(get_emoji_sentiment(ext_emoji, new_df_emoji))
    print_emo_val_avg = 0 if len(ext_emoji) == 0 else senti_emoji_value/len(ext_emoji)
    print(f'Emoji average value: {print_emo_val_avg}')
    # avg the sentiment of emojis and text
    senti_avg = (senti_emoji_value + senti_text) / (len(ext_emoji) + 1)
    print(f'Average value: {senti_avg}')
    # set value of avg sentiment to either pos or neg
    senti_truth = "Positive" if senti_avg >= 0.5 else "Negative"
    emtext = analyze_emotion(ext_text)
    print(f'Text Emotion: {emtext}')
    return senti_truth
print(get_text_emoji_sentiment())



