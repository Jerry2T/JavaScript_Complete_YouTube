import pandas as pd
import numpy as np
from textblob import TextBlob
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.simplefilter('ignore')
from transformers import pipeline
from langdetect import detect
from googletrans import Translator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix

df_emoji = pd.read_csv("Emoji_Sentiment_Data.csv",usecols = ['Emoji', 'Negative', 'Neutral', 'Positive'])
df_emoji.head(3)

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
new_df_emoji.head(3)

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
    remove_keys = ('@', 'http://','https://', '&', '#')
    clean_text = ' '.join(txt for txt in text.split() if not txt.startswith(remove_keys))
    print(clean_text)
    allchars = [str for str in text]
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
    results = sentiment_pipeline(s_input)
    pred_senti = results[0]['label']
    if pred_senti == 'POSITIVE':
        pc = 1
    else:
        pc = 0
    return pc
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

def analyze_emotion(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
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
    (ext_text, ext_emoji) = extract_text_and_emoji(input_test)
    print(f'Extracted: "{ext_text}" , {ext_emoji}')
    ttext=""
    lang = detect(ext_text)
    if lang!="en":
      ext_text= detect_and_translate2(ext_text)
      print(f'Translated: "{ext_text}"')
    senti_text = get_sentiment(ext_text)
    print(f'Text value: {senti_text}')
    senti_emoji_value = sum(get_emoji_sentiment(ext_emoji, new_df_emoji))
    print_emo_val_avg = 0 if len(ext_emoji) == 0 else senti_emoji_value/len(ext_emoji)
    print(f'Emoji average value: {print_emo_val_avg}')
    senti_avg = (senti_emoji_value + senti_text) / (len(ext_emoji) + 1)
    print(f'Average value: {senti_avg}')
    senti_truth = "Positive" if senti_avg >= 0.5 else "Negative"
    emtext = analyze_emotion(ext_text)
    print(f'Text Emotion: {emtext}')
    return senti_truth
print(get_text_emoji_sentiment())

test_df = pd.read_csv("test.csv",usecols = ['text', 'label'])
test_df=test_df[test_df['label'] != 'neutral']
test_df['label'] = test_df['label'].apply(lambda x: 1 if x == 'positive' else 0)
test_df.head(6)


def preprocess_text(text):
    text = text.lower()
    text = ' '.join(word for word in text.split() if not word.startswith(('@', 'http', '#')))   
    return text if text else " " 

test_df['text'] = test_df['text'].apply(preprocess_text)

def detect_and_translate3(text):
    if not text.strip():
        return " "
    translator = Translator()
    try:
        lang = detect(text)
        if lang != 'en':
            trans = translator.translate(text, src=lang, dest='en')
            return trans.text
    except:
        pass
    return text  

def get_sentiment(text):
    text= detect_and_translate3(text)
    result = sentiment_pipeline(text)[0]['label']
    return 1 if result == 'POSITIVE' else 0

test_df['predicted_sentiment'] = test_df['text'].apply(get_sentiment)

y_true = test_df['label']
y_pred = test_df['predicted_sentiment']

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Accuracy ", accuracy)
print("Precision ", precision)
print("Recall ", recall)
print("F1 Score ", f1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
