import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import neattext.functions as nfx
from textblob import TextBlob
from collections import Counter
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("data/emotion_dataset_2.csv")

# df['Emotion'].value_counts().plot(kind = 'bar')
# plt.show()

# sns.countplot(x = 'Emotion', data = df)
# plt.show()

def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        result = 'Positive'
    elif sentiment < 0:
        result = 'Negative'
    else:
        result = 'Neutral'
    return result

df['Sentiment'] = df['Text'].apply(get_sentiment)
# print(get_sentiment("I love Coding"))
# print(df.head())

df['Clean_Text'] = df['Text'].apply(nfx.remove_stopwords)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_userhandles)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_punctuations)
# print(df[['Text', 'Clean_Text']])

def extract_keywords(text, num = 50):
    tokens = [token for token in text.split()]
    most_common_tokens = Counter(tokens).most_common(num)
    return dict(most_common_tokens)

emotion_list = df['Emotion'].unique().tolist()
# print(emotion_list)

joy_list = df[df['Emotion'] == "joy"]['Clean_Text'].tolist()
joy_docx = ' '.join(joy_list)
joy_keywords = extract_keywords(joy_docx)
# print(joy_keywords)

surprise_list = df[df['Emotion'] == "surprise"]['Clean_Text'].tolist()
surprise_docx = ' '.join(surprise_list)
surprise_keywords = extract_keywords(surprise_docx)

neutral_list = df[df['Emotion'] == "neutral"]['Clean_Text'].tolist()
neutral_docx = ' '.join(neutral_list)
neutral_keywords = extract_keywords(neutral_docx)

sadness_list = df[df['Emotion'] == "sadness"]['Clean_Text'].tolist()
sadness_docx = ' '.join(sadness_list)
sadness_keywords = extract_keywords(sadness_docx)

fear_list = df[df['Emotion'] == "fear"]['Clean_Text'].tolist()
fear_docx = ' '.join(fear_list)
fear_keywords = extract_keywords(fear_docx)

anger_list = df[df['Emotion'] == "anger"]['Clean_Text'].tolist()
anger_docx = ' '.join(anger_list)
anger_keywords = extract_keywords(anger_docx)

shame_list = df[df['Emotion'] == "shame"]['Clean_Text'].tolist()
shame_docx = ' '.join(shame_list)
shame_keywords = extract_keywords(shame_docx)

disgust_list = df[df['Emotion'] == "disgust"]['Clean_Text'].tolist()
disgust_docx = ' '.join(disgust_list)
disgust_keywords = extract_keywords(disgust_docx)

# def plot_wordcloud(docx):
#     mywordcloud = WordCloud().generate(docx)
#     plt.figure(figsize=(20,10))
#     plt.imshow(mywordcloud,interpolation='bilinear')
#     plt.axis('off')
#     plt.show()
# plot_wordcloud(joy_docx)

Xfeautures = df['Clean_Text']
Ylabels = df['Emotion']

cv = CountVectorizer()
X = cv.fit_transform(Xfeautures)

names = cv.get_feature_names_out()

X_train,X_test,Y_train,Y_test = train_test_split(X,Ylabels,test_size=0.3,random_state=42)

nv_model = MultinomialNB()
nv_model.fit(X_train,Y_train)

#Accuracy
# print(nv_model.score(X_test,Y_test))

#Predictions
y_pred_for_nv = nv_model.predict(X_test)
# print(y_pred_for_nv)

# text = input()
# sample = [text]
# vect = cv.transform(sample).toarray()

# #Making prediction
# print(nv_model.predict(vect))

# #prediction probability
# print(np.max(nv_model.predict_proba(vect)))
# print(nv_model.classes_)


def predict_emotion(sample,model):
    chunks = [sample[i:i + 500] for i in range(0, len(sample), 500)]
    myvect = cv.transform(chunks).toarray()
    prediction = model.predict(myvect)
    pred_proba = model.predict_proba(myvect)
    pred_percentage_for_all = dict(zip(model.classes_,pred_proba[0]))
    # return "Prediction:{}, Prediction Score:{}".format(prediction[0], np.max(pred_proba))
    return pred_percentage_for_all
# print(predict_emotion(sample,nv_model))

#Model Evaluation
#Classification report
# print(classification_report(Y_test,y_pred_for_nv))

# print(confusion_matrix(Y_test,y_pred_for_nv))

model_file = open("emotion_prediction_nv_model_24_april_2024.pkl","wb")
joblib.dump(nv_model,model_file)
model_file.close()