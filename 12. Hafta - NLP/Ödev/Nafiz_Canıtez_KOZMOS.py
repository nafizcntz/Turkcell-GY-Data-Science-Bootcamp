##################################################
# Amazon Yorumları için Duygu Analizi
##################################################

##################################################
# Görev 1:  Metin Ön İşleme
##################################################
from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# ADIM 1 #
df = pd.read_excel("12. Hafta/nlp/datasets/amazon.xlsx")
df.head()

# ADIM 2 #

# Küçük Harf'e Çevirme
df['Review'] = df['Review'].str.lower()

# Noktalama Kaldırma
df['Review'] = df['Review'].str.replace('[^\w\s]', '')

# Sayıları Çıkarma
df['Review'] = df['Review'].str.replace('\d', '')

# Stopwords Çıkarma
nltk.download('stopwords')
sw = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

# Nadir Geçen Kelimeleri Çıkarma
temp_df = pd.Series(' '.join(df['Review']).split()).value_counts()
drops = temp_df[temp_df <= 1]
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

# Lemmatization
nltk.download('wordnet')
df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

##################################################
# Görev 2:  Metin Görselleştirme
##################################################
## ADIM 1 ##

## a
tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

## b
tf.columns = ["words", "tf"]
tf.sort_values("tf", ascending=False)

## c
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

## ADIM 2 ##

## a
text = " ".join(i for i in df.Review)

## b, c, d

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

##################################################
# Görev 3:  Duygu Analizi
##################################################

### ADIM 1 ###
# nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

### ADIM 2 ###
### a
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))

### b
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

### c
df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

### d
df["polarity_score"] = df["Review"].apply(lambda x: sia.polarity_scores(x)["compound"])
df["sentiment_label"] = df["polarity_score"].apply(lambda x: "pos" if x > 0 else "neg")

##################################################
# Görev 4:  Makine Öğrenmesine Hazırlık
##################################################

#### ADIM 1 ####
df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

y = df["sentiment_label"]
X = df["Review"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

#### ADIM 2 ####

#### a
tf_idf_word_vectorizer = TfidfVectorizer()

#### b
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X_train)

#### c
X_test = tf_idf_word_vectorizer.transform(X_test)

##################################################
# Görev 5:  Modelleme (LojistikRegresyon)
##################################################

##### ADIM 1 #####
log_model = LogisticRegression().fit(X_tf_idf_word, y_train)

##### ADIM 2 #####
##### a

y_preds = log_model.predict(X_test)

##### b
print(classification_report(y_test, y_preds))

##### c
log_model_cross = cross_val_score(log_model, X_test, y_test, scoring="accuracy", cv=5).mean()
# Accuracy : 0.8477222222222223

##### ADIM 3 #####
##### a
random_review = pd.Series(df["Review"].sample(1).values)

##### b
vectorizer = CountVectorizer()
vectorizer.fit(X_train)

##### c
random_review_count = vectorizer.transform(random_review)

##### d
random_review_pred = log_model.predict(random_review_count)
# array([1])

##### e
print("Örneklem : ", random_review[0], "\n", "Örneklem Tahmini : ", random_review_pred[0])

##################################################
# Görev 6:  Modelleme (Random Forest)
##################################################

###### ADIM 1 ######

###### a
rf_model = RandomForestClassifier().fit(X_tf_idf_word, y_train)

###### b
rf_model_cross = cross_val_score(rf_model, X_test, y_test, scoring="accuracy", cv=5, n_jobs=-1).mean()

###### c
print("Log Model Accuracy : ", log_model_cross, "\n", "RF Model Accuracy : ", rf_model_cross)
# Log Model Accuracy :  0.8477222222222223
#  RF Model Accuracy :  0.8824444444444446
# Görüldüğü üzere RF modeli daha iyi sonuç veriyor.



