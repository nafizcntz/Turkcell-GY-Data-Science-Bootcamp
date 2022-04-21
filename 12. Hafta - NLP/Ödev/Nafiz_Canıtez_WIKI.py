##################################################
# Metin Ön işleme ve Görselleştirme
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

df = pd.read_csv("12. Hafta/nlp/datasets/wiki_data.csv", index_col=0)
df.head()

# ADIM 1 #
def clean_text(dataframe, column):
    # Küçük Harf'e Çevirme
    dataframe[column] = dataframe[column].str.lower()
    # Noktalama Kaldırma
    dataframe[column] = dataframe[column].str.replace('[^\w\s]', '')
    # Sayıları Çıkarma
    dataframe[column] = dataframe[column].str.replace('\d', '')

# ADIM 2 #
clean_text(df, "text")
df

# ADIM 3 #
nltk.download('stopwords')
def remove_stopwords(dataframe, column):
    sw = stopwords.words('english')
    dataframe[column] = dataframe[column].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

# ADIM 4 #
remove_stopwords(df, "text")
df

# ADIM 5 #
temp_df = pd.Series(' '.join(df['text']).split()).value_counts()
drops = temp_df[temp_df <= 1]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

# ADIM 6 #
nltk.download("punkt")
df["text"].apply(lambda x: TextBlob(x).words).head()

# ADIM 7 #
nltk.download('wordnet')
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


##################################################
# Görev 2:  Veriyi Görselleştiriniz
##################################################
## ADIM 1 ##
tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf.sort_values("tf", ascending=False)

## ADIM 2 ##
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

## ADIM 3 ##
text = " ".join(i for i in df.text)
wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


##################################################
# Görev 3:  Tüm Aşamaları Tek Bir Fonksiyon Olarak Yazınız
##################################################

def text_prep(dataframe, column, rarew_tresh=1, barplot_=False, wordcloud_=False):
    """

        Veri setindeki text içeren kolonlara metin ön işleme adımlarını uygulayan ve görselleştiren fonksiyondur.

        Parameters
        ------
            dataframe: dataframe
                    Üzerinde işlem yapılacak dataframe
            column: string
                    Üzerinde işlem yapılacak kolondur
            rarew_tresh: int
                    Nadir geçen kelimelerin frekens sınırını belirler
            barplot_: bool
                    Kelimelerin frekanslarını barplot ile görselleştirir
            wordcloud_: bool
                    Kelimelerin frekanslarını wordcloud ile görselleştirir
        Returns
        ------
            dataframe: dataframe
                    Üzerinde işlem yapılacak dataframe

        Examples
        ------
            df = pd.read_csv("12. Hafta/nlp/datasets/wiki_data.csv", index_col=0)
            df = metin_on_isleme(df, "text", rarew_tresh=100)
        """
    dataframe[column] = dataframe[column].str.lower()
    dataframe[column] = dataframe[column].str.replace('[^\w\s]', '')
    dataframe[column] = dataframe[column].str.replace('\d', '')

    nltk.download('stopwords')
    sw = stopwords.words('english')
    dataframe[column] = dataframe[column].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

    temp_df = pd.Series(' '.join(dataframe[column]).split()).value_counts()
    drops = temp_df[temp_df <= rarew_tresh]
    dataframe[column] = dataframe[column].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

    if barplot_:
        tf = dataframe[column].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        tf.columns = ["words", "tf"]
        tf.sort_values("tf", ascending=False)
        tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
        plt.show()

    if wordcloud_:
        text = " ".join(i for i in dataframe.column)
        wordcloud = WordCloud(max_font_size=50,
                              max_words=100,
                              background_color="white").generate(text)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

text_prep(df, "text", rarew_tresh=100)
































