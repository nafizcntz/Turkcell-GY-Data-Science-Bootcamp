##################################################
# Görev 1 : Keşifçi Veri Analizi
##################################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load_diyabet():
    data = pd.read_csv(r"6. Hafta/feature_engineering/datasets/diabetes.csv")
    df = data.copy()
    return df
# ADIM 1 #

df = load_diyabet()
df.head()
df.info()


# ADIM 2 #
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# ADIM 3 #

df[num_cols].describe().T

df[cat_cols] # Sadece hedef değişken var

# ADIM 4 #

df.groupby("Outcome")[num_cols].mean()

# ADIM 5 #

def outlier_plot(dataframe, numeric_cols):
    fig, ax = plt.subplots(nrows=len(numeric_cols), ncols=1, figsize=(5,3))
    for i in range(len(numeric_cols)):
        sns.boxplot(x = dataframe[numeric_cols[i]], ax = ax[i])

outlier_plot(df, num_cols)



def outlier_replace(dataframe, numeric_cols, replace=False, lb_down=0.75, ub_up=1.25):
    lower_and_upper = {}
    for col in numeric_cols:
        q1 = dataframe[col].quantile(0.25)
        q3 = dataframe[col].quantile(0.75)
        iqr = 1.5 * (q3 - q1)

        lower_bound = q1 - iqr
        upper_bound = q3 + iqr

        lower_and_upper[col] = (lower_bound, upper_bound)
        if replace:
            dataframe.loc[(dataframe.loc[:, col] < lower_bound), col] = lower_bound * lb_down
            dataframe.loc[(dataframe.loc[:, col] > upper_bound), col] = upper_bound * ub_up

    print(lower_and_upper)

outlier_replace(df, num_cols)



# ADIM 6 #

df.isnull().sum()
# Eksik gözlem yok


# ADIM 7 #
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, annot=True)

df.corr()



##################################################
# Görev 2 : Feature Engineering
##################################################

## ADIM 1 ##

outlier_plot(df, num_cols)

outlier_replace(df, num_cols, replace=True)

outlier_plot(df, num_cols)

df[df["Glucose"] == 0].groupby("Outcome")[num_cols].mean()

df[df["Insulin"] == 0].groupby("Outcome")[num_cols].mean()

df.groupby("Outcome")[num_cols].mean()

df[df["Insulin"] != 0].describe().T

df.describe().T

df["Glucose"].mask(df["Glucose"] == 0, df["Glucose"].mean(), inplace = True)

df["Insulin"].mask(df["Insulin"] == 0, df["Insulin"].mean(), inplace = True)

df["SkinThickness"].mask(df["SkinThickness"] == 0, df["SkinThickness"].mean(), inplace = True)


## ADIM 2 ##

df.head()
df.describe().T

df.groupby(["Age", "Outcome"]).size()
df.Age.value_counts()

df['BMI_seg'] = ""
for i, j in enumerate(df["BMI"]):
    if j < 18.5:
        df['BMI_seg'][i] = "Underweight"
    elif 25 > j >= 18.5:
        df['BMI_seg'][i] = "Healthy"
    elif 30 > j >= 25:
        df['BMI_seg'][i] = "Overweight"
    elif j >= 30:
        df['BMI_seg'][i] = "Obese"
df['BMI_seg'].value_counts()

"""
df['BloodPressure_seg'] = ""
for i, j in enumerate(df["BloodPressure"]):
    if j < 80:
        df['BloodPressure_seg'][i] = "Optimal"
    elif 85 > j >= 80:
        df['BloodPressure_seg'][i] = "Normal"
    elif 90 > j >= 85:
        df['BloodPressure_seg'][i] = "High Normal"
    elif 100 > j >= 90:
        df['BloodPressure_seg'][i] = "Grade 1 hypertension"
    elif 110 > j >= 100:
        df['BloodPressure_seg'][i] = "Grade 2 hypertension"
    elif j >= 100:
        df['BloodPressure_seg'][i] = "Grade 3 hypertension"

df['BloodPressure_seg'].value_counts()
"""
# BloodPressure segmentleri eklediğimiz zaman model başarıları düşüyor.
# O yüzden eklememeyi tercih ediyorum.

## ADIM 3 ##

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
# Sadece hedef değişkenimiz binary olarak gözüküyor.
# Kategorik dğişkenimiz olmadığından rare analyze'da yapamıyoruz.


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ["BMI_seg"])
df.info()


## ADIM 4 ##

def robust_scaler(dataframe, numeric_cols):
    rs = RobustScaler()
    for i in numeric_cols:
        dataframe[i] = rs.fit_transform(dataframe[[i]])

robust_scaler(df, num_cols)

df.head()


## ADIM 5 ##

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)


import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('XGB', xgb.XGBClassifier()))


names = []
scores = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)
tr_split = pd.DataFrame({'Name': names, 'Score': scores})
tr_split = tr_split.sort_values(by="Score", ascending=False)
tr_split

#---------------------------------------------------------
axis = sns.barplot(x='Name', y='Score', data=tr_split)
axis.set(xlabel='Classifier', ylabel='Accuracy')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width() / 2, height + 0.005, '{:1.4f}'.format(height), ha="center")

plt.show()