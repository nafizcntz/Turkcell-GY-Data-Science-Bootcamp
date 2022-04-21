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

def load_telco():
    data = pd.read_csv(r"6. Hafta/feature_engineering/datasets/Telco-Customer-Churn.csv")
    df = data.copy()
    return df

# ADIM 1 #

df = load_telco()
df.head()
df.info()

df["customerID"].unique().shape # Her bir satır eşsiz bir kullanıcıya ait.

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

num_cols.append(cat_but_car[1]) # Numerik kolona TotalCharges eklendi.
cat_but_car.pop() # Listeden TotalCharges çıkarıldı.

df[df["TotalCharges"]==" "]
df[df["tenure"]==0]
# TotalCharges'ı boş olan müşteriler tenure değişkeni 0 olanlar.
# Yani bu müşteriler sadece kayıt olup hemen sildirenler olabilir.
# Bu yüzden MonthlyCharges değerlerini TotalCharges değerlerine atıyoruz.
for i in df.loc[df["TotalCharges"] == " "]["TotalCharges"].index:
    df["TotalCharges"][i] = df["MonthlyCharges"][i]

df["TotalCharges"] = df["TotalCharges"].astype(float)

df["SeniorCitizen"] = df["SeniorCitizen"].astype(object)

# ADIM 3 #

df[num_cols].describe().T

df[cat_cols]


# ADIM 4 #
df.groupby("Churn")[num_cols].mean()

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

outlier_replace(df, num_cols, lb_down=1, ub_up=1)

df[num_cols].describe().T
# Aykırı değer görünmüyor. O yüzden baskılama yapmadık.


# ADIM 6 #

df.isnull().sum()
# Eksik gözlem yok


# ADIM 7 #
plt.figure(figsize=(15, 10))

matrix = np.triu(df.corr())

sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, annot=True, mask=matrix)

df.corr()



##################################################
# Görev 2 : Feature Engineering
##################################################

## ADIM 1 ##

# Bakıldığında herhangi bir eksik veya aykırı gözlem bulunmamakta.


## ADIM 2 ##
df.head()

df.loc[df["SeniorCitizen"] == 1]

df.groupby("gender")["tenure"].mean()
df.groupby("gender")["TotalCharges"].mean()

df.groupby("SeniorCitizen")["tenure"].mean()


df.groupby("InternetService")["TotalCharges"].mean() # Fiber Altyapı kullananlar daha çok kazandırıyor.
df.groupby("InternetService")["tenure"].mean() # Ancak müşteri olarak kaldıkları ay süresine bakılırsa bir değişiklik yok.

df.groupby("PaymentMethod")["TotalCharges"].mean()

df.info()

df["tenure_seg"] = 0
for i, j in enumerate(df["tenure"]):
    if 1 <= j <= 12:
        df['tenure_seg'][i] = 1
    elif 13 <= j <= 24:
        df['tenure_seg'][i] = 2
    elif 25 <= j <= 36:
        df['tenure_seg'][i] = 3
    elif 37 <= j <= 48:
        df['tenure_seg'][i] = 4
    elif 49 <= j <= 60:
        df['tenure_seg'][i] = 5
    elif 61 <= j <= 72:
        df['tenure_seg'][i] = 6

df["Services_Count"] = (df[["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]] == "Yes").sum(axis=1)

df["Contract_Completed"] = 1
df["Contract_Completed"] = np.where(((df["Contract"] == "One year") &
                                    (df["tenure"] < 12)) |
                                    ((df["Contract"] == "Two year") &
                                    (df["tenure"] < 24)) |
                                    ((df["tenure"] == 0)), 0, 1)


## ADIM 3 ##

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    label_encoder(df, col)


df.loc[(df["InternetService"]=="No") & (df["OnlineBackup"]=="No internet service")]
# İnternet Servisi kolonunda "Hayır" olan satırlar diğer "İnternet servisi yok" olan kolonlardada bu şekilde.
# Toplam 1526 satır direk olarak böyle.


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if (10 >= df[col].nunique() > 2) &
            (col not in ["tenure_seg", "Services_Count"])]

df = one_hot_encoder(df, ohe_cols)

df.info()

## ADIM 4 ##
mms = MinMaxScaler()
df["MonthlyCharges_mms"] = mms.fit_transform(df[["MonthlyCharges"]])
df["TotalCharges_mms"] = mms.fit_transform(df[["TotalCharges"]])

## ADIM 5 ##

df.drop("customerID", inplace=True, axis = 1)

y = df["Churn"]
X = df.drop(["Churn"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)


import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
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
