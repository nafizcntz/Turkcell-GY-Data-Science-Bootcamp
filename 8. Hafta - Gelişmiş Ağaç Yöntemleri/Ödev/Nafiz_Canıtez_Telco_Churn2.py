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
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)


def load_telco():
    data = pd.read_csv(r"8. Hafta/HousePrice/dataset/Telco-Customer-Churn.csv")
    df = data.copy()
    return df


df = load_telco()
df.info()


# ADIM 1 #

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

num_cols.append(cat_but_car[1])  # Numerik kolona TotalCharges eklendi.
cat_but_car.pop()  # Listeden TotalCharges çıkarıldı.

df["customerID"] # cat_but_car

# ADIM 2 #

# TotalCharges'ı boş olan müşteriler tenure değişkeni 0 olanlar.
# Yani bu müşteriler sadece kayıt olup hemen sildirenler olabilir.
# Bu yüzden MonthlyCharges değerlerini TotalCharges değerlerine atıyoruz.
for i in df.loc[df["TotalCharges"] == " "]["TotalCharges"].index:
    df["TotalCharges"][i] = df["MonthlyCharges"][i]

# TotalCharges değişkeni object ama numeric olmalı.
df["TotalCharges"] = df["TotalCharges"].astype(float)

df["SeniorCitizen"] = df["SeniorCitizen"].astype(object)

df.info()

# ADIM 3 #

df[num_cols].describe().T

df[cat_cols]
for i in cat_cols:
    print("-----------------------------")
    print(df[i].value_counts())

# ADIM 4 #
df.groupby("Churn")[num_cols].mean()


def count_plot(dataframe, numeric_cols, target):
    fig, ax = plt.subplots(nrows=int(len(numeric_cols)/3), ncols=3, figsize=(10, 10))
    fig.tight_layout(pad=2.5)
    t = 0
    for i in range(int(len(numeric_cols)/3)):
        for j in range(3):
            plt.gca().set_title(numeric_cols[t])
            sns.countplot(x=numeric_cols[t], data=dataframe, hue=target, ax=ax[i, j])
            t += 1

count_plot(df, cat_cols, "Churn")


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)


# ADIM 5 #

def outlier_plot(dataframe, numeric_cols):
    fig, ax = plt.subplots(nrows=len(numeric_cols), ncols=1, figsize=(5, 3))
    fig.tight_layout(pad=2.5)
    for i in range(len(numeric_cols)):
        sns.boxplot(x=dataframe[numeric_cols[i]], ax=ax[i])


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
plt.figure(figsize=(16, 10))
df.corr()['Churn'].sort_values(ascending=False).plot(kind='bar', figsize=(20, 5))

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

df.groupby("InternetService")["TotalCharges"].mean()  # Fiber Altyapı kullananlar daha çok kazandırıyor.
df.groupby("InternetService")["tenure"].mean()  # Ancak müşteri olarak kaldıkları ay süresine bakılırsa bir değişiklik yok.

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

df["Services_Count"] = (df[["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                            "StreamingTV", "StreamingMovies"]] == "Yes").sum(axis=1)

df["Contract_Completed"] = 1
df["Contract_Completed"] = np.where(((df["Contract"] == "One year") &
                                     (df["tenure"] < 12)) |
                                    ((df["Contract"] == "Two year") &
                                     (df["tenure"] < 24)) |
                                    ((df["tenure"] == 0)), 0, 1)

df["Streaming_Tv_Movies"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)
df["AutomaticPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)

df.info()

## ADIM 3 ##

## Label Encoder
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

binary_cols.remove("Churn") # Churn değişkenini dışarı attık.

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    label_encoder(df, col)

# İnternet Servisi kolonunda "Hayır" olan satırlar diğer "İnternet servisi yok" olan kolonlardada bu şekilde.
# Toplam 1526 satır direk olarak böyle.


## One Hot Encoder
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if (10 >= df[col].nunique() > 2) &
            (col not in ["tenure_seg", "Services_Count"])]

df = one_hot_encoder(df, ohe_cols, drop_first=True)


## ADIM 4 ##

rs = RobustScaler()
for i in num_cols:
    df[i+"_robuts_scaler"] = rs.fit_transform(df[[i]])

df


##################################################
# Görev 3 : Modelleme
##################################################

### ADIM 1 ###

df.drop("customerID", inplace=True, axis=1)

y = df["Churn"]
X = df.drop(["Churn"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

def plot_importance(model, features, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()

# --------------------------

models = []

models.append(('RF', RandomForestClassifier()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(("XGBoost", XGBClassifier(objective='reg:squarederror')))
models.append(("LightGBM", LGBMClassifier()))
models.append(("CatBoost", CatBoostClassifier(verbose=False)))


names = []
score = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score.append(accuracy_score(y_test, y_pred))
    names.append(name)
tr_split = pd.DataFrame({'Name': names, 'Score': score})
tr_split = tr_split.sort_values(by="Score", ascending=False).reset_index(drop=True)
tr_split

# 0 GBM  0.803407
# 1 LightGBM  0.794180
# 2 CatBoost  0.790632
# 3 XGBoost   0.789922
# 4 RF        0.787793

### ADIM 2 ###

### 0. GBM
gbm_model = GradientBoostingClassifier()
gbm_modelfit = gbm_model.fit(X_train, y_train)
y_pred_gbm = gbm_modelfit.predict(X_test)
accuracy_score(y_test, y_pred_gbm)

plot_importance(gbm_model, X_train)

gbm_model.get_params()


### 1. LightGBM
lgbm_model = LGBMClassifier()
lgbm_modelfit = lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_modelfit.predict(X_test)
accuracy_score(y_test, y_pred_lgbm)

plot_importance(lgbm_model, X_train)

lgbm_model.get_params()


### 2. CatBoost
cb_model = LGBMClassifier()
cb_modelfit = cb_model.fit(X_train, y_train)
y_pred_cb = cb_modelfit.predict(X_test)
accuracy_score(y_test, y_pred_cb)

plot_importance(cb_model, X_train)

cb_model.get_params()


### 3. XGBoost
xgb_model = XGBClassifier()
xgb_modelfit = xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_modelfit.predict(X_test)
accuracy_score(y_test, y_pred_xgb)

plot_importance(xgb_model, X_train)

xgb_model.get_params()


###  GridSearchCV

parameters = {'learning_rate': [0.1, 0.2, 0.3],
              'subsample': [0.5, 1, 1.5, 2],
              'n_estimators': [100, 250, 500],
              'max_depth': [1, 3, 5]
              }


models_grid = []
models_grid.append(('GBM_Grid', gbm_model))
models_grid.append(("XGBoost_Grid", xgb_model))
models_grid.append(("LightGBM_Grid", lgbm_model))
models_grid.append(("CatBoost_Grid", cb_model))

names_grid = []
score_grid = []
for name, model in models_grid:
    grid_model = GridSearchCV(estimator=model, param_grid=parameters, cv=2, n_jobs=-1).fit(X_train, y_train)
    y_pred_grid = grid_model.predict(X_test)
    score_grid.append(accuracy_score(y_test, y_pred_grid))
    names_grid.append(name)
tr_split_grid = pd.DataFrame({'Name': names_grid, 'Score': score_grid})
tr_split_grid = tr_split_grid.sort_values(by="Score", ascending=False).reset_index(drop=True)
tr_split_grid

# 0  LightGBM_Grid  0.809794
# 1  CatBoost_Grid  0.809794
# 2  GBM_Grid       0.806246
# 3  XGBoost_Grid   0.805536


### ADIM 3 ###

grid_model_best = GridSearchCV(estimator=lgbm_model, param_grid=parameters, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)
y_pred_best = grid_model_best.predict(X_test)
accuracy_score(y_test, y_pred_best)


feature_imp = pd.DataFrame({'Value': grid_model_best.best_estimator_.feature_importances_, 'Feature': X_train.columns})
feature_imp.sort_values(by="Value", ascending=False)

feature_imp_best = feature_imp[feature_imp["Value"] > 0].sort_values(by="Value", ascending=False)

# Yeni Model

feature_imp_best.shape

y = df["Churn"]

X = df.drop(["Churn"], axis=1)
X = X[feature_imp_best["Feature"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

lgbm_model_b = LGBMClassifier()
lgbm_modelfit_b = lgbm_model_b.fit(X_train, y_train)
y_pred_lgbm_b = lgbm_modelfit_b.predict(X_test)
accuracy_score(y_test, y_pred_lgbm_b)

grid_model_best_b = GridSearchCV(estimator=lgbm_model_b, param_grid=parameters, cv=2, n_jobs=-1).fit(X_train, y_train)
y_pred_best_b = grid_model_best_b.predict(X_test)
accuracy_score(y_test, y_pred_best_b)
# 0.8076650106458482

metrics.confusion_matrix(y_test, y_pred_best_b)
print(classification_report(y_test, y_pred_best_b))

# Bilgisayarımı çok zorladığından GridSearchCV yaparken çok fazla deneme yapamadım.
# Ancak işe yaramayan değişkenleri çıkarıp tekrar model kurduğumuzda çok fazla bir değişim olmadığını gördüm.



cv_results = cross_validate(grid_model_best_b, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
# Bu işlemleri yaptıüımda cv_result değerleri nan dönüyor. Sebebini henüz çözemedim.


### BONUS ###

df["Churn"].value_counts()

drop_indices = np.random.choice(df[df["Churn"] == "No"].index, 3000, replace=False)

df_subset = df.drop(drop_indices).reset_index(drop=True)

X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X, y, test_size=0.20, random_state=17)


lgbm_model_bonus = LGBMClassifier()
lgbm_modelfit_bonus = lgbm_model_bonus.fit(X_train_sub, y_train_sub)
y_pred_lgbm_bonus = lgbm_modelfit_bonus.predict(X_test_sub)
accuracy_score(y_test_sub, y_pred_lgbm_bonus)
# Sınıflar eşitlensin diye rastgele olarak Churn no olanlardan çıkarma yaptım.
# Ancak herhangi bir farklılık oluşturmadı accury açısından.



















