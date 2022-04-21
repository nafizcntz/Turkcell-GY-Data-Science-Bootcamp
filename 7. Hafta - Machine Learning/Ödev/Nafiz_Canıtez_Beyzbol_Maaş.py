######################################################
# Makine Öğrenmesi ile Beyzbolcu Maaş Tahmini
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score, confusion_matrix, \
    classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


def load_hitters():
    data = pd.read_csv(r"7. Hafta/machine_learning/datasets/hitters.csv")
    df = data.copy()
    return df


df = load_hitters()
df


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


##################################################
# Görev 1 : Keşifçi Veri Analizi ve Veri Ön İşleme
##################################################

# ADIM 1 #
df.info()
df.describe().T

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# ADIM 2 #
def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)


for col in df.columns:
    plot_numerical_col(df, col)


# ADIM 3 #
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "League", col)

# ADIM 4 #
plt.figure(figsize=(15, 10))
matrix = np.triu(df.corr())
sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, annot=True, mask=matrix)

# ADIM 5 #
df.describe().T

df.isnull().sum()

# ---------------------------------------------------- #
# Burada target değişkenimizdeki boş olan değerleri basic model kurarak tahmin ettirmeye çalıştım.
# Bazı çalışmalarda direk olarak KNNImputer kullanılmış ama ben daha iyi olabileceğini gördüğümden kendi modelimi kullandım.

df_null = df[df["Salary"].isnull()]
df_nnull = df[~df["Salary"].isnull()]
dff = df.copy()
df_nan_index = df_null.index
df_nnull.info()


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df_nnull = one_hot_encoder(df_nnull, cat_cols, drop_first=True)

y = df_nnull["Salary"]
X = df_nnull.drop(["Salary"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.15, random_state=17)
# Linear Regression
log_model = LinearRegression().fit(X_train, y_train)
y_pred = log_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
log_model.score(X_test, y_test)

df_null = one_hot_encoder(df_null, cat_cols, drop_first=True)
df_null.drop("Salary", axis=1, inplace=True)
y_pred_null = log_model.predict(df_null)

# --------
# KNNImputer
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)

dff = one_hot_encoder(dff, cat_cols, drop_first=True)

dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
# ----------


pd.DataFrame({"Knn": dff["Salary"].loc[df_nan_index], "Linear Model": y_pred_null})

df["Salary"].loc[df_nan_index] = y_pred_null

df.isnull().sum()


# ---------------------------------------------------- #

# ADIM 6 #
def outlier_plot(dataframe, numeric_cols):
    fig, ax = plt.subplots(nrows=len(numeric_cols), ncols=1, figsize=(5, 3))
    for i in range(len(numeric_cols)):
        sns.boxplot(x=dataframe[numeric_cols[i]], ax=ax[i])


outlier_plot(df, num_cols)


def outlier_replace(dataframe, numeric_cols, replace=False, lb_down=1, ub_up=1):
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


outlier_replace(df, num_cols, replace=True)

outlier_plot(df, num_cols)

# ADIM 7 #

df = one_hot_encoder(df, cat_cols, drop_first=True)

# ADIM 8 #

# 1986-1987 sezonunda yaptığı vuruşların başarı oranı
df["Hits_AtBat"] = df["Hits"] / df["AtBat"]
# 1986-1987 sezonunda yaptığı vuruşlar başına kazandırdığı puan
df["Runs_AtBat"] = df["Runs"] / df["AtBat"]
# 1986-1987 sezonunda yaptığı vuruşlar başına kazandırdığı en değerli puan
df["HmRun_AtBat"] = df["HmRun"] / df["AtBat"]

# Oyuncunun tüm kariyerinde yaptığı vuruşların başarı oranı
df["Hits_AtBat"] = df["CHits"] / df["CAtBat"]
# Oyuncunun tüm kariyerinde yaptığı vuruşlar başına kazandırdığı puan
df["CHmRun_CAtBat"] = df["CRuns"] / df["CAtBat"]
# Oyuncunun tüm kariyerinde yaptığı vuruşlar başına kazandırdığı en değerli puan
df["CHmRun_CAtBat"] = df["CHmRun"] / df["CAtBat"]

# Belli değerleri kategorikleştirme
df['CRBI_Seg'] = pd.qcut(x=df['CRBI'], q=3, labels=[1, 2, 3]).astype(int)
df['Walks_Seg'] = pd.qcut(x=df['Walks'], q=3, labels=[1, 2, 3]).astype(int)
df['CWalks_Seg'] = pd.qcut(x=df['CWalks'], q=3, labels=[1, 2, 3]).astype(int)
df['PutOuts_Seg'] = pd.qcut(x=df['PutOuts'], q=2, labels=[1, 2]).astype(int)
df["Years_Seg"] = pd.cut(df["Years"], [0, 3, 5, 10, 22], labels=[1, 2, 3, 4]).astype(int)

df.describe().T

# Oyuncun sene başına yaptıkları
df["CHits_Years"] = df["CHits"] / df["Years"]
df["CHmRun_Years"] = df["CHmRun"] / df["Years"]
df["CRuns_Years"] = df["CRuns"] / df["Years"]
df["CRBI_Years"] = df["CRBI"] / df["Years"]
df["CWalks_Years"] = df["CWalks"] / df["Years"]

df.info()

# ADIM 9 #
y = df["Salary"]
X = df.drop(["Salary"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.15, random_state=17)

# --------------------------

models = []

models.append(('LR', LinearRegression()))
models.append(('RF', RandomForestRegressor()))
models.append(('GBM', GradientBoostingRegressor()))
models.append(("XGBoost", XGBRegressor(objective='reg:squarederror')))
models.append(("LightGBM", LGBMRegressor()))

names = []
rmse = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    names.append(name)
tr_split = pd.DataFrame({'Name': names, 'RMSE': rmse})
tr_split = tr_split.sort_values(by="RMSE", ascending=True).reset_index(drop=True)
tr_split
# RMSE 199.39

gbm_model = GradientBoostingRegressor()
gbm_modelfit = gbm_model.fit(X_train, y_train)
y_pred = gbm_modelfit.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Train Rmse
y_pred_tr = gbm_modelfit.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred_tr))

def plot_importance(model, features, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()


plot_importance(gbm_model, X_train)

# ---------------------------------
# GridSearchCV

parameters = {'learning_rate': [0.01, 0.02, 0.03, 0.04],
              'subsample': [0.9, 0.5, 0.2, 0.1],
              'n_estimators': [100, 500, 1000, 1500],
              'max_depth': [4, 6, 8, 10]
              }

grid_GBR = GridSearchCV(estimator=gbm_model, param_grid=parameters, cv=2, n_jobs=-1)
grid_GBR.fit(X_train, y_train)

grid_GBR.best_estimator_
grid_GBR.best_score_
grid_GBR.best_params_

y_pred_grid = grid_GBR.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred_grid))
