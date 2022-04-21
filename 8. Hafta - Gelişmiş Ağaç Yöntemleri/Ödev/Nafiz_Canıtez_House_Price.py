######################################################
# Makine Öğrenmesi ile Ev Fiyatları Tahmini
######################################################

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import pickle



pd.set_option('display.max_columns', 10)
pd.set_option('display.width', None)

warnings.simplefilter(action='ignore', category=Warning)


train = pd.read_csv("8. Hafta/HousePrice/dataset/train.csv")
test = pd.read_csv("8. Hafta/HousePrice/dataset/test.csv")

df = train.copy()


def grab_col_names(dataframe, cat_th=0, car_th=25):
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


for col in df.columns[0:2]:
    plot_numerical_col(df, col)


# ADIM 3 #
"""def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "League", col)"""


# ADIM 4 #
"""plt.figure(figsize=(15, 10))
matrix = np.triu(df.corr())
sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, annot=True, mask=matrix)
"""


# ADIM 5 #
df.isnull().sum()

id = df["Id"]

columns_null = df.columns[df.isnull().sum() > 0].tolist()

df[columns_null].info()

df[columns_null].isnull().sum()

df.shape

df.drop(["Id", "LotFrontage", "Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"], axis=1, inplace=True)


df[columns_null] = df[columns_null].apply(lambda x: x.fillna(x.mode()[0]) if x.dtypes == "O" else x.fillna(x.median()))

#df.dropna(inplace=True)

df.columns[df.isnull().sum() > 0].tolist()

#-------------------
"""columns_null = df.columns[df.isnull().sum() > 0].tolist()

df_null = pd.DataFrame(df[columns_null])

df_null[df["GarageType"].isnull()]

df_null_index = df_null[df["GarageType"].isnull()].index

df.drop(df.index[df_null_index], inplace=True)"""
#-------------------

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# ADIM 6 #
def outlier_plot(dataframe, numeric_cols):
    fig, ax = plt.subplots(nrows=int(len(numeric_cols)/3), ncols=3, figsize=(10, 10))
    fig.tight_layout(pad=1.0)
    t = 0
    for i in range(int(len(numeric_cols)/3)):
        for j in range(3):
            sns.boxplot(x=dataframe[numeric_cols[t]], ax=ax[i, j])
            t += 1

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

outlier_replace(df, num_cols)

outlier_replace(df, num_cols, replace=True)

outlier_plot(df, num_cols)

df["BsmtFinSF2"] # Bazı kolonlar sadece sıfır veya birden oluşuyor.

# Sadece sıfır ve bir içeren kolonları kaldırıyoruz.
df = df.loc[:, (df != (0 and 1)).any(axis=0)].reset_index(drop=True)


# ADIM 7 #

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


plt.figure(figsize=(16,10))
df.corr()['SalePrice'].sort_values(ascending=False).plot(kind='bar', figsize=(20,5))

df.info()
# Label Encoder

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols

df["Neighborhood"].value_counts()

df.groupby("Neighborhood")["SalePrice"].mean().sort_values(ascending=False)

# Mahallelerdeki ev fiyatlarının ortalamasına göre pahalıdan ucuza bir kategorikleştirme gerçekletirdik.
a = pd.DataFrame(pd.qcut(df.groupby("Neighborhood")["SalePrice"].mean(), 4, labels=[1, 2, 3, 4]))
df["Neighborhood_seg"] = ""
for i in df.index:
    for j in a.index:
        if df["Neighborhood"][i] == j:
            df["Neighborhood_seg"][i] = a.loc[j][0]
            break

df["Neighborhood_seg"].value_counts()


# Binary Encoder
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

df["Street"].value_counts()
df["Utilities"].value_counts()
df["CentralAir"].value_counts()

# Binary olan kolonlardaki dağılımlar düzgün olmadığından ötürü bize bir bilgi vermesi imkansız.
# Bu yüzden binary olan kolonları kaldırıyoruz.

df.drop(binary_cols, axis=1, inplace=True)
df


# Rare Encoding

df["MSZoning"].value_counts()

#----------------------------------------------------

"""wr_num_cols = df[cat_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [ele for ele in cat_cols if ele not in wr_num_cols]

outlier_plot(df, wr_num_cols)
outlier_replace(df, wr_num_cols, replace=True)

df.drop(["BsmtHalfBath", "KitchenAbvGr", "PoolArea"], axis=1, inplace=True)
df["PoolArea"].value_counts()"""
#----------------------------------------------------
cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols.remove("Neighborhood_seg")

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SalePrice", cat_cols)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.3)

rare_analyser(new_df, "SalePrice", cat_cols)

neig_seg = df["Neighborhood_seg"]
df = rare_encoder(df.loc[:, df.columns != 'Neighborhood_seg'], 0.3) # !!!!!!
df["Neighborhood_seg"] = neig_seg
df["Neighborhood_seg"] = df["Neighborhood_seg"].astype(int)
df.drop(["LandContour", "LandSlope", "Neighborhood", "Condition2",
         "RoofMatl", "BsmtCond", "Heating", "Electrical",
         "Functional", "GarageQual", "GarageCond", "PavedDrive"],
         axis=1, inplace=True)

df.info()

# One Hot Encoder

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.info()


# ADIM 8 #


y = df["SalePrice"]
X = df.drop(["SalePrice"], axis=1)

rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()

cv_results = cross_validate(rf_model, X, y, cv=5, scoring=["accuracy"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)

# --------------------------

models = []

models.append(('RF', RandomForestRegressor()))
models.append(('GBM', GradientBoostingRegressor()))
models.append(("XGBoost", XGBRegressor(objective='reg:squarederror')))
models.append(("LightGBM", LGBMRegressor()))
models.append(("CatBoost", CatBoostRegressor(verbose=False)))

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

# 0  CatBoost  17386.582202
# 1  LightGBM  19745.869969
# 2       GBM  20008.745313
# 3        RF  20324.102602
# 4   XGBoost  21423.125956


cb_model = CatBoostRegressor(verbose=False)
cb_modelfit = cb_model.fit(X_train, y_train)
y_pred = cb_modelfit.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Train Rmse
y_pred_tr = cb_modelfit.predict(X_train)
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



plot_importance(cb_model, X_train)

cb_model.get_params()
# ---------------------------------
# GridSearchCV

cb_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}


cb_best_grid = GridSearchCV(cb_model, cb_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
cb_best_grid.best_params_

cb_final = CatBoostRegressor(**cb_best_grid.best_params_, verbose=False, random_state=17).fit(X, y)
cb_final.best_score_
cb_final.best_params_

y_pred_grid = cb_final.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred_grid))
# 4578.863686395691


filename = 'finalized_model.sav'
pickle.dump(cb_final, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))

y_pred_grid_a = loaded_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred_grid_a))

