######################################################
# Gözetimsiz Öğrenme ile Müşteri Segmentasyonu
######################################################

import warnings
import numpy as np
import datetime as dt
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_samples, silhouette_score

pd.set_option('display.max_columns', None)

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=5, car_th=20):
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

def outlier_plot(dataframe, numeric_cols):
    fig, ax = plt.subplots(nrows=int(len(numeric_cols)/2), ncols=2, figsize=(10, 10))
    fig.tight_layout(pad=1.0)
    t = 0
    for i in range(int(len(numeric_cols)/2)):
        for j in range(2):
            sns.boxplot(x=dataframe[numeric_cols[t]], ax=ax[i, j])
            t += 1

def outlier_replace(dataframe, numeric_cols, replace=False, lb_down=0.75, ub_up=1.25):
    lower_and_upper = {}
    for col in numeric_cols:
        q1 = dataframe[col].quantile(0.05)
        q3 = dataframe[col].quantile(0.95)
        iqr = 1.5 * (q3 - q1)

        lower_bound = q1 - iqr
        upper_bound = q3 + iqr

        lower_and_upper[col] = (lower_bound, upper_bound)
        if replace:
            dataframe.loc[(dataframe.loc[:, col] < lower_bound), col] = lower_bound * lb_down
            dataframe.loc[(dataframe.loc[:, col] > upper_bound), col] = upper_bound * ub_up

    print(lower_and_upper)

##################################################
# Görev 1 : Veriyi Hazırlama
##################################################

flo = pd.read_csv("9. Hafta/machine_learning/datasets/flo_data_20K.csv")
df = flo.copy()

check_df(df)

# ADIM 1 #
# Değişken türlerinin ayrıştırılması
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

# Kategorik değişkenlerin incelenmesi
for col in cat_cols:
    cat_summary(df, col)

# Sayısal değişkenlerin incelenmesi
df[num_cols].describe().T

# for col in num_cols:
#     num_summary(df, col, plot=True)

# Sayısal değişkenkerin birbirleri ile korelasyonu
correlation_matrix(df, num_cols)


# ADIM 2 #

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)


# ADIM 3 #
df.isnull().sum()


# ADIM 4 #

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# ADIM 5 #
outlier_plot(df, num_cols)

outlier_replace(df, num_cols)

outlier_replace(df, num_cols, replace=True)

outlier_plot(df, num_cols)

# ADIM 6 #

df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021, 6, 1)

df["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')
df.rename(columns={"order_num_total": "frequency",
                   "customer_value_total": "monetary"}, inplace=True)


df["recency_score"] = pd.qcut(df['recency'], 5, labels=[5, 4, 3, 2, 1])
df["frequency_score"] = pd.qcut(df['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
df["monetary_score"] = pd.qcut(df['monetary'], 5, labels=[1, 2, 3, 4, 5])

df['recency_score'] = df['recency_score'].astype(int)
df['frequency_score'] = df['frequency_score'].astype(int)
df['monetary_score'] = df['monetary_score'].astype(int)

df.info()

df["RF_SCORE"] = (df['recency_score'].astype(str) + df['frequency_score'].astype(str))
df["RFM_SCORE"] = (df['recency_score'].astype(str) + df['frequency_score'].astype(str) + df['monetary_score'].astype(str))

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

df['segment'] = df['RF_SCORE'].replace(seg_map, regex=True)

df['RF_SCORE'] = df['RF_SCORE'].astype(int)
df['RFM_SCORE'] = df['RFM_SCORE'].astype(int)

df.info()

# ADIM 7 #
df.drop("master_id", axis=1, inplace=True)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.groupby("interested_in_categories_12")["monetary"].mean()

cat_cols.append(cat_but_car[0])

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.info()
df.shape

##################################################
# Görev 2 : K-Means ile Müşteri Segmentasyonu
##################################################

## ADIM 1 ##


date_cols = df[num_cols].columns[df[num_cols].columns.str.contains("date")]
num_cols = df[num_cols].columns[~df[num_cols].columns.str.contains("date")]
for i in date_cols:
    df[i +'_Day'] = df[i].dt.day
    df[i +'_Month'] = df[i].dt.month
    df[i +'_Year'] = df[i].dt.year

df.drop(date_cols, axis=1, inplace=True)
df.info()

"""scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(scaled, columns=df[num_cols].columns)"""

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

## ADIM 2 ##

kmeans = KMeans()
ssd = []
K = range(2, 30)
for i in K:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_

# Silhouette Analizi

range_n_clusters = range(2, 30)
silhouette_avg = []
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(df)
    cluster_labels = kmeans.labels_
    silhouette_avg.append(silhouette_score(df, cluster_labels))

plt.plot(range_n_clusters, silhouette_avg, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title('Silhouette analysis For Optimal k')
plt.show()


## ADIM 3 ##
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_


## ADIM 4 ##
df_ = flo.copy()
clusters_kmeans = kmeans.labels_

df_["cluster"] = clusters_kmeans
df_["cluster"] = df_["cluster"] + 1

df_["cluster"].value_counts()

df_.groupby("cluster").agg(["count", "mean", "median"])


##################################################
# Görev 3 : Hierarchical Clustering ileMüşteri Segmentasyonu
##################################################

### ADIM 1 ###
hc_average = linkage(df, "average")

plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average, leaf_font_size=10)
plt.show()
# Bilgisayarım bu çıktıyı alamadığından optimal küme sayısını belirleme konusunda sıkıntı yaşadım.


# Küme Sayısını Belirlemek
plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()


### ADIM 2 ###

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")

clusters = cluster.fit_predict(df)

df__ = flo.copy()

df__["hi_cluster_no"] = clusters
df__["hi_cluster_no"] = df__["hi_cluster_no"] + 1

df__["kmeans_cluster_no"] = df_["cluster"]



### ADIM 3 ###

df__["hi_cluster_no"].value_counts()

df__["kmeans_cluster_no"].value_counts()

df__.groupby("kmeans_cluster_no").agg(["count", "mean", "median"])

df__.groupby("hi_cluster_no").agg(["count", "mean", "median"])

df
























