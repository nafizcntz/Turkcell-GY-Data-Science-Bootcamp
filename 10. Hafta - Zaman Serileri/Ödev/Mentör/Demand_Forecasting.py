#####################################################
# Store Item Demand Forecasting
#####################################################
# Dataset hakkinda genel bilgiler:
# Farklı store için 3 aylık item-level sales tahmini.
# 5 yıllık bir veri setinde 10 farklı mağaza ve 50 farklı item var.
# Buna göre mağaza-item kırılımında 3 ay sonrasının tahminlerini vermemiz gerekiyor.
# hiyerarşik forecast ya da...


# Değişkenler
# date – Satış verilerinin tarihi (Tatil efekti veya mağaza kapanışı yoktur.)
# Store – Mağaza ID’si (Her bir mağaza için eşsiz numara.)
# Item – Ürün ID’si (Her bir ürün için eşsiz numara.)
# Sales – Satılan ürün sayıları (Belirli bir tarihte belirli bir mağazadan satılan ürünlerin sayısı.)

# validasyon setindeki hata değerini 13.8429 değerinin altında bulmak

#####################################################
# Loading Libraries
#######################################################
import time
import numpy as np
import pandas as pd
import seaborn as sns
#!pip install lightgbm
#import lightgbm as lgb
import warnings
from helpers.eda import *
from helpers.data_prep import *

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

########################
# Loading the data
########################

train = pd.read_csv('datasets/train.csv', parse_dates=['date'])  # object data tipini datetime olarak almis olduk
test = pd.read_csv('datasets/test.csv', parse_dates=['date'])

#sample_sub = pd.read_csv('10.hafta_time_series/demand_forecasting/sample_submission.csv')
df = pd.concat([train, test])  # ön işlemleri çin bir araya getirdik

df.head()

# 91300+45000=136300
#####################################################
# EDA
#####################################################
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


df["date"].min()  # ('2013-01-01 00:00:00')
df["date"].max()  # ('2018-03-31 00:00:00')

check_df(train)
check_df(test)
#check_df(sample_sub)
check_df(df)

# Satış dağılımı nasıl?
df["sales"].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])

# sns.boxplot(df["sales"])
# plt.show()
# sns.countplot(df.sales)
# plt.show()

# Kaç store var?
df[["store"]].nunique()

# Kaç item var?
df[["item"]].nunique()

# Her store'da eşit sayıda mı eşsiz item var?
df.groupby(["store"])["item"].nunique()

# Peki her store'da eşit sayıda mı sales var?
df.groupby(["store", "item"]).agg({"sales": ["sum"]})

# mağaza-item kırılımında satış istatistikleri
# Buradaki verilerden yola çıkarak(satış istatistiklerine bakarak )bir mağazaya yatırım yapılıp yapılmayacağı hakkında yorum yapılabilir
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})


#####################################################
# FEATURE ENGINEERING
#####################################################

def create_date_features(df):
    df['month'] = df.date.dt.month  # ay
    df['day_of_month'] = df.date.dt.day  # gun
    df['day_of_year'] = df.date.dt.dayofyear  # yilin kacinci gunu
    df['week_of_year'] = df.date.dt.weekofyear  # yilin kacinci haftasi
    df['day_of_week'] = df.date.dt.dayofweek  # haftanin kacinci gunu
    df['year'] = df.date.dt.year  # yil
    # alışverişlerde cuma-ctes-pazar önemli olduğu için 0'dan başlayarak haftanın 4. günü olan Cuma'dan itibaren çıktı 1 gelir.
    df["is_wknd"] = df.date.dt.weekday // 4  # Kalanı 1 olanlar haftasonudur (cuma, cmts, pazar)
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)  # ayin baslangici mi?
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)  # ayın bitişi mi  ?
    return df


# datasetimizi fonksiyona gonderiyoruz
df = create_date_features(df)
check_df(df)
df.shape

# data sette "store", "item", "month" genel bir bakis
# toplam satışlara bakarak trendi elde etmiş olduk
df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})


#####################################################
# Random Noise
#####################################################

# Train ve test veri setlerini birleştirmiştik. Tüm veriyi light gbm'de kullandığımızda overfitting'i engellemek için
# veriye rassal gürültü eklenmesi düşünülmüştür.

# sales değişkeninden ürettiğimizden dolayı
# Veri seti boyutunda, standart sapması 1.6 olacak şekilde rassal gürültü değerleri oluşturulur.
df1 = df.copy()


def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

np.random.normal(scale=1.6, size=[3,2])

#####################################################
# Lag/Shifted Features
#####################################################
# Zaman serisinde geçmiş değerler kullanılır.
# Light gbm ile belirlenecek adım sayısı kadar önceki değerlere ulaşmak için bu değerler yeni değişken olarak veriye eklenir.


# Geçmiş değerlere ilişkin featurelar üreteceğiz neden??
# Zaman serisinde bir sonraki değer en çok kendinden önceki değerden etkilenir. (geçmiş gerçek değerler)
# yani şunu biliyoruz ki zaman serileri ile çalışıyorsak geçmiş değerlere öncelik vermeliyiz. (zorundayız)


# Satır bazında aynı store, item ve date kırılımı göz önünde bulundurularak sıralama yapılır.
# Burdaki sales değerinin bir önceki değerine ulaşmam lazım ama bunun neye göre geldiğini bilmiyorum
# Zemini sağlamlaştırmak adına Satır bazında aynı store, item ve date kırılımı göz önünde bulundurularak sıralama yapılır.

df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)
df.head()

# shift fonksiyonu icin bir ornek  1.gecikme alır
df["sales"].shift(1).values[0:10]

# birden fazla shift degiskeni iceren bir sozlukten dataframe yaratmak
pd.DataFrame({"sales": df["sales"].values[0:10],
              "lag1": df["sales"].shift(1).values[0:10],
              "lag2": df["sales"].shift(2).values[0:10],
              "lag3": df["sales"].shift(3).values[0:10],
              "lag4": df["sales"].shift(4).values[0:10]})

# bir datframemin  store ve item kiriliminda Sales icin transform ile shift fonksiyonunu uygulama
# NAN olanlar test setinden geldiği için NAN
df.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(1))
df.sales[0:5]


def lag_features(dataframe, lags):
    for lag in lags:  # lag listesi icin loop
        # dataframedeki sales degiskeni icin shift ile lag miktari kadar kaydirip, random noise ekle ve
        # ilgili kolonu lag degerinden yaratilmis kolon ismiyle dataframe ekle
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df.head()
df.tail()


# lag_features fonksiyonunu lag listesi icin cagir
# 1.gün öncesine gitmemem nedeni? ->çünkü 3 ayın sonrasını üretmemiz gerekiyor
df = lag_features(df,
                  [91, 98, 105, 112, 119, 126, 133, 140, 147, 154, 161, 168, 175, 182, 189, 196, 203, 364, 546, 728])
check_df(df)

df.tail()

# Sales değeri NaN olanlar Test verisidir. Test veri setine lag ile trainden gelen değerler kontrol edildi.
df[df["sales"].isnull()]
test.shape
########################
# Rolling Mean Features
########################

# Hareketli Ortalamalar
df["sales"].head(10)
df["sales"].rolling(window=2).mean().values[0:10]

# kendisi dahil önceki değerlerin ortalaması:
pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].rolling(window=5).mean().values[0:10]})

# kendisi dahil olmamalı ki geçmişteki trendi ifade edebilecek mevcut değerden bağımsız bir feature türetilebilsin:
pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].shift(1).rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].shift(1).rolling(window=5).mean().values[0:10]})


# dataframedeki sales degiskeni icin shift ile lag miktari kadar kaydirip, random noise ekle ve
# ilgili kolonu lag degerinden yaratilmis kolon ismiyle dataframe ekle

## min_periods: en az 10 değer olmalı ki, bu windowdan sonuç dönsün değilse NA döner.
# window: kac adim geriye gideyim
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


# Hareketli Ortalamalar Fonksiyonu cagir
# yıllık trendi yansıtmaya çalışıyoruz
df = roll_mean_features(df, [365, 546])  #1 ve 1.5 yil oncesine iliskin veri
df.head()
df.tail()
########################
# Exponentially Weighted Mean Features  :Üstel Ağırlıklı Ortalama Feature Üretme
########################
# Hareketli ortalama yerine eksponensiyel olarak ağırlıklandırılmış ortalama değerleri kullanılarak yeni değişkenler oluşturulur.

# Ustsel ve agirlikli ortalamalar
# yakın döneme daha fazla ağırlık verelim:

# Üssel ağırlıklı ortalama featureları türetiyoruz.
# Buradaki alfa geçmiş değerlere ne kadar ağırlık vereceğimizi gösteren parametre.
# 0' yakın ise uzak geçmiş değerler, 1'e yakın ise yakın geçmiş değerlere ağırlık veriliyo

pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "ewm099": df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10],  #en yakına daha fazla ağırlıklı ağırlık ver
              "ewm095": df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm02": df["sales"].shift(1).ewm(alpha=0.1).mean().values[0:10]})  # en yakin az agirlik ver

# Ustsel ve agirlikli ortalamalar fonsiyonu
# alphas: ortalamalar icin katsayi degeri(0 - 1) listesi
# lags: ortalama periotlari listesi

# üssel ağırlıklı ortalama featurelar türetir.
# alfa geçmiş değerlere ne kadar ağırlık verileceğini belirten parametredir.
# 0'a yakın olması : uzak geçmiş
# 1'e yakın olması : yakın geçmişe ağırlık verileceği anlamına geliyor.

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:   # katsayilar icin dongu
        for lag in lags:   # periodlar icin dongu
            # sale degiskenine katsayilar ve periodlar icin ustsel ve agirlikli ortalama hesapla
            # ve uygun isimle kolon olarak dataframe ekle
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


# katsayilar ve periodlar listesi
alphas = [0.98, 0.95, 0.92, 0.9, 0.86, 0.83, 0.8, 0.79, 0.75, 0.7, 0.65, 0.5]
lags = [91, 98, 105, 112, 119, 126, 133, 140, 147, 180, 270, 365, 546, 728]

# Ustsel ve agirlikli ortalamalar fonsiyonu cagir
df = ewm_features(df, alphas, lags)
df.head()

########################
# One-Hot Encoding
########################
# 'store', 'item', 'day_of_week', 'month' degiskenleri icin oneHot encoder uygula
df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])
df.head()
df.shape
########################
# Converting sales to log(1+sales)
########################
# sales'e 1 eklendikten sonra logaritmik dönüşüm yapılıyor.
# çünki 0'ın logaritması alınmaz.
# bağımlı değişkeni dönüştürmek
# düzenleme

# Ağaç modellerinde artıklar üzerinden bir model kurulduğu için artıkların küçültülmesi ve iterasyon sayısının azaltılması için
# bağımlı değişkene logaritmik dönüşüm yapılır.

# sales degerlerini logunu al
df['sales'] = np.log1p(df["sales"].values)
check_df(df)


#####################################################
# Model
#####################################################

########################
# Custom Cost Function
########################

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# büyük olan hataların dominantlığını kırmak için:
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)


# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)
# smape ne kadar dusuk o kadar iyi, yuzdelik cinsten basari gosterir
# gerçek değer:10         | 10 - 40 | / (|10| + |40|) / 2
# tahmin değer: 40


# preds: tahmin edilen degerler
# target: gercek degerler

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]  # tahmin edilen ve gercek degerler
    num = np.abs(preds - target)  # mutlak hatalar
    denom = np.abs(preds) + np.abs(target)  # SMAPE hesabindaki duzelme icin payda degeri
    smape_val = (200 * np.sum(num / denom)) / n  # SMAPE degerini hesaplama
    return smape_val


# preds: tahmin edilen degerler
# train_data: train dataframe
# return olarak hesaplanan sample değerlerini verir
def lgbm_smape(preds, train_data):
    # egitim dataframedeki degisken isimlerini al
    labels = train_data.get_label()
    # Tahmin edilen degisken degerleri ve gercek degisken degerleri ile SMAPE fonksiyonunu cagir
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


#####################################################
# MODEL VALIDATION
#####################################################

# Light GBM: optimizasyon 2 açıdan ele alınmalı.

########################
# Time-Based Validation Sets
########################
# Kaggle test seti tahmin edilecek değerler: 2018'in ilk 3 ayı.

train["date"].min(), train["date"].max()  # ocak 2013-aralik 2017 dahil
test["date"].min(), test["date"].max()  # ocak 2018- mart 2018 dahil


# 2017'nin başına kadar (2016'nın sonuna kadar) train seti.
train = df.loc[(df["date"] < "2017-01-01"), :]  # minik train  , 2017 basina kadar seciyorum, 2017 ilk 3 senesini tahmin edicem
train["date"].min(), train["date"].max()  # ocak 2013 - aralik 2016 dahil

# 2017'nin ilk 3'ayı validasyon seti.
# Neden 2017 yılının ilk 3 ayını aldık??
# 2018 yılının ilk 3 ayını tahmin etmek istedğimiz için 2017 yılının ilk 3 ayını alıyoruz
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]  # ocak-nisan 2017

df.columns

# ML hesaplamasinda kullanilacak degiskenleri liste seklinde kaydet  bağımsız değişkenleri çıkardk
cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

# Egitim verisini kaydet
Y_train = train['sales']
X_train = train[cols]

# validasyon verisini kaydet
Y_val = val['sales']
X_val = val[cols]

# 730k train-45 k val
Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

# LightGBM parameters sozluk objesi
# LightGBM de en önemli parametre :
lgb_params = {'metric': {'mae'},  # loss fonksiyonu icin kullanilacak hata hesaplama metodu
              'num_leaves': 10,  # mbir agactaki max yaprak sayisi
              'learning_rate': 0.02,  # loss funksiyon hesaplamasinda kullanilacak katsayi
              'feature_fraction': 0.8,  # ağaç oluşturmaya başlarken randomly olarak featureların %80'ini al, rf'in random subspace ozelligi
              'max_depth': 3,  # agacin maximum derinligi
              'verbose': 0,  # ciktinin formatini belirleyen parametre
              'num_boost_round': 10000,  # iterasyon sayisi (kaç iterasyonda artık modelleyeceğimi belirtir) en az 10k, 15k
              'early_stopping_rounds': 250,
              # iyilesme gorulmediği zaman hesaplamayi erken birakma sayisi : her 250 satırda bir hata düşüyor mu düşmüyor mu kontrol et!
              'nthread': -1}  # kullanilacak islemci cekirdek sayisi

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error
# learning_rate: shrinkage_rate, eta
# num_boost_round: n_estimators, number of boosting iterations.
# nthread: num_thread, nthread, nthreads, n_jobs

# islemlerin daha hizli olmasi icin dataframeleri LGM dataset formatina cevrilmesi
lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

# LGM model nesnesi
model = lgb.train(lgb_params, lgbtrain,  # parametereler ve egitim dataset
                  valid_sets=[lgbtrain, lgbval],  # validasyon dataset
                  num_boost_round=lgb_params['num_boost_round'],  # iterasyon sayisi
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  # # iyilesme gorulmedi zaman hesaplamayi erken birakma sayisi
                  feval=lgbm_smape,  # loss fonksiyon hesaplama methodu
                  verbose_eval=100)  # cikti formati

# en iyi model parametreleri ile tahmin edilen validasyon degerleri
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
smape(np.expm1(y_pred_val), np.expm1(Y_val))  # tahmin edilen degerler icin hata hesabi , validasyon hatası


# 13.501222504859241
########################
# Değişken önem düzeyleri
########################
# LGM icin featurelarin onem degerlerini cizdir
# plot: plot cizdirilmesi veya cizdirlmemesi icin secenek
# num: gosterilecek feature sayisi
def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    # feature sozluk nesnesini feature isimleri, kac kere kullanildigi ve toplam gain degerleri icin olustur
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))  # plot size
        sns.set(font_scale=1)  # font boyutu
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])  # plot tipi ve plot edilecek degerler
        plt.title('feature')  # plot basligi
        plt.tight_layout()  # plot uzerinde nesnelerin bir biri uzerine binmemesi icin duzenleme
        plt.show()  # plotu goster
    else:
        # ozet ciktisi
        print(feat_imp.head(num))


# LGM icin featurelarin onem degerlerini yazdir
plot_lgb_importances(model, num=30)
# LGM icin featurelarin onem degerlerini cizdir
plot_lgb_importances(model, num=30, plot=True)
# LGM featurelarin onem degerlerini model nesnesinden cizdir ve yazdir
lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()










##########################################
# Final Model
##########################################
# final modelini olusturmak icin egitim datasetleri olustur
# sales içindeki NA olamayn demek trainleri al demek
train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

# final modelini olusturmak icin test datasetleri olustur
# sales ları NA olanları al
test = df.loc[df.sales.isna()]
X_test = test[cols]

# feature fraction: her iterasyonda göz önünde bulundurulacak değişken sayısı.
# num_boost_round: iterasyon sayısı
# early_stopping_rounds: validasyonda artık hata düşmüyorsa dur der.
# bu aynı zamanda num_bost_round kadar iterasyon yapmaması için de iyi bir özelliktir.
# verbose: ekrana verilecek bilgi
# nthread: kullanılacak işlemci sayısı: -1 hepsini kullan demek.


# LightGBM optimum parametre degeri icin sozluk objesi
lgb_params = {'metric': {'mae'},  # loss fonksiyonu icin kullanilacak hata hesaplama metodu
              'num_leaves': 15,  # maksimum eleman sayisi
              'learning_rate': 0.02,  # loss funksiyon hesaplamasinda kullanilacak katsayi
              'feature_fraction': 0.8,  # kullanilacak maksimum degisken sayisi
              'max_depth': 5,  # agacin maximum derinligi
              'verbose': 0,  # ciktinin formatini belirleyen parametre
              'nthread': -1,  # kullanilacak islemci cekirdek sayisi
              "num_boost_round": model.best_iteration}  # iterasyon sayisi

# Tum, egitim, test datasetleri icin LightGBM dataset objelerini olustur
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)
# 10000 (best params)

# Create submission
# id ve sales boş kaggle attığı
submission_df = test.loc[:, ['id', 'sales']]
submission_df['sales'] = np.expm1(test_preds)  # sales degiskenini exponential degerini hesapla
submission_df['id'] = submission_df.id.astype(int)
submission_df.to_csv('submission_demand.csv', index=False)
submission_df.head(20)
