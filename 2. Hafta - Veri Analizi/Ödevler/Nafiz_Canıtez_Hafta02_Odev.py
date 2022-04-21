
###############################################
# ÖDEV 1: List Comprehension Applications
###############################################

###############################################
# Görev 1: car_crashes verisindeki numeric değişkenlerin isimlerini büyük harfe çeviriniz ve başına NUM ekleyiniz.
###############################################

import seaborn as sns
df = sns.load_dataset("car_crashes")

# Veri setini baştan okutarak aşağıdaki çıktıyı elde etmeye çalışınız.

# ['NUM_TOTAL',
#  'NUM_SPEEDING',
#  'NUM_ALCOHOL',
#  'NUM_NOT_DISTRACTED',
#  'NUM_NO_PREVIOUS',
#  'NUM_INS_PREMIUM',
#  'NUM_INS_LOSSES',
#  'ABBREV']

# Notlar:
# Numerik olmayanların da isimleri büyümeli.
# Tek bir list comp yapısı ile yapılmalı.



###############################################
# Görev 1 Çözüm
###############################################

df

df.columns

["NUM_" + col.upper() if df[col].dtypes != 'O' else col.upper() for col in df.columns]

###############################################
# Görev 2: İsminde "no" BARINDIRMAYAN değişkenlerin isimlerininin SONUNA "FLAG" yazınız.
###############################################

# Tüm değişken isimleri büyük olmalı.
# Tek bir list comp ile yapılmalı.



# Beklenen çıktı:

# ['TOTAL_FLAG',
#  'SPEEDING_FLAG',
#  'ALCOHOL_FLAG',
#  'NOT_DISTRACTED',
#  'NO_PREVIOUS',
#  'INS_PREMIUM_FLAG',
#  'INS_LOSSES_FLAG',
#  'ABBREV_FLAG']

###############################################
# Görev 2 Çözüm
###############################################

[col.upper() + "_FlAG" if "no" not in col else col.upper() for col in df.columns]

###############################################
# Görev 3: Aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin isimlerini seçerek yeni bir df oluşturunuz.
###############################################

# df.columns
# og_list = ["abbrev", "no_previous"]

# Önce yukarıdaki listeye göre list comprehension kullanarak new_cols adında yeni liste oluşturunuz.
# Sonra df[new_cols] ile bu değişkenleri seçerek yeni bir df oluşturunuz adını new_df olarak isimlendiriniz.


# Beklenen çıktı:

# new_df.head()
#
#    total  speeding  alcohol  not_distracted  ins_premium  ins_losses
# 0 18.800     7.332    5.640          18.048      784.550     145.080
# 1 18.100     7.421    4.525          16.290     1053.480     133.930
# 2 18.600     6.510    5.208          15.624      899.470     110.350
# 3 22.400     4.032    5.824          21.056      827.340     142.390
# 4 12.000     4.200    3.360          10.920      878.410     165.630

###############################################
# Görev 3 Çözüm
###############################################

og_list = ["abbrev", "no_previous"]

new_cols = [col for col in df.columns if col not in og_list]

new_cols

new_df = df[new_cols]

new_df.head()


###############################################
# ÖDEV 2: Fonksiyonlara Özellik Eklemek.
###############################################

# Görev: cat_summary() fonksiyonuna 1 özellik ekleyiniz. Bu özellik argumanla biçimlendirilebilir olsun.
# Not: Var olan özelliği de argumandan kontrol edilebilir hale getirebilirsiniz.


# Fonksiyona arguman ile biçimlendirilebilen bir özellik eklemek ne demek?
# Örnek olarak aşağıdaki check_df fonksiyonuna argumanla biçimlendirilebilen 2 özellik eklenmiştir.
# Bu özelliler ile tail fonksiyonunun kaç gözlemi göstereceği ve quantile değerlerinin gösterilip gösterilmeyeceği
# fonksiyona özellik olarak girilmiştir ve bu özellikleri kullanıcı argumanlarla biçimlendirebilmektedir.

# ÖNCESİ
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


# SONRASI
def check_df(dataframe, head=5, tail=5, quan=False):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    if quan:
        print("##################### Quantiles #####################")
        print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

import pandas as pd
df = pd.read_csv("2. Hafta/Ödevler/titanic.csv")
check_df(df, head=3, tail=3)

###############################################
# Ödev 2 Çözüm
###############################################

import matplotlib.pyplot as plt
import seaborn as sns

# ÖNCESİ
def cat_summary(dataframe, col_name, plot=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(df, "Survived")

# SONRASI

def cat_summary(dataframe, col_name, plot=False, missingv=False, val_counts=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

    if missingv:
        print(col_name + "'s column missing values: ", dataframe[col_name].isnull().sum())
        print("##########################################")

    if val_counts:
        print(col_name + "'s value counts\n", dataframe[col_name].value_counts())
        print("##########################################")


cat_summary(df, "Cabin", missingv=True, val_counts=True, plot=True)


###############################################
# ÖDEV 3: Docstring.
###############################################
# Aşağıdaki fonksiyona 4 bilgi (uygunsa) barındıran numpy tarzı docstring yazınız.
# (task, params, return, example)
# cat_summary()

def cat_summary(dataframe, col_name, plot=False, missingv=False, val_counts=False):
    """

        Dataframe içerisinde istenen kolondaki kategorik değişkenlere ait eleman sayısının/sayılarının
        toplam eleman sayısına yüzde cinsinden oranını, bu analize ait grafiği, kolondaki null değerlerin sayısını
        (eğer mevcutsa) ve her bir unique değerin kaç kez kullanıldığını gösteren konsol çıktısı olarak verir.


        Parameters
        ----------
        dataframe: dataframe
                Üzerinde analiz yapılacak dataframe
        col_name: str
                Dataframe içinde analizinin yapılması istenen sütun ismi
        plot: bool, optional, default: False
                Grafik çıktısını aktifleştiren bool parametresi
        missingv: bool, optional, default: False
                Null value analizini aktifleştiren bool parametresi
        val_counts: bool, optional, default: False
                Unique değerin kaç kez kullanıldığını gösteren bool parametresi

        Examples
        -------
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            df = pd.read_csv("2. Hafta/Ödevler/titanic.csv")
            cat_summary(df, "Cabin", missingv=True, val_counts=True)

        """

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

    if missingv:
        print(col_name + "'s column missing values: ", dataframe[col_name].isnull().sum())
        print("##########################################")

    if val_counts:
        print(col_name + "'s value counts\n",dataframe[col_name].value_counts())
        print("##########################################")





