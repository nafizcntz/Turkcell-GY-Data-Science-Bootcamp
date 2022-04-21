###############################################
# ÖDEV 1: List Comprehension Applications
###############################################

###############################################
# Görev 1: car_crashes verisindeki numeric değişkenlerin isimlerini büyük harfe çeviriniz ve başına NUM ekleyiniz.
###############################################


import seaborn as sns
from matplotlib import pyplot as plt

df = sns.load_dataset("car_crashes")
df.columns

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

[sonuc if kosul else sonuc2  for col in df.columns  ]
["NUM_"+col.upper() if df[col].dtype!="O" else col.upper() for col in df.columns]


###############################################
# Görev 2: İsminde "no" BARINDIRMAYAN değişkenlerin isimlerininin SONUNA "FLAG" yazınız.
###############################################

# Notlar:
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
df.columns
[ col.upper()+"FLAG" if "NO" not in col.upper() else col.upper() for col in df.columns]
[ col.upper()+"FLAG" if "no" not in col else col.upper() for col in df.columns]


###############################################
# Görev 3: Aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin isimlerini seçerek yeni bir df oluşturunuz.
###############################################

# df.columns
# og_list = ["abbrev", "no_previous"]

# Notlar:
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

new_columns=[col for col in df.columns if col not in og_list]
df[new_columns].head()
df_new=df[new_columns]
df_new.head()


df[[col for col in df.columns if col not in og_list]].head()



#########################################################################################
##########################################################################################


###############################################
# ÖDEV 2: Fonksiyonlara Özellik Eklemek.
###############################################

# Görev: cat_summary() fonksiyonuna 1 özellik ekleyiniz. Bu özellik argumanla biçimlendirilebilir olsun.
# Not: Var olan özelliği de argumandan kontrol edilebilir hale getirebilirsiniz.


# Fonksiyona arguman ile biçimlendirilebilen bir özellik eklemek ne demek?
# Örnek olarak aşağıdaki check_df fonksiyonuna argumanla biçimlendirilebilen 2 özellik eklenmiştir.
# Bu özelliler ile tail fonksiyonunun kaç gözlemi göstereceği ve quantile değerlerinin gösterilip gösterilmeyeceği
# fonksiyona özellik olarak girilmiştir ve bu özellikleri kullanıcı argumanlarla biçimlendirebilmektedir.


# check_df ÖNCESİ
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


# check_df SONRASI
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
df = pd.read_csv("Turkcell/week2/titanic.csv")
check_df(df, head=3, tail=3, quan=True)


# check_df SONRASI2
def check_df(dataframe, head=5, tail=5, quan=False, upper=False):
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

    if upper:
        dataframe.columns = [x.upper() for x in dataframe.columns]
        return dataframe


df2=check_df(df, head=5, tail=5, quan=True, upper=True)
df2.columns
df2.head()




# cat_summary ONCESI
def cat_summary(dataframe, col_name, plot=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

df = pd.read_csv("Turkcell/week2/titanic.csv")
cat_summary(df, "Survived", plot=True)


def cat_summary(dataframe, col_name, plot=False, describe_stats=False, isnull_=False, yazdır=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
    if describe_stats:
        print(dataframe[col_name].describe().T)
    if isnull_:
        print("##########################################")
        print(f"{col_name.upper()} DEGISKENI ICIN NULL VERI SAYISI")
        print(dataframe[col_name].isnull().sum())
    if yazdır:
        print("##########################################")
        print("Bu fonknun işlevi kategorik değişkenleri incelemektir.")

df = pd.read_csv("Turkcell/week2/titanic.csv")
cat_summary(df, "Sex", plot=True, describe_stats=True, isnull_=True, yazdır=True)

#########################################################################################
##########################################################################################

###############################################
# ÖDEV 3: Docstring.
###############################################
# Aşağıdaki fonksiyona 4 bilgi (uygunsa) barındıran numpy tarzı docstring yazınız.
# (task, params, return, example)
# cat_summary()


# ORNEK1

def check_df(dataframe, head=5):
    """
    Prints what emerges from a quick glance at the dataframe

    Parameters
    ----------
        dataframe: dataframe
            Dataframe object
        head: int
            Integer is used to get the first n rows

    Returns
    -------
    None
        This function prints a summary of the DataFrame and returns None.

    Examples
    -------
    >>>import seaborn as sns
    >>>df = sns.load_dataset("iris")
    >>>check_df(df,10)
    ##################### Shape #####################
    (150, 5)
    ##################### Types #####################
    sepal_length    float64
    sepal_width     float64
    petal_length    float64
    petal_width     float64
    species          object
    dtype: object
    ##################### Head #####################
       sepal_length  sepal_width  petal_length  petal_width species
    0           5.1          3.5           1.4          0.2  setosa
    1           4.9          3.0           1.4          0.2  setosa
    2           4.7          3.2           1.3          0.2  setosa
    3           4.6          3.1           1.5          0.2  setosa
    4           5.0          3.6           1.4          0.2  setosa
    5           5.4          3.9           1.7          0.4  setosa
    6           4.6          3.4           1.4          0.3  setosa
    7           5.0          3.4           1.5          0.2  setosa
    8           4.4          2.9           1.4          0.2  setosa
    9           4.9          3.1           1.5          0.1  setosa
    ##################### Tail #####################
         sepal_length  sepal_width  petal_length  petal_width    species
    140           6.7          3.1           5.6          2.4  virginica
    141           6.9          3.1           5.1          2.3  virginica
    142           5.8          2.7           5.1          1.9  virginica
    143           6.8          3.2           5.9          2.3  virginica
    144           6.7          3.3           5.7          2.5  virginica
    145           6.7          3.0           5.2          2.3  virginica
    146           6.3          2.5           5.0          1.9  virginica
    147           6.5          3.0           5.2          2.0  virginica
    148           6.2          3.4           5.4          2.3  virginica
    149           5.9          3.0           5.1          1.8  virginica
    ##################### NA #####################
    sepal_length    0
    sepal_width     0
    petal_length    0
    petal_width     0
    species         0
    dtype: int64
    ##################### Quantiles #####################
                  0.00   0.05  0.50   0.95   0.99  1.00
    sepal_length   4.3  4.600  5.80  7.255  7.700   7.9
    sepal_width    2.0  2.345  3.00  3.800  4.151   4.4
    petal_length   1.0  1.300  4.35  6.100  6.700   6.9
    petal_width    0.1  0.200  1.30  2.300  2.500   2.5


   """
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

print(check_df.__doc__)

check_df(df, 5)



# ORNEK2
def cat_summary(dataframe, col_name, plot=False, cat_type=False):
    """

        Veri setindeki kategorik değişkenlerin özetini verir. plot True verilirse grafik çizilir.
        Kategorik değişkenlerin numeric mi yoksa kategorik mi olduğunu opsiyona bağlı öğrenebiliriz.

        Parameters
        ------
            dataframe: dataframe
                    Değişken isimleri alınmak istenilen dataframe
            col_name: list
                    Categorical değişkenlerin isimlerinn yer aldığı bir listedir.
            plot: bool, optional
                    True durumunda grafik çizilir.
            cat_type: bool, optional
                    Categorical type yazdırmak istersek bu özelliği açabiliriz.

        Returns
        ------

            None

            Her bir categorical değişken içinde yer alan her bir unique değer için raito yazdırır.

            Opsiyonel olarak grafik ve tip bilgisi verir.

        Examples
        ------
            df2 = pd.read_csv("titanic.csv")

            cat_cols = [col for col in df2.columns if df2[col].dtypes == "O"]

            num_but_cat = [col for col in df2.columns if df2[col].nunique() < 10 and df2[col].dtypes != "O"]

            cat_but_car = [col for col in df2.columns if df2[col].nunique() > 20 and df2[col].dtypes == "O"]

            cat_cols = cat_cols + num_but_cat

            cat_cols = [col for col in cat_cols if col not in cat_but_car ]

            for i in cat_cols:
                cat_summary(df2,i,plot=True,cat_type=True)

        """

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if cat_type:
        if dataframe[col_name].nunique() < 10 and dataframe[col_name].nunique() != "O":
            print("Numerical but categorical")
        else:
            print("Categorical gibi categorical")

    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(df, 'Sex', True, True)
print(cat_summary.__doc__)