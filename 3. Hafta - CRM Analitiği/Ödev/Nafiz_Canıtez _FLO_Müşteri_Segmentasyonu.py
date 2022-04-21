##################################################
# Görev 1: Veriyi Anlama ve  Hazırlama
##################################################
import pandas as pd
import datetime as dt

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# ADIM 1 #

flo = pd.read_csv(r"C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\3. Hafta\FLO_RFM_Analizi\flo_data_20K.csv")
df = flo.copy()

# ADIM 2 #

df.head(10)

df.columns

df.describe().T

df.isnull().sum()

df.info()

# ADIM 3 #

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# ADIM 4 #

df.info()

cols_date = df.columns[3:7]

df[cols_date] = df[cols_date].apply(pd.to_datetime)

df.info()

# ADIM 5 #

df.groupby('order_channel').agg({'master_id': lambda master_id: len(master_id),
                                 'order_num_total': lambda order_num_total: order_num_total.mean(),
                                 'customer_value_total': lambda customer_value_total: customer_value_total.mean()})

# ADIM 6 #

df.sort_values(by="customer_value_total", ascending=False).head(10)

# ADIM 7 #

df.sort_values(by="order_num_total", ascending=False).head(10)["master_id"]

# ADIM 8 #

print("Datadaki Eksik Veriler :\n", df.isnull().sum())


def Seg_Prep(dataframe):
    print("Data Info\n")
    print(dataframe.info())
    print("############################\n")

    print("Datadaki Eksik Veriler :\n", dataframe.isnull().sum())
    print("############################\n")

    cols_date = dataframe.columns[3:7]
    dataframe[cols_date] = dataframe[cols_date].apply(pd.to_datetime)
    print("Tarih Kolonları : ", [col for col in dataframe.columns if dataframe[col].dtype == 'datetime64[ns]'])
    print("############################\n")

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe[
        "customer_value_total_ever_online"]

    print("Alışveris Kanallarına Göre Dağılımlar\n",
          dataframe.groupby('order_channel').agg({'master_id': lambda master_id: len(master_id),
                                                  'order_num_total': lambda order_num_total: order_num_total.mean(),
                                                  'customer_value_total': lambda
                                                      customer_value_total: customer_value_total.mean()}))
    print("############################\n")

    print("En Fazla Kazanç Sağlanan Müşteri Id'leri :\n",
          dataframe.sort_values(by="customer_value_total", ascending=False).head(10)["master_id"])
    print("############################\n")

    print("En Fazla Satış Yapılan Müşteri Id'leri :\n",
          dataframe.sort_values(by="order_num_total", ascending=False).head(10)["master_id"])
    print("############################\n")


df = flo.copy()
Seg_Prep(df)

##################################################
# Görev 2: RFM Metriklerinin Hesaplanması
##################################################

## Adım 1 ##

# Recency: Müşterinin son alışveriş tarihinden, analiz yaptığımız tarihe olan zaman dilimidir.
# Frequency: Müşterinin firmadan toplamda kaç adet alışveriş yaptığını ifade eder.
# Monetary: Müşterinin aldığı tüm ürünlerin toplam fiyatıdır.

## Adım 2 & 3 ##

df.info()

df["last_order_date"].max()  # Timestamp('2021-05-30 00:00:00')

today_date = dt.datetime(2021, 5, 31)

rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                   'order_num_total': lambda order_num_total: order_num_total,
                                   'customer_value_total': lambda customer_value_total: customer_value_total})
rfm.head()

df[df["master_id"] == "00034aaa-a838-11e9-a2fc-000d3a38a36f"]

## Adım 4 ##

rfm.columns = ['recency', 'frequency', 'monetary']

rfm.head()

##################################################
# Görev 3:  RF Skorunun Hesaplanması
##################################################

### Adım 1 & 2 ###

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

### Adım 3 ###

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

rfm.head()

##################################################
# Görev 4:  RF Skorunun Segment Olarak Tanımlanması
##################################################

#### Adım 1 ####

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

#### Adım 2 ####

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

rfm

##################################################
# Görev 5:  Aksiyon Zamanı !
##################################################

##### Adım 1 #####

rfm.groupby('segment').agg({'recency': 'mean',
                            'frequency': 'mean',
                            'monetary': 'mean'})

##### Adım 2.a #####

df_rfm = pd.merge(df, rfm, on="master_id")
adim_2a = df_rfm[(df_rfm['segment'] == 'champions')
                 | (df_rfm['segment'] == 'loyal_customers')
                 & (df_rfm['monetary'] > 250)
                 & (df_rfm["interested_in_categories_12"].str.contains('KADIN'))]

adim_2a["master_id"].to_csv("adim_2a.csv")
rfm
##### Adım 2.b #####

df_rfm

adim_2b = df_rfm[((df_rfm["segment"] == "cant_loose")
                  | (df_rfm["segment"] == "about_to_sleep")
                  | (df_rfm["segment"] == "new_customers"))
                  & ((df_rfm["interested_in_categories_12"].str.contains('ERKEK'))
                  | (df_rfm["interested_in_categories_12"].str.contains('COCUK')))]

adim_2b["master_id"].to_csv("adim_2b.csv")