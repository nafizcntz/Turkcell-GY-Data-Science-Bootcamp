##################################################
# Görev 1:  Veriyi Hazırlama
##################################################

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# ADIM 1 #

flo = pd.read_csv(r"C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\3. Hafta\FLO_RFM_Analizi\flo_data_20K.csv")
df = flo.copy()


df.describe().T

# ADIM 2 #

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# ADIM 3 #

replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

df.describe().T


# ADIM 4 #

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


# ADIM 5 #

df.info()

cols_date = df.columns[3:7]

df[cols_date] = df[cols_date].apply(pd.to_datetime)

df.info()



##################################################
# Görev 2: CLTV Veri Yapısının Oluşturulması
##################################################

## Adım 1 ##

df.last_order_date.max()

today_date = dt.datetime(2021, 6, 1)

## Adım 2 ##

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df['recency_cltv_weekly'] = (df['last_order_date'] - df['first_order_date']).dt.days / 7
cltv_df['T_weekly'] = ((today_date - df['first_order_date']).dt.days) / 7
cltv_df['frequency'] = df['order_num_total']
cltv_df['monetary_cltv_avg'] = df['customer_value_total'] / df['order_num_total']
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
cltv_df



##################################################
# Görev 3:  BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’ninHesaplanması
##################################################

### Adım 1 ###
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

cltv_df["exp_sales_3_month"] = bgf.predict(12, cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])


cltv_df["exp_sales_6_month"] = bgf.predict(24, cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

cltv_df["exp_sales_3_month"].sort_values(ascending=False).head(10)
cltv_df["exp_sales_6_month"].sort_values(ascending=False).head(10)
# Herhangi fark yok.

plot_period_transactions(bgf)
plt.show()

### Adım 2 ###

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])

### Adım 3 ###

cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'],
                                              cltv_df['monetary_cltv_avg'],
                                              time=6,  # 6 aylık
                                              freq="W",  # T_weekly'nin frekans bilgisi.
                                              discount_rate=0.01)


scaler = MinMaxScaler()
cltv_df["scaled_cltv"] = scaler.fit_transform(cltv_df[["cltv"]])

cltv_df["cltv"].sort_values(ascending=False).head(20)




##################################################
# Görev 3:  BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’ninHesaplanması
##################################################

#### Adım 1 ####

cltv_df["segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])


#### Adım 2 ####

cltv_df.groupby("segment").agg(
    {"count", "mean", "sum"})
# Şuanki değerlere bakılırsa bence 4 segment dağılımı yeterli olmuş.

#### Adım 3 ####

# A segmenti bizle en çok ilişkisi olan müşterilerdir. Bu müşterileri hem kaybetmemek hemde daha çok satın almasını
# sağlamak üzere stratejiler geliştirilebilir. Bu müşterilere özel indirimler ve kampanyalar oluşturulabilir. Örneğin
# onlara bir gold üye adı altında farklı ayrıcalıklar tanınırsa eğer hem elde tutulması kolay hemde daha çok
# alışveriş yaptırılabilir.

# B segmenti, A segmentinden sonra gelen müşterilerdir. Bu müşterilerin daha çok alışveriş yapması için ve A
# segmentine çıkabilmeleri için A segmentine olan ayrıcalıkların bir kısmını tanıyıp, belli alışkanlıklar kazandırıp
# A segmenti çok daha cazibeli kılınabilir ve bu sayede kullanıcıya statü atlatırılabilir.