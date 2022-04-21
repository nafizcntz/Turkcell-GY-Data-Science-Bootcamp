############################################
# Proje: Association Rule Learning Recommender
############################################

# Amaç: Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

# Aşağıda 3 farklı kullanıcının sepet bilgileri verilmiştir.
# Bu sepet bilgilerine en uygun ürün önerisini yapınız. Ürün önerileri 1 tane ya da 1'den f
# azla olabilir. Karar kurallarını 2010-2011 Germany müşterileri üzerinden türetiniz.

# Kullanıcı 1 ürün id'si: 21987
# Kullanıcı 2 ürün id'si: 23235
# Kullanıcı 3 ürün id'si: 22747


############################################
# Gerekli Kütüphane ve Fonksiyonlar
############################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
#!pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

############################################
# Görev 1: Veriyi Hazırlama
############################################

# Adım 1: Online Retail II veri setinden 2010-2011 sheet’ini okutunuz.
df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()

# Adım 2: StockCode’u POST olan gözlem birimlerini drop ediniz. (POST her faturaya eklenen bedel, ürünü ifade etmemektedir.)
df.drop(df[df["StockCode"] == "POST"].index, inplace=True)


# Adım 3: Boş değer içeren gözlem birimlerini drop ediniz.
df.dropna(inplace=True)


# Adım 4: Invoice içerisinde C bulunan değerleri veri setinden çıkarınız. (C faturanın iptalini ifade etmektedir.)
df = df[~df["Invoice"].str.contains("C", na=False)]


# Adım 5: Price değeri sıfırdan küçük olan gözlem birimlerini filtreleyiniz.
df = df[df["Price"] > 0]
df = df[df["Quantity"] > 0]

# Adım 6: Price ve Quantity değişkenlerinin aykırı değerlerini inceleyiniz, gerekirse baskılayınız.

# df["Quantity"].describe([0.9,0.95,0.99,.9977,.999])
# sns.boxplot(df["Quantity"])
# plt.show()
# outlier_thresholds(df, "Quantity")

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")


# Bonus : Fonksiyonlaştırma

# def retail_data_prep(dataframe):
#     dataframe.drop(dataframe[dataframe["StockCode"]=="POST"].index, inplace=True)
#     dataframe.dropna(inplace=True)
#     dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
#     dataframe = dataframe[dataframe["Quantity"] > 0]
#     dataframe = dataframe[dataframe["Price"] > 0]
#     replace_with_thresholds(dataframe, "Quantity")
#     replace_with_thresholds(dataframe, "Price")
#     return dataframe
#
# df = retail_data_prep(df)
#
# df.head()


############################################
# Görev 2: Germany Müşterileri Üzerinden Birliktelik Kuralları Üretiniz
############################################

# Adım 1: Aşağıdaki gibi fatura ürün pivot table’i oluşturacak create_invoice_product_df fonksiyonunu tanımlayınız.

# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET...
# Invoice
# 536370                              0                                 1                       0..
# 536852                              1                                 0                       1..
# 536974                              0                                 0                       0..
# 537065                              1                                 0                       0..
# 537463                              0                                 0                       1..

# sutunlara olası tüm ürünler koyulacak
# sepette ürün varsa 1 yoksa 0 değeri atanacak

# Alman müşterilerin seçilmesi
df = df[df['Country'] == "Germany"]

# invoice_product_df'in oluşturulması

# invoice_product_df = df.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


invoice_product_df = create_invoice_product_df(df)
invoice_product_df.head()


# Adım 2: Kuralları oluşturacak create_rules fonksiyonunu tanımlayınız ve Alman müşteriler için kurallarını bulunuz.

frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)


def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

rules_grm = create_rules(df, country="Germany")
rules_grm.head()


############################################
# Görev 3: Sepet İçerisindeki Ürün Id’leri Verilen Kullanıcılara Ürün Önerisinde Bulunma
############################################

# Adım 1: check_id fonksiyonunu kullanarak verilen ürünlerin isimlerini bulunuz.

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)



check_id(df, 21987)
# (PACK OF 6 SKULL PAPER CUPS)
check_id(df, 23235)
# (STORAGE TIN VINTAGE LEAF)
check_id(df, 22747)
# (POPPY'S PLAYHOUSE BATHROOM)


#Adım 2: arl_recommender fonksiyonunu kullanarak 3 kullanıcı için ürün önerisinde bulununuz.

# Kullanıcı 1 ürün id'si: 21987
# Kullanıcı 2 ürün id'si: 23235
# Kullanıcı 3 ürün id'si: 22747


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]


arl_recommender(rules_grm, 21987, 1)

arl_recommender(rules_grm, 23235, 1)

arl_recommender(rules_grm, 22747, 2)

# Adım 3: Önerilecek ürünlerin isimlerine bakınız.

check_id(df, arl_recommender(rules_grm, 21987, 1)[0])
check_id(df, arl_recommender(rules_grm, 23235, 1)[0])
check_id(df, arl_recommender(rules_grm, 22747, 1)[0])





