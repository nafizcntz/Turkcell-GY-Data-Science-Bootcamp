##################################################
# Görev 1: Veriyi Hazırlama
##################################################
# !pip install mlxtend
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# ADIM 1 #

OR2 = pd.read_excel(r"C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\4. Hafta\recommender_systems\datasets\online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = OR2.copy()
df.head()

# ADIM 2 #
df = df[~df["StockCode"].str.contains("POST", na=False)]

# ADIM 3 #
df.dropna(inplace=True)

# ADIM 4 #
df = df[~df["Invoice"].str.contains("C", na=False)]

# ADIM 5 #
df = df[df["Price"] > 0]


# ADIM 6 #
df.describe().T

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

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")


def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[~dataframe["StockCode"].str.contains("POST", na=False)]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

##################################################
# Görev 2: Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme
##################################################

## ADIM 1 ##
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

## ADIM 2 ##
def create_rules(dataframe, id=True, country="Germany"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

df = OR2.copy()
df = retail_data_prep(df)
rules = create_rules(df)
rules

##################################################
# Görev 3: Sepet İçerisindeki Ürün Id’leriVerilen Kullanıcılara Ürün Önerisinde Bulunma
##################################################

### ADIM 1 ###

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

product_id = 71053
check_id(df, product_id)
df


### ADIM 2 ###

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


a_rec = arl_recommender(rules, 22629, 1)
check_id(df, int(a_rec[0]))

b_rec = arl_recommender(rules, 22467, 1)
check_id(df, int(b_rec[0]))

c_rec = arl_recommender(rules, 22077, 1)
check_id(df, int(c_rec[0]))