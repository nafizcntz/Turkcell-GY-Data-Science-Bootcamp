##################################################
# Görev 1:  Average Rating’i güncel yorumlara göre hesaplayınız ve var olan averageratingile kıyaslayınız.
##################################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

amazon = pd.read_csv(
    r"C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\5. Hafta\measurement_problems\datasets\amazon_review.csv")
df = amazon.copy()

# ADIM 1 #
df["overall"].mean()

# ADIM 2 #

df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date = df["reviewTime"].max()

df["day_diff"]

df["day_diff_2"] = (current_date - df["reviewTime"]).dt.days

q1, q2, q3 = df["day_diff_2"].quantile([.25, .5, .75])


# ADIM 3 #
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    q1, q2, q3 = dataframe["day_diff_2"].quantile([.25, .5, .75])
    return dataframe.loc[dataframe["day_diff_2"] <= q1, "overall"].mean() * w1 / 100 + \
           dataframe.loc[
               (dataframe["day_diff_2"] > q1) & (dataframe["day_diff_2"] <= q2), "overall"].mean() * w2 / 100 + \
           dataframe.loc[
               (dataframe["day_diff_2"] > q2) & (dataframe["day_diff_2"] <= q3), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff_2"] > q3), "overall"].mean() * w4 / 100


time_based_weighted_average(df)

# ADIM 4 #

df.loc[df["day_diff_2"] <= q1, "overall"].mean()  # 4.6957928802588995

df.loc[(df["day_diff_2"] > q1) & (df["day_diff_2"] <= q2), "overall"].mean()  # 4.636140637775961

df.loc[(df["day_diff_2"] > q2) & (df["day_diff_2"] <= q3), "overall"].mean()  # 4.571661237785016

df.loc[(df["day_diff_2"] > q3), "overall"].mean()  # 4.4462540716612375

# En yakın tarihte yorum yapanların ve yıldız verenlerin ortalaması daha yüksek.
# Buda ürünün son zamanlarda iyileştirilmiş olması veya insanların ihtiyacını daha iyi karşılıyor olabileceği anlamına gelebilir.


##################################################
# Görev 2:  Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.
##################################################

amazon = pd.read_csv(
    r"C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\5. Hafta\measurement_problems\datasets\amazon_review.csv")
df = amazon.copy()

## ADIM 1 ##
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]


## ADIM 2 ##
def score_up_down_diff(up, down):
    return up - down


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"],
                                                                 x["helpful_no"]), axis=1)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],
                                                                     x["helpful_no"]), axis=1)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                 x["helpful_no"]), axis=1)



## ADIM 3 ##

df.sort_values("wilson_lower_bound", ascending=False).head(20)

# wilson Lower Bound'a göre sıralama yaptığımızda yıldız sayısına bakılmaksızın yorumun faydasının ne kadar olduğu öncelenmiştir.
# Örneğin ilk sıralarda 1 yıldızlı yorum olsada bu yorum insanların bu ürünü alırkenki faydasına olduğundan ötürü ilk sıralardadır.
# Zaten helpful kolonuna baktığımızda en çok faydalı bulununan yorumlar olduğuda görülmektedir.











