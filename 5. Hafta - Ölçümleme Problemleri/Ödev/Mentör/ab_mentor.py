import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
from scipy.stats import shapiro
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest

#  Gorev 1: Veriyi Hazırlama ve Analiz Etme

Control_Group = pd.read_excel("datasets/ab_testing.xlsx",
                              sheet_name='Control Group')  # maximum bidding
Test_Group = pd.read_excel("datasets/ab_testing.xlsx", sheet_name='Test Group')

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


check_df(Control_Group)
check_df(Test_Group)


"""

- Veri setinde 40 gözlem ve 4 değişken vardır.
- Değişkenler:
  - Impression : görüntülenme sayısı
  - Click : tıklanma sayısı
  - Purchase : satın alınma sayısı
  - Earning : kazanç
- Boş değer yok.
"""

# Aykırı değerler için eşik değeri belirleme
def outlier_thresholds(dataframe, variable, low_quantile=0.05, up_quantile=0.95):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


# Değişkende herhangi bir aykırı değer olup olmadığını kontrol ediyor.
def has_outliers(dataframe, numeric_columns):
    for col in numeric_columns:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col, " : ", number_of_outliers, "outliers")

# Control Group un aykırılık incelemesi
for var in Control_Group:
    print(var, "has ", has_outliers(Control_Group, [var]), "Outliers")

# Test Group un aykırılık incelemesi
for var in Control_Group:
    print(var, "has ", has_outliers(Test_Group, [var]), "Outliers")


# Test ve Kontrol gruplarında herhangi bir aykırılığa rastlanılmadı.

# kontrol ve test grubunun birleştirilmesi
Control_Group["Group"] = "A"  # Maximum Bidding
Test_Group["Group"] = "B"  # Average Bidding

AB = Control_Group.append(Test_Group)
AB.head()

"""
AB teste için AB isminde yeni dataframe oluşturulmuştur. 
- Control değişkeni -> Kontrol grubu Purchase değerleri
- Test değişkeni ->Test grubu Purchase değerleri 
"""

#  Gorev 2: A/B Testinin Hipotezinin Tanımlanması

# Adım 1: Hipotezi tanımlayınız.

"""
HO: "Maximum Bidding" kampanyası sunulan Kontrol grubu ile "Average Bidding" kampanyası sunulan Test grubunun 
satın alma sayılarının ortalaması arasında istatistiksel olarak anlamlı farklılık yoktur.(M1=M2) (≥, ≤)

H1:** "Maximum Bidding" kampanyası sunulan Kontrol grubu ile "Average Bidding" kampanyası sunulan Test grubunun 
satın alma sayılarının ortalaması arasında istatistiksel olarak anlamlı farklılık vardır.(M1≠M2) (<, >)
"""

# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz.Görev 2:  A/B Testinin Hipotezinin Tanımlanması

print(" Mean of purchase of control group(A): %.3f" % AB[AB["Group"] == "A"]["Purchase"].mean(), "\n",
      "Mean of purchase of test group(B): %.3f" % AB[AB["Group"] == "B"]["Purchase"].mean())


print(" Mean of purchase of control group(A): %.3f" % AB[AB["Group"] == "A"]["Purchase"].median(), "\n",
      "Mean of purchase of test group(B): %.3f" % AB[AB["Group"] == "B"]["Purchase"].median())


# İki yöntemin ortalama değerlerine bakıldığında aralarındaki farklılık olduğu görülmektedir.
# Ortalama satın alma değeri test grubu(B) lehinedir.

# Görev 3:  Hipotez Testinin Gerçekleştirilmesi

def AB_Test(dataframe, group, target):
    # A ve B gruplarının ayrılması
    groupA = dataframe[dataframe[group] == "A"][target]
    groupB = dataframe[dataframe[group] == "B"][target]

    # Varsayım: Normallik
    # Shapiro-Wilks Test
    # H0: Örnek dağılımı ile teorik normal dağılım arasında istatistiksel olarak anlamlı bir fark yoktur! -False
    # H1: Örnek dağılımı ile teorik normal dağılım arasında istatistiksel olarak anlamlı bir fark vardır! -True
    # p-value 0.05 den küçük ise H0 reddedilir.
    ntA = shapiro(groupA)[1] < 0.05
    ntB = shapiro(groupB)[1] < 0.05

    if (ntA == False) & (ntB == False):  # "H0: Normal dağılım" sağlandıysa
        # Parametric Test
        # Varsayım: Varyans Homojenliği
        leveneTest = stats.levene(groupA, groupB)[1] < 0.05
        # H0: karşılaştırılan gruplar eşit varyansa sahiptir. - False
        # H1: karşılaştırılan gruplar eşit varyansa sahip değildir. - Ture
        if leveneTest == False:  # eşit varyansa sahiplerse
            # Homogeneity
            ttest = stats.ttest_ind(groupA, groupB, equal_var=True)[1]
            # H0: M1 == M2 - False
            # H1: M1 != M2 - True
        else:  # eşit varyansa sahip değillerse
            # Heterogeneous
            ttest = stats.ttest_ind(groupA, groupB, equal_var=False)[1]
            # H0: M1 == M2 - False
            # H1: M1 != M2 - True
    else:  # Normal dağılıma sahip değilse
        # Non-Parametric Test
        ttest = stats.mannwhitneyu(groupA, groupB)[1]
        # H0: M1 == M2 - False
        # H1: M1 != M2 - True

    # Sonuç
    temp = pd.DataFrame({"AB Hypothesis": [ttest < 0.05], "p-value": [ttest]})
    temp["Test Type"] = np.where((ntA == False) & (ntB == False), "Parametric", "Non-Parametric")
    temp["AB Hypothesis"] = np.where(temp["AB Hypothesis"] == False, "Fail to Reject H0", "Reject H0")
    temp["Comment"] = np.where(temp["AB Hypothesis"] == "Fail to Reject H0", "A/B groups are similar!",
                               "A/B groups are not similar!")

    if (ntA == False) & (ntB == False):
        temp["Homogeneity"] = np.where(leveneTest == False, "Yes", "No")
        temp = temp[["Test Type", "Homogeneity", "AB Hypothesis", "p-value", "Comment"]]
    else:
        temp = temp[["Test Type", "AB Hypothesis", "p-value", "Comment"]]

    return temp


AB_Test(AB, group="Group", target="Purchase")

# Görev 4:  Sonuçların Analizi

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

# Normallik testi ve varyans homojenliği testi sonucunda 2 varsayımın da sağlandığı görülmüştür.
# Bu sebeple "Bağımsız İki Örneklem T Testi" uygulanmıştır.

# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz

# → Kontrol grubu ile test grubunun ürün satın alma ortalamaları arasında istatistiksel olarak anlamlı bir farklılık bulunmamaktadır !!!
# Yani aslında ilk bakışta gördüğümüz ve daha iyi olduğunu düşündüğümüz kontrol grubunun yorumuna şans eseri ulaşmışız!
# Bir süre geçtikten sonra yeniden test yapılmasını öneririz.
# ya da
# Diğer mentriklerin de testi yapıldıktan sonra karara varılmasını öneririz.






#######################################
"""Website Click Through Rate (CTR)

-->Reklamı GÖREN kullanıcıların, reklamı ne sıklıkta TIKLADIKLARINI gösteren orandır.
-->Reklam Tıklanma Sayısı/ Reklam Gösterilme Sayısı
-->Örnek 5 tıklama, 100 gösterimde CTR= %5"""

AB.head()

control_CTR = AB.loc[AB["Group"] == "A", "Click"].sum() / AB.loc[AB["Group"] == "A", "Impression"].sum()
test_CTR = AB.loc[AB["Group"] == "B", "Click"].sum() / AB.loc[AB["Group"] == "B", "Impression"].sum()
print("Control_CTR: ", control_CTR, "\n", "test_CTR: ", test_CTR)
# İlk bakışta tıklama oranının kontrol grubu lehine olduğunu görüyoruz. Yani reklamı görüp de tıklayanların oranı mevcut sistemde daha iyi gibi görünüyor.
"""
Varsayım: n≥ 30 sağlandı.
Hipotezler
H0: Deneyin kullanıcı davranışına istatistiksel olarak anlamlı etkisi yoktur. (p_cont = p_test)
H1: Deneyin kullanıcı davranışına istatistiksel olarak anlamlı etkisi vardır. (p_cont ≠ p_test)
"""
click_count= AB.loc[AB["Group"] == "A", "Click"].sum() ,  AB.loc[AB["Group"] == "B", "Click"].sum()
impression_count= AB.loc[AB["Group"] == "A", "Impression"].sum(), AB.loc[AB["Group"] == "B", "Impression"].sum()
proportions_ztest(count=click_count, nobs=impression_count)
"""
Sonuç:
pval< 0.05 → Reject H0
Reklam teklifi yöntemleri incelendiğinde, bu yöntemlerin kullanıcı davranışına(tıklama) etkisi farklıdır. 
Ve bu farklılık mevcut durumdaki reklam teklif yöntemi lehinedir."""



"""Conversion Rate (Dönüşüm Oranı)

-->Dönüşüm oranı, dönüşüm sayısının toplam ziyaretçi sayısına bölünmesiyle elde edilir.
-->Purchase/Impression
-->Örneğin, bir e-ticaret sitesi bir ayda 200 ziyaretçi alırsa ve 50 satış varsa, dönüşüm oranı 50/200'e yani % 25'e eşit olur."""

control_CR = AB.loc[AB["Group"] == "A", "Purchase"].sum() / AB.loc[AB["Group"] == "A", "Impression"].sum()
test_CR = AB.loc[AB["Group"] == "B", "Purchase"].sum() / AB.loc[AB["Group"] == "B", "Impression"].sum()
print("Control_CR: ", control_CR, "\n", "test_CR: ", test_CR)
# İlk bakışta dönüşüm oranının kontrol grubu lehine olduğunu görüyoruz. Yani görüntüleyenlerin ne kadarı satın almış dediğimizde,
# mevcut durumdaki oran daha iyi çıkıyor.
# Ama bu durum istatistiksel olarak da anlamlı bir farklılık içeriyor mu? Yoksa tesadüfen mi oluştu?
"""
Varsayım: n≥ 30 sağlandı.
Hipotezler
H0: Deneyin kullanıcı davranışına istatistiksel olarak anlamlı etkisi yoktur. (p_cont = p_test)
H1: Deneyin kullanıcı davranışına istatistiksel olarak anlamlı etkisi vardır. (p_cont ≠ p_test)
"""
purchase_count= AB.loc[AB["Group"] == "A", "Purchase"].sum() ,  AB.loc[AB["Group"] == "B", "Purchase"].sum()
impression_count= AB.loc[AB["Group"] == "A", "Impression"].sum(), AB.loc[AB["Group"] == "B", "Impression"].sum()
proportions_ztest(count=purchase_count, nobs=impression_count)
"""
Sonuç:
pval< 0.05 → Reject H0
Reklam teklifi yöntemleri incelendiğinde, bu yöntemlerin kullanıcı davranışına(satın alma) etkisi farklıdır. 
Ve bu farklılık mevcut durumdaki reklam teklif yöntemi lehinedir."""

###Yorumumuzu yenileyecek olursa, yeni duruma geçiş için bir süre sonra yeniden test yapılması önerilir.
### Şu an karar alınması gerekirse mevcut durumda kalmak daha mantıklı olacaktır.