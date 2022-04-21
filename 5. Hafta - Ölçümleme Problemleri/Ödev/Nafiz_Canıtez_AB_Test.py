##################################################
# Görev 1: Veriyi Hazırlama ve Analiz Etme
##################################################

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# ADIM 1 #
control = pd.read_excel(r"C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\5. Hafta\measurement_problems\datasets\ab_testing.xlsx",
                    sheet_name="Control Group")

test = pd.read_excel(r"C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\5. Hafta\measurement_problems\datasets\ab_testing.xlsx",
                    sheet_name="Test Group")

control = control[control.columns[0:4]]
test = test[test.columns[0:4]]


# ADIM 2 #
control.describe().T

test.describe().T
#mean ve medyanlarına bakılırsa çok yanlı bir data gibi durmuyor.


# ADIM 3 #
df = pd.concat([control, test], axis=0).reset_index(drop=True)


##################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
##################################################

## ADIM 1 ##

# H0 : M1 = M2 (Kontrol grubu ve test grubu satın alma ortalamaları arasında fark yoktur.)
# H1 : M1!= M2 (Kontrol grubu ve test grubu satın alma ortalamaları arasında fark vardır.)

## ADIM 2 ##

control["Purchase"].mean()
test["Purchase"].mean()


##################################################
# Görev 3:  Hipotez Testinin Gerçekleştirilmesi
##################################################

### ADIM 1 ###

# Normallik Varsayımı
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.
# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

test_stat, pvalue = shapiro(control["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value iki data içinde 0.05'den büyüktür.
# Bu yüzden H0 reddedilemez deriz.


# Varyans Homojenligi Varsayımı
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir
# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

test_stat, pvalue = levene(control["Purchase"],
                           test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value burada da 0.05 den büyük olduğundan H0 reddedilemez deriz.
# Normallik varsayımı sağlanıyor.


### ADIM 2 & 3 ###

# Varsayımlar sağlanıyor. Bağımsız iki örneklem t testi (parametrik test) kullanıyoruz.
# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

test_stat, pvalue = ttest_ind(control["Purchase"],
                              test["Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value yine 0.05'den büyük olduğundan ötürü H0 reddedilemez.
# Bu yüzden Control ve Test datalarındaki Purchase kolonunun ortalaması birbirine eşittir deriz.
# Yani istatistiki olarak aralarında bir farklılık yoktur.
# Görünen farklılık şans eseri oluşmuş olabilir deriz.


##################################################
# Görev 4:  Sonuçların Analizi
##################################################

# Başta hipotezimizi kurduk.
# Sonrasında varsayım kontollerini yaptık.
# Ardından normallik varsayımı sağlandığı için parametrik testi kullandık.
# p-value değerlerdirmesi yaparak H0 hipotezini reddedemedik.
#
# Yani iki data arasında kazanç sayılarının şans eseri oluştuğu görüldü.
# Bu yüzden iki grubada aynı önem verilerek beraberce kazanç arttırılması sağlanabilir.
# Çünkü birbirlerinden farklı olarak kazanç getirmediklerini istatistiki olarak göstermiş olduk.













