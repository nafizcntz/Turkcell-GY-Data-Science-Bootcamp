
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

from helpers.data_prep import *
from helpers.eda import *

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# train ve test setlerinin bir araya getirilmesi.
train = pd.read_csv("datasets/house_prices/train.csv")
test = pd.read_csv("datasets/house_prices/test.csv")
df = train.append(test).reset_index(drop=True)
df.head()



######################################
# EDA
######################################

check_df(df)

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)

######################################
# KATEGORIK DEGISKEN ANALIZI
######################################

for col in cat_cols:
    cat_summary(df, col)

for col in cat_but_car:
    cat_summary(df, col)

for col in num_but_cat:
    cat_summary(df, col)


######################################
# SAYISAL DEGISKEN ANALIZI
######################################

df[num_cols].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

for col in num_cols:
    num_summary(df, col, plot=True)

######################################
# TARGET ANALIZI
######################################

df["SalePrice"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99])

# target ile bagımsız degiskenlerin korelasyonları
def find_correlation(dataframe, numeric_cols, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "SalePrice":
            pass
        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


low_corrs, high_corrs = find_correlation(df, num_cols)


######################################
# DATA PREPROCESSING & FEATURE ENGINEERING
######################################


######################################
# RARE ENCODING
######################################

rare_analyser(df, "SalePrice", 0.01)
df = rare_encoder(df, 0.01)

drop_list = ["Street", "Utilities", "LandSlope", "PoolQC", "MiscFeature"]

cat_cols = [col for col in cat_cols if col not in drop_list]

for col in drop_list:
    df.drop(col, axis=1, inplace=True)


rare_analyser(df, "SalePrice", 0.01)

######################################
# LABEL ENCODING & ONE-HOT ENCODING
######################################

cat_cols = cat_cols + cat_but_car

df = one_hot_encoder(df, cat_cols, drop_first=True)

check_df(df)

# def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
#     original_columns = list(dataframe.columns)
#     dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
#     new_columns = [c for c in dataframe.columns if c not in original_columns]
#     return dataframe, new_columns


######################################
# MISSING_VALUES
######################################

missing_values_table(df)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0 and "SalePrice" not in col]
df[na_cols] = df[na_cols].apply(lambda x: x.fillna(x.median()), axis=0)

######################################
# OUTLIERS
######################################

for col in num_cols:
    print(col, check_outlier(df, col))


df["SalePrice"].describe().T

replace_with_thresholds(df, "SalePrice")


######################################
# TRAIN TEST'IN AYRILMASI
######################################

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

train_df.to_pickle("datasets/house_prices/prepared_data/train_df.pkl")
test_df.to_pickle("datasets/house_prices/prepared_data/test_df.pkl")

#######################################
# MODEL: Random Forests
#######################################

X = train_df.drop(['SalePrice', "Id"], axis=1)
# y = train_df["SalePrice"]
y = np.log1p(train_df['SalePrice'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=46)

rf_model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
y_pred = rf_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

y.mean()

y_pred = rf_model.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))


#######################################
# Model Tuning
#######################################

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 15],
             "n_estimators": [200, 500],
             "min_samples_split": [2, 5, 8]}

rf_model = RandomForestRegressor(random_state=42)
rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
rf_cv_model.best_params_


#######################################
# Final Model
#######################################

rf_tuned = RandomForestRegressor(**rf_cv_model.best_params_).fit(X_train, y_train)

y_pred = rf_tuned.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

y_pred = rf_tuned.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))


#######################################
# Feature Importance
#######################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_tuned, X_train, 20)


#######################################
# SONUCLARIN YUKLENMESI
#######################################

submission_df = pd.DataFrame()
submission_df['Id'] = test_df["Id"]

y_pred_sub = rf_tuned.predict(test_df.drop("Id", axis=1))
y_pred_sub = np.expm1(y_pred_sub)

submission_df['SalePrice'] = y_pred_sub

submission_df.head()

submission_df.to_csv('submission_rf.csv', index=False)

