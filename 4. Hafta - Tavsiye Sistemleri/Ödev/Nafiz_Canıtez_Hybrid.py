##################################################
# USER BASED RECOMMENDATION
##################################################

##################################################
# Görev 1: Veriyi Hazırlama
##################################################
import pandas as pd

# ADIM 1 #
movie = pd.read_csv(r'C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\4. Hafta\recommender_systems\datasets\movie_lens_dataset\movie.csv')
rating = pd.read_csv(r'C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\4. Hafta\recommender_systems\datasets\movie_lens_dataset\rating.csv')

# ADIM 2 #
df = movie.merge(rating, how="left", on="movieId")
df.head()

# ADIM 3 #
#comment_counts = pd.DataFrame(df["title"].value_counts())
#rare_movies = comment_counts[comment_counts["title"] <= 1000].index
#common_movies = df[~df["title"].isin(rare_movies)]

common_movies = df[df['title'].map(df['title'].value_counts()) >= 10000]

# ADIM 4 #
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

# ADIM 5 #

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv(
        r'C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\4. Hafta\recommender_systems\datasets\movie_lens_dataset\movie.csv')
    rating = pd.read_csv(
        r'C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\4. Hafta\recommender_systems\datasets\movie_lens_dataset\rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    common_movies = df[df['title'].map(df['title'].value_counts()) >= 1000]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df


##################################################
# Görev 2:  Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
##################################################

## ADIM 1 ##
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=7).values)

## ADIM 2 ##
random_user_df = user_movie_df[user_movie_df.index == random_user]

## ADIM 3 ##
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()


##################################################
# Görev 3:  Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişilmesi
##################################################

### ADIM 1 ###
movies_watched_df = user_movie_df[movies_watched]

### ADIM 2 ###
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

### ADIM 3 ###
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


##################################################
# Görev 4:  Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
##################################################

#### ADIM 1 ####
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

#### ADIM 2 ####
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

#### ADIM 3 ####
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

#### ADIM 4 ####
rating = pd.read_csv(r'C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\4. Hafta\recommender_systems\datasets\movie_lens_dataset\rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')


##################################################
# Görev 5:  Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 FilminTutulması
##################################################

##### ADIM 1 #####
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

##### ADIM 2 #####
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

##### ADIM 3 #####
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

##### ADIM 4 #####
movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]])


##################################################
# ITEM BASED RECOMMENDATION
##################################################

##################################################
# Görev 1:  Kullanıcının izlediği en son ve en yüksek puan verdiği filme göre item-based öneri yapınız.
##################################################

# ADIM 1 #
import pandas as pd
#pd.set_option('display.max_columns', 500)
movie = pd.read_csv(r'C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\4. Hafta\recommender_systems\datasets\movie_lens_dataset\movie.csv')
rating = pd.read_csv(r'C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\4. Hafta\recommender_systems\datasets\movie_lens_dataset\rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

# ADIM 2 #
ru_movieID = df[(df["userId"] == random_user) & (df["rating"] == 5)].sort_values("timestamp", ascending=False)["movieId"].to_list()[0]

# ADIM 3 #
movie_title = df.loc[(df.movieId == ru_movieID), ["title"]].values[0]

item_movie_df = user_movie_df[movie_title[0]]

# ADIM 4 #
new_corr_df = user_movie_df.corrwith(item_movie_df).sort_values(ascending=False)

# ADIM 5 #
new_corr_df[1:6]








