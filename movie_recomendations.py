__author__ = 'pranavgoel'


from pyspark import SparkConf,SparkContext,SQLContext
import sys,unicodedata
from pyspark.sql.functions import levenshtein
from pyspark.mllib.recommendation import ALS
from pyspark.sql.types import StructType, StructField, StringType, IntegerType


conf = SparkConf().setAppName('Ratings')
sc = SparkContext(conf=conf)
assert sc.version >= '1.5.1'
sqlContext = SQLContext(sc)

def user_ratings_match(df_movies,dfUserRatings):

    #function to find the closest match to user input movie title based on levenshtein distance

    myRatings = dfUserRatings.join(df_movies).select('*',levenshtein(dfUserRatings.User_Title,df_movies.Movies_Title).alias('distance')).cache()


    myRatings_best_title_match = myRatings.groupBy('User_Title').agg({'distance':'min'}).withColumnRenamed('min(distance)','min_dis')

    join_condition = [myRatings.User_Title == myRatings_best_title_match.User_Title  ,myRatings.distance == myRatings_best_title_match.min_dis  ]

    myRatings_movie_id = myRatings_best_title_match.join(myRatings,join_condition).select('movie_id','User_Ratings').withColumnRenamed('User_Ratings','Rating')
    myRatings_user_id  = myRatings_movie_id.withColumn('user_id',myRatings_movie_id.Rating - myRatings_movie_id.Rating)

    return(myRatings_user_id)


def model(User_Ratings,Ratings,movies):

    User_Ratings = User_Ratings.map(lambda (movie_id,Rating,user_id): (user_id,movie_id,Rating))
    training_set = Ratings.union(User_Ratings)

    rank = 10
    numIterations = 10
    model = ALS.train(training_set,rank,numIterations)

    User_Rated_movie_id = User_Ratings.map(lambda (user_id,movie_id,Rating): movie_id ).collect()

    User_unrated_movies = movies.filter(lambda x: x[0] not in User_Rated_movie_id).map(lambda x: (0, x[0]))


    predictions = model.predictAll(User_unrated_movies).map(lambda prediction: (prediction.product,prediction.rating))

    print("Predictions=======================")
    print(predictions).take(100)

    return(predictions)

def main():


    homeDirectory = sys.argv[1]
    userRatings = sys.argv[2]
    output = sys.argv[3]

    ratings = sc.textFile(homeDirectory + "/ratings.dat")
    movies = sc.textFile(homeDirectory + "/movies.dat")
    ratings = ratings.map( lambda line: line.split("::")).map(lambda (user_id,movie_id,Rating,rating_timestamp): (int(user_id),movie_id,Rating))

    movies= movies.map(lambda line: line.split("::")).map(lambda (movie_id,movie_title,Genre): (movie_id, unicodedata.normalize('NFD',movie_title))).cache()

    schema_movies = StructType([
    StructField('movie_id', StringType(), False),StructField('Movies_Title', StringType(), False)
    ])

    df_movies = sqlContext.createDataFrame(movies,schema_movies).cache()

    user = sc.textFile(userRatings).map(lambda line: line.split(' ',1)).map(lambda (ratings,User_Title): (int(ratings),unicodedata.normalize('NFD',User_Title)))

    schema_User = StructType([
    StructField('User_Ratings', IntegerType(), False),StructField('User_Title', StringType(), False)
    ])

    dfUserRatings = sqlContext.createDataFrame(user,schema_User).cache()
    User_Ratings = user_ratings_match(df_movies,dfUserRatings).rdd

    print("User Ratings ------------------------------")
    print(User_Ratings).take(10)

    predictions = model(User_Ratings,ratings,movies)

    schema_recommendation = StructType([
    StructField('movie_id', StringType(), False),StructField('Rating', StringType(), False)
    ])

    df_predictions = sqlContext.createDataFrame(predictions,schema_recommendation)

    recomendations = df_predictions.join(df_movies,[df_predictions.movie_id == df_movies.movie_id]).select(df_movies.Movies_Title,df_predictions.Rating).orderBy(df_predictions.Rating.desc()).coalesce(1).limit(25)

    recomendations.rdd.map(lambda (Title,Ratings): u"Movie = %s Rating = %s" % (Title, Ratings)).saveAsTextFile(output)


if __name__ == "__main__":
    main()
