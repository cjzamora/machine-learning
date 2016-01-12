from pyspark import SparkConf
from pyspark import SparkContext

from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt

# Create spark configuration
config = (
    # Init object
    SparkConf()
    # set client
    .setMaster('local')
    # set app name
    .setAppName("Title Prediction")
    # set max cores
    .set("spark.cores.max", "1")
    # set max memory
    .set("spark.executor.memory", "1g")
)

# initialize spark context
sc = SparkContext(conf=config)

# Load and parse the data
data = sc.textFile("/usr/local/Cellar/apache-spark/1.5.2/libexec/data/mllib/kmeans_data.txt")
parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))

# Build the model (cluster the data)
clusters = KMeans.train(parsedData, 2, maxIterations=10,
        runs=10, initializationMode="random")

# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]

    print center
    
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

# Save and load model
# clusters.save(sc, "trained")
# sameModel = KMeansModel.load(sc, "trained")