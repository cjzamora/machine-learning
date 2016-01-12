# -->

# system
import sys

# formatters
import string
import json

# Spark config and context
from pyspark import SparkConf
from pyspark import SparkContext

# Spark MLlib
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.classification import LogisticRegressionWithSGD

# numpy array / vectors
from numpy import array

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

# create hash data
def createHashData(rdd):
    original = rdd.map(lambda line : line.split(", "))
    # load up the json string
    data = rdd.map(lambda line : line.split(", ")).collect();

    def fn(line):
        label = 0.0

        if line[9] == 'title':
            label = 1.0

        return (label, data[0:9])

    # create paired data
    data_pared = original.map(fn)

    print data_pared

    htf = HashingTF(100)

    # hash data
    data_hashed = data_pared.map(lambda (label, f) : LabeledPoint(label, htf.transform(f)))

    return data_hashed

# load up training data set
def loadTrainingSet(path):
    # load the data set as an RDD
    data_raw = sc.textFile(path)

    # create hash data
    data_hashed = createHashData(rdd=data_raw)

    return data_hashed

# load up the given test data set
def loadTestSet(path):
    # load the data set as an RDD
    data_raw = sc.textFile(path)

    # create hash data
    data_hashed = createHashData(rdd=data_raw)

    return data_hashed

# calculate accuracy of predictions
def computeAccuracy(p, a):
    # filter correct predictions between predicted and actual data
    correct = p.filter(lambda (predicted, actual) : predicted == actual)

    # calculate accuracy
    return (correct.count() / float(a.count())) * 100

# compute mean squared error (mse)
# to see or evaluate goodness of fit
def computeMSE(rdd):
    return (
        rdd
        # map the data
        .map(lambda (v, p) : (v - p)**2)
        # reduce
        .reduce(lambda x, y : x + y) / rdd.count()
    )

# get command line arguments
def getArguments(argv):
    # holds up the arguments
    args = {}

    # iterate on each arguments
    for i in argv:
        # get the parts
        parts = i.split("=")

        # if we have key + val pair
        if len(parts) > 1:
            # set key value pair
            args[parts[0][2:]] = parts[1]

    return args

# get command line arguments
arguments = getArguments(sys.argv);

# load the training data
training = loadTrainingSet(arguments['training'])

# train a logistic regression model
model = NaiveBayes.train(training)

print model

# load the actual test data
actual = loadTestSet(arguments['test'])

# compare and predict actual data
predictions = actual.map(lambda point : (model.predict(point.features), point.label))

print predictions.collect()

# compute accuracy
accuracy = computeAccuracy(predictions, actual)

print ("Accuracy: " + str(accuracy) + "%")

# compute mse
mse = computeMSE(predictions)

print ("Mean Squared Error: " + str(mse))