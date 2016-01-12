from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from numpy import array

# Create a spark configuration
conf = SparkConf()

# set client
conf.setMaster('local')
# set app name
conf.setAppName("Some spark")
# spark config
conf.set("spark.cores.max", "1")
# spak config
conf.set("spark.executor.memory", "1g")

# Create spark context
sc = SparkContext(conf=conf)

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]

    return LabeledPoint(values[0], values[1:])

data = sc.textFile("/usr/local/Cellar/apache-spark/1.5.2/libexec/data/mllib/sample_svm_data.txt")
parsedData = data.map(parsePoint)

# Build the model
model = LogisticRegressionWithSGD.train(parsedData)

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())

print("Training Error = " + str(trainErr))