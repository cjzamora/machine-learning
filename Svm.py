from pyspark import SparkConf
from pyspark import SparkContext

from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint

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
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile("/usr/local/Cellar/apache-spark/1.5.2/libexec/data/mllib/sample_svm_data.txt")
parsedData = data.map(parsePoint)

# Build the model
model = SVMWithSGD.train(parsedData, iterations=100)

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.map(lambda (v,p): p)
print trainErr.collect()

# Save and load model
# model.save(sc, "myModelPath")
# sameModel = SVMModel.load(sc, "myModelPath")