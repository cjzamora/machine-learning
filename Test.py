from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

# Create a spark configuration
conf = SparkConf()

# set client
conf.setMaster('local');
# set app name
conf.setAppName("Some spark");
# spark config
conf.set("spark.cores.max", "1");
# spak config
conf.set("spark.executor.memory", "1g")

# Create spark context
sc = SparkContext(conf=conf)

# Create a labeled point with a positive label and a dense feature vector.
pos = LabeledPoint(1.0, [1.0, 0.0, 3.0])

# Create a labeled point with a negative label and a sparse feature vector.
neg = LabeledPoint(0.0, SparseVector(3, [0, 2], [1.0, 3.0]))

print neg