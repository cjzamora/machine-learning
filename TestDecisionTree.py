# system
import os
import sys
import string
from time import time

# spark runtime
from pyspark import SparkConf
from pyspark import SparkContext

# spark mllib
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel

# numpy
from numpy import array

 # create spark configuration
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

# run spark context
sc = SparkContext(conf=config)

data_file = "dataset/kddcup.data_10_percent"
raw_data = sc.textFile(data_file)

print "Train data size is {}".format(raw_data.count())

test_data_file = "dataset/kddcup.newtestdata_10_percent_unlabeled"
test_raw_data = sc.textFile(test_data_file)

print "Test data size is {}".format(test_raw_data.count())

csv_data = raw_data.map(lambda x: x.split(","))
test_csv_data = test_raw_data.map(lambda x: x.split(","))

protocols = csv_data.map(lambda x: x[1]).distinct().collect()
services = csv_data.map(lambda x: x[2]).distinct().collect()
flags = csv_data.map(lambda x: x[3]).distinct().collect()

def create_labeled_point(line_split):
    # leave_out = [41]
    clean_line_split = line_split[0:41]

    # convert protocol to numeric categorical variable
    try: 
        clean_line_split[1] = protocols.index(clean_line_split[1])
    except:
        clean_line_split[1] = len(protocols)

    # convert service to numeric categorical variable
    try:
        clean_line_split[2] = services.index(clean_line_split[2])
    except:
        clean_line_split[2] = len(services)

    # convert flag to numeric categorical variable
    try:
        clean_line_split[3] = flags.index(clean_line_split[3])
    except:
        clean_line_split[3] = len(flags)

    # convert label to binary label
    attack = 1.0
    if len(line_split) >= 42 and line_split[41]=='normal.':
        attack = 0.0

    return LabeledPoint(attack, array([float(x) for x in clean_line_split]))

training_data = csv_data.map(create_labeled_point)
test_data = test_csv_data.map(create_labeled_point)

# Build the model
t0 = time()
tree_model = DecisionTree.trainClassifier(training_data, numClasses=2, 
                                          categoricalFeaturesInfo={1: len(protocols), 2: len(services), 3: len(flags)},
                                          impurity='gini', maxDepth=4, maxBins=100)
tt = time() - t0

print "Classifier trained in {} seconds".format(round(tt,3))

predictions = tree_model.predict(test_data.map(lambda p: p.features))
labels_and_preds = test_data.map(lambda p: p.label).zip(predictions)

t0 = time()
test_accuracy = labels_and_preds.filter(lambda (v, p): v == p).count() / float(test_data.count())
tt = time() - t0

print "Prediction made in {} seconds. Test accuracy is {}".format(round(tt,3), round(test_accuracy,4))

print "Learned classification tree model:"
print tree_model.toDebugString()
print labels_and_preds.collect()