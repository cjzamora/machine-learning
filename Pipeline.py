from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import Row, SQLContext

import os
import sys
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *

from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from numpy import array

# Assign resources to the application
conf = SparkConf()
conf.setMaster('local')
conf.setAppName('pysparkregression')
conf.set("spark.cores.max", "4")
conf.set("spark.executor.memory", "4g")

sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

LabeledDocument = Row("BuildingID", "SystemInfo", "label")

# data = sqlContext.createDataFrame([
#         ("6/1/2013",  "0:00:01",  66, 58, 13, 20, 4),
#         ("6/2/2013",  "1:00:01",  69, 68, 3,  20, 17), 
#         ("6/3/2013",  "2:00:01",  70, 73, 17, 20, 18), 
#         ("6/4/2013",  "3:00:01",  67, 63, 2,  23, 15), 
#         ("6/5/2013",  "4:00:01",  68, 74, 16, 9,  3), 
#         ("6/6/2013",  "5:01:01",  67, 56, 13, 28, 4), 
#         ("6/7/2013",  "6:00:01",  70, 58, 12, 24, 2),
#         ("6/8/2013",  "7:00:01",  70, 73, 20, 26, 16), 
#         ("6/9/2013",  "8:00:01",  66, 69, 16, 9,  9), 
#         ("6/10/2013", "9:00:01",  65, 57, 6,  5,  12), 
#         ("6/11/2013", "10:01:01", 67, 70, 10, 17, 15), 
#         ("6/12/2013", "11:00:01", 69, 62, 2,  11, 7), 
#         ("6/13/2013", "12:01:01", 69, 73, 14, 2,  15), 
#         ("6/14/2013", "13:00:01", 65, 61, 3,  2,  6), 
#         ("6/15/2013", "14:00:01", 67, 59, 19, 22, 20), 
#         ("6/16/2013", "15:00:01", 65, 56, 19, 11, 8), 
#         ("6/17/2013", "16:00:01", 67, 57, 15, 7,  6) 
#     ], ["Date", "Time", "TargetTemp", "ActualTemp", "System", "SystemAge", "BuildingID"])

def parseDocument(line):
    values = [str(x) for x in line.split(',')]
    if (values[3] > values[2]):
        hot = 1.0
    else:
        hot = 0.0        

    textValue = str(values[4]) + " " + str(values[5])

    return LabeledDocument((values[6]), textValue, hot)

data = sc.textFile("dataset/features.csv")
documents = data.filter(lambda s: "Date" not in s).map(parseDocument)
training = documents.toDF()

tokenizer = Tokenizer(inputCol="SystemInfo", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr        = LogisticRegression(maxIter=10, regParam=0.01)
pipeline  = Pipeline(stages=[tokenizer, hashingTF, lr])

model = pipeline.fit(training)

training.show()

# SystemInfo here is a combination of system ID followed by system age
Document = Row("id", "SystemInfo")
test = sc.parallelize([
        (1L, "20 26"),
        (2L, "4 15"),
        (3L, "16 9"),
        (4L, "9 22"),
        (5L, "17 10"),
        (6L, "7 22")]) \
    .map(lambda x: Document(*x)).toDF() 

test.show()

# Make predictions on test documents and print columns of interest
prediction = model.transform(test)
selected = prediction.select("SystemInfo", "prediction", "probability")
for row in selected.collect():
    print row