# -->

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

# Decision Tree Class
class DecisionTreeTraining:
    # constructor
    def __init__(self):
        # get command line arguments
        self.options = self.getCommandLineArgs(sys.argv)

        # create spark context
        self.sc = self.createSparkContext()

        # run the training
        return self.run()

    # run spark context
    def createSparkContext(self):
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
        return SparkContext(conf=config)

    # load data set
    def loadDataSet(self):
        # load up the training data set
        if 'training' in self.options:
            # try to load training set
            self.training = self.sc.textFile(self.options['training'])

            # print total training data
            try:
                # count the training data set
                print "Training size: {}".format(self.training.count())
            except:
                # throw an error
                print "Unable to read training dataset from path '{}'".format(self.options['training'])

                # exit
                sys.exit(1)
        else:
            # throw an error message
            print "Unable to load training data set. --training=[path] to set data set location."

            # exit
            sys.exit(1)

        # load up the testing data set
        if 'testing' in self.options:
            # try to load training set
            self.testing  = self.sc.textFile(self.options['testing'])

             # print total training data
            try:
                # count the training data set
                print "Testing size: {}".format(self.testing.count())
            except:
                # throw an error
                print "Unable to read testing dataset from path '{}'".format(self.options['training'])

                # exit
                sys.exit(1)
        else:
            # throw an error message
            print "Unable to load testing data set. --testing=[path] to set data set location."

            # exit
            sys.exit(1)

    # run the traning
    def run(self):
        # load data set
        self.loadDataSet()

        # format the data
        self.training_data = self.training.map(lambda x : x.split(", "))
        # format the data
        self.testing_data  = self.testing.map(lambda x : x.split(", "))

        # get the tags
        self.tags    = self.training_data.map(lambda x : x[0]).distinct().collect()
        # get the numeric tags
        self.numeric = self.training_data.map(lambda x : x[1]).distinct().collect()
        # get x pos
        self.x       = self.training_data.map(lambda x : x[2]).distinct().collect()
        # get y post
        self.y       = self.training_data.map(lambda x : x[3]).distinct().collect()

        # set global tag scope (annoying)
        tags    = self.tags
        numeric = self.numeric
        x       = self.x
        y       = self.y

        # create labeled point data
        def fn(line):
            # get line, ignore index 9
            clean_line = line[0:9]

            # convert tag to numeric categorical value
            try:
                clean_line[0] = tags.index(clean_line[0])
            except:
                clean_line[0] = len(tags)

            # convert tag to numeric categorical value
            try:
                clean_line[1] = numeric.index(clean_line[1])
            except:
                clean_line[1] = len(numeric)

            # convert tag to numeric categorical value
            try:
                clean_line[2] = x.index(clean_line[2])
            except:
                clean_line[2] = len(x)

            # convert tag to numeric categorical value
            try:
                clean_line[3] = y.index(clean_line[3])
            except:
                clean_line[3] = len(y)

            # set label to binary label
            label = 0.0

            # if it's a title
            if line[9] == "title":
                label = 1.0

            return LabeledPoint(label, array([float(z) for z in clean_line]))

        # create labeled data
        self.training_labeled = self.training_data.map(fn)
        # create labeled data
        self.testing_labeled  = self.testing_data.map(fn)

        # train the classifier
        self.trainClassifier()

    # train the classifier
    def trainClassifier(self):
        # get the current time
        current = time()

        # get the tags
        tags    = self.tags
        numeric = self.numeric
        x       = self.x
        y       = self.y

        # get the training data
        training_data = self.training_labeled

        # start training the tree model
        self.tree_model = DecisionTree.trainClassifier(
                            training_data,
                            numClasses=4,
                            categoricalFeaturesInfo={0 : len(tags), 1 : len(numeric), 2 : len(x), 3 : len(y)},
                            impurity="gini",
                            maxDepth=5,
                            maxBins=1000)

        print self.tree_model

        # total time
        total = time() - current

        print "Classifier trained in {} seconds.".format(round(total, 3))

        # start evaluating the model
        self.evaluate()

    # evaluate the model
    def evaluate(self):
        # get the predictions
        self.predictions = self.tree_model.predict(self.testing_labeled.map(lambda p : p.features))
        # compare labels and predictions
        self.labels_and_preds = self.testing_labeled.map(lambda p : (p.label, p.features)).zip(self.predictions)

        # get the labeled testing data
        testing_labeled = self.testing_labeled

        # get the current time
        current = time()
        # caclulate the accuracy
        test_accuracy = self.labels_and_preds.filter(lambda (v, p): v[0] == p).count() / float(testing_labeled.count())
        # get the total time
        total = time() - current

        # get the results
        self.results = self.labels_and_preds.filter(lambda (v, p) : v[0] == p).collect()

        # calculate mean squared error
        mse = self.labels_and_preds.map(lambda (v, p): (v[0] - p) * (v[0] - p)).sum() / float(testing_labeled.count())

        # print predictions time
        print "Prediction made in {} seconds. Test accuracy is {}%".format(round(total, 3), round(test_accuracy,4))
        # print mse
        print "Mean Squared Error {}".format(mse)

        # print classification tree model
        print "Learned classification tree model:"
        print self.tree_model.toDebugString()

        # get the results
        self.getResults()

    # get possible results
    def getResults(self):
        # collect the data
        data = self.results

        # if there are no predictions
        if(len(data) <= 0):
            print "Prediction Results: No results for the given testing data :("

            # exit
            sys.exit(1)

        print "Prediction Results: "

        # iterate on each data
        for i in data:
            # get the features
            features = i[0][1]

            print (
                "Possible title features: lbl({}), tag({}), x({}), y({}), offset_x({}), offset_y({}), width({}), height({}), text_length({})"
                .format(
                    str(i[0][0]),
                    str(features[0]),
                    str(features[1]),
                    str(features[2]),
                    str(features[3]),
                    str(features[4]),
                    str(features[5]),
                    str(features[6]),
                    str(features[7])))

        # save the model
        try:
            self.saveModel()
        except:
            pass

    # save the model
    def saveModel(self):
        # save the model to the given path
        self.tree_model.save(self.sc, "trained")

        # re-load the saved model
        self.tree_model = DecisionTreeModel.load(self.sc, "trained")

        # re-evaluate
        self.evaluate()

    # get command line arguments
    def getCommandLineArgs(self, argv):
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


# run the training
DecisionTreeTraining();