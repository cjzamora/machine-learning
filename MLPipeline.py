# -->

# system
import os
import sys
import json
from time import time

# spark config and context
from pyspark import SparkConf
from pyspark import SparkContext

# spark sql context for dataframes
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql.types import *

# spark ml
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import Tokenizer

# spark mllib
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint

# numpy
from numpy import array

'''
MLPipeline.py
 
Copyright and license information can be found at LICENSE
distributed with this package.
'''
class MLPipelineTraining:
    '''
    Training data RDD

    @var : RDD
    '''
    training_data = []
    
    '''
    Testing data RDD

    @var : RDD
    '''
    testing_data  = []

    '''
    Command line options

    @var : dictionary
    '''
    options = {}

    '''
    Global Spark Configuration

    @var : SparkConf
    '''
    spark_config  = ()

    '''
    Global Spark context

    @var : SparkContext
    '''
    context = ()
    
    '''
    Global SQL Context

    @var : SQLContext
    '''
    sql_context = ()

    '''
    Log color definition

    @var : dictionary
    '''
    log_colors = {
        'HEADER'      : '\033[95m',
        'OKBLUE'      : '\033[94m',
        'OKGREEN'     : '\033[92m',
        'WARNING'     : '\033[93m',
        'FAIL'        : '\033[91m',
        'ENDC'        : '\033[0m',
        'BOLD'        : '\033[1m',
        'UNDERLINE'   : '\033[4m'
    }

    '''
    Construct

    @return : MLPipelineTraining
    '''
    def __init__(self):
        # set start time
        start = time()

        # get command line arguments
        MLPipelineTraining.options = self.getCommandLineArgs(sys.argv)

        # notify spark context creation
        self.success('Starting spark context ...')

        # start spark context
        try:
            # create spark context
            MLPipelineTraining.context = self.createSparkContext()
        except:
            # throw error message
            self.fail('Unable to create spark context ...')

            # exit
            sys.exit(1)

        # run the training pipeline process
        self.process()

        # set end time
        end = time() - start

        # notify end time
        self.success('ML Pipeline completed in {} seconds'.format(round(end,3)))

    '''
    Create spark context

    @return : SparkContext
    '''
    def createSparkContext(self):
        # create spark configuration
        MLPipelineTraining.spark_config = (
            # Init object
            SparkConf()
            # set client
            .setMaster('local')
            # set app name
            .setAppName('Title Prediction')
            # set max cores
            .set('spark.cores.max', '1')
            # set max memory
            .set('spark.executor.memory', '1g')
        )

        # run spark context
        sc = SparkContext(conf=MLPipelineTraining.spark_config)
        # initialize sql context
        sql_context = SQLContext(sc)

        return sc

    '''
    Start training and processing

    @return : MLPipelineTraining
    '''
    def process(self):
        # get options
        options = MLPipelineTraining.options

        # check training path option
        if not 'training' in options:
            # throw error
            self.fail('Training dataset path is not set, please set training path using --training=[path].')

            # exit
            sys.exit(1)

        # check testing path option
        if not 'testing' in options:
            # throw error
            self.fail('Testing dataset path is not set, please set testing path using --testing=[path].')

            # exit
            sys.exit(1)

        # load up training data set
        MLPipelineTraining.training_data = self.formatTrainingData(self.loadDataAsRDD(options['training'], 1))
        # load up testing data set
        MLPipelineTraining.testing_data  = self.formatTestingData(self.loadDataAsRDD(options['testing'], 0))

        # cache up the data
        MLPipelineTraining.training_data.cache()

        # get the pipeline model
        model = self.getPipeline(MLPipelineTraining.training_data.toDF())

        # start predicting results
        predictions = self.predict(model, MLPipelineTraining.testing_data.toDF())

        # notify success
        self.success('ML Pipeline Process Completed!')

    '''
    Load up data set from the given path as RDD

    @param  : string
    @return : RDD
    '''
    def loadDataAsRDD(self, path, training):
        # temporary handler
        rdd = MLPipelineTraining.context.textFile(path)

        # try counting the data
        try:
            # notify total data set
            if training:
                self.system('Training Data Size: {}'.format(rdd.count()))
            else:
                self.system('Testing Data Size: {}'.format(rdd.count()))

            return rdd
        except:
            # notify failure
            self.fail('Unable to load dataset from \'{}\'.').format(path)

            # exit
            sys.exit(1)

    '''
    Format training data set.

    @param  : RDD
    @return : RDD
    '''
    def formatTrainingData(self, rdd):
        # notify training dataframe formatting
        self.system('Creating Training DataFrame ...')

        # create document label
        LabeledDocument = Row('tag', 'tag_features', 'label')

        # temporary rdd
        temp = []

        # map function
        def fn(line):
            # split the line
            line = line.split(', ')

            # get the tag
            tag         = line[0]
            # get the features
            features    = line[4:6]
            # get the type
            type        = line[9]

            # default label
            label = 0.0

            # if title
            if type == 'title':
                label = 1.0

            # get the value by joining features
            value = line[1] + ' ' + ' '.join(str(s) for s in features)

            # return labeled document formatted data
            return LabeledDocument((tag), value, label)

        # format the training data
        temp = rdd.map(fn)

        # notify training dataframe
        self.success('Training DataFrame: ')

        # show trainig data set
        temp.toDF().show()

        return temp

    '''
    Format Testing data

    @param  : RDD
    @return : RDD
    '''
    def formatTestingData(self, rdd):
        # notify testing data
        self.system('Formatting Testing Data ...')

        # create document
        Document = Row('id', 'tag', 'tag_features')

        # collect the rdd data
        data = rdd.collect()

        # initialize index
        index = 0

        # initialize set
        set = []

        # iterate on each data
        for value in data:
            # get the line
            line = value.split(', ')

            # we will skip features that is zero
            if(int(line[4]) <= 0 and int(line[5]) <= 0):
                continue

            # we will also skip elements with no text
            if(int(line[8]) <= 0):
                continue

            # get the tag
            tag         = line[0]
            # get the features
            features    = line[4:6]

            # get the value by joining features
            text = line[1] + ' ' + ' '.join(str(s) for s in features)

            # append data to our set
            set.append((int(index), tag, text))

            index = index + 1

        # temporary handler, conversion to dataframe
        temp = MLPipelineTraining.context.parallelize(set).map(lambda x : Document(*x))

        # notify testing dataframe
        self.success('Testing DataFrame: ')

        # show testing dataframe
        temp.toDF().show()

        return temp

    '''
    Get training model pipeline

    @param  : DataFrame
    @return : Pipeline
    '''
    def getPipeline(self, df):
        # notify pipeline 
        self.success('Initializing ML Pipeline ...')

        # initialize our tokenizer, we're going to tokenize features
        tokenizer = Tokenizer(inputCol='tag_features', outputCol='words')
        # convert the tokenize data to vectorize data
        hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol='features')
        # initialize logistic regression algorithm
        lr        = LogisticRegression(maxIter=10, regParam=0.01)
        # create / initialize the ml pipeline
        pipeline  = Pipeline(stages=[tokenizer, hashingTF, lr])

        # fit the pipeline on our training dataframe
        model = pipeline.fit(df)

        return model

    '''
    Predict results from testing data

    @param  : Pipeline
    @param  : DataFrame
    @return : Row
    '''
    def predict(self, model, df):
        # notify prediction
        self.success('Starting Predictions ...')

        # start making predictions
        predictions = model.transform(df)

        # select prediction columns
        selected = predictions.select('tag', 'tag_features', 'prediction', 'probability')

        # notify prediction results
        self.system('Prediction Results:')
        self.system('Possible Title ({}), Non-Title ({}), Total Predicted ({})'
            .format(
                selected.filter(selected.prediction > 0).count(),
                selected.filter(selected.prediction <= 0).count(),
                selected.count()))

        # iterate on each row
        if 'showResults' in MLPipelineTraining.options:
            # if no results
            if selected.filter(selected.prediction > 0).count() <= 0 :
                self.warning("There are no predicted results ...")

                return selected.collect()

            # show prediction dataframe
            selected.filter(selected.prediction > 0).show()

            # show results text (debugging)
            self.getResultText(selected.filter(selected.prediction > 0))
            

        return selected.collect()

    '''
    Get the text data of the results.

    @param  : DataFrame
    @return : MLPipelineTraining
    '''
    def getResultText(self, df):
        # results path exists?
        if not 'results' in MLPipelineTraining.options:
            # warn about it
            self.warning('Cannot output result text, --results=[path] option is not set.')

            return

        # notify open json results
        self.success("Opening results file for data gathering ...")

        # collect data frame data
        df = df.collect();

        # try to open results file
        try:
            # load up the json results
            with open(MLPipelineTraining.options['results']) as json_file:
                data = json.load(json_file)
        except:
            # notify error
            self.fail("Unable to open results file \'{}\'.".format(MLPipelineTraining.options['results']))

            # exit
            sys.exit(1)

        # notify possible text
        self.success("Possible Title Text: ")

        # iterate on the results file
        for i in data:
            # iterate on each predicted results
            for k in df:
                # get the tag
                tag      = k.tag
                # get tag features
                features = k.tag_features.split(' ')

                # compare features
                if (i['tag'] == tag 
                and float(i['offset_x']) == float(features[1])
                and float(i['offset_y']) == float(features[2])):
                    self.warning('Title: ' + i['text'])

    '''
    Success notification

    @param  : string
    @return : MLPipelineTraining
    '''
    def success(self, message):
        print MLPipelineTraining.log_colors['OKGREEN'] + '[MLPipelineTraining]: ' + message + \
              MLPipelineTraining.log_colors['ENDC']

        return self

    '''
    Warning notification

    @param  : string
    @return : MLPipelineTraining
    '''
    def warning(self, message):
        print MLPipelineTraining.log_colors['WARNING'] + '[MLPipelineTraining]: ' + message + \
              MLPipelineTraining.log_colors['ENDC']

        return self

    '''
    Fail notification

    @param  : string
    @return : MLPipelineTraining
    '''
    def fail(self, message):
        print MLPipelineTraining.log_colors['FAIL'] + '[MLPipelineTraining]: ' + message + \
              MLPipelineTraining.log_colors['ENDC']

        return self

    '''
    System notification

    @param  : string
    @return : MLPipelineTraining
    '''
    def system(self, message):
        print MLPipelineTraining.log_colors['HEADER'] + '[MLPipelineTraining]: ' + message + \
              MLPipelineTraining.log_colors['ENDC']

        return self

    '''
    Get command line arguments

    @param  : MLPipelineTraining
    @param  : array
    @return : dictionary
    '''
    def getCommandLineArgs(self, argv):
         # holds up the arguments
        args = {}

        # iterate on each arguments
        for i in argv:
            # get the parts
            parts = i.split('=')

            # if we have key + val pair
            if len(parts) > 1:
                # set key value pair
                args[parts[0][2:]] = parts[1]

        return args

# Run the class
MLPipelineTraining()