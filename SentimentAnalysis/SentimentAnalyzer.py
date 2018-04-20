import numpy as np
import tensorflow as tf
from random import randint
import os
import datetime
import math

def getBatch(posIds, negIds, batchSize, maxSeqLength):
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    posCount, _ = posIds.shape
    negCount, _ = negIds.shape
    for i in range(batchSize):
        if (i % 2 == 0): 
            num = randint(1, posCount)
            labels.append([1,0])
            arr[i] = posIds[num-1:num]
        else:
            num = randint(1, negCount)
            labels.append([0,1])
            arr[i] = negIds[num-1:num]
    return arr, labels

def generateWordIDMatrix(maxSeqLength, textSamples, wordMap, mapLength):
    numFiles = len(textSamples)
    print("generating word ID matrix for " + str(numFiles) + " samples")
    ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
    fileCounter = 0
    for text in textSamples:
        print("processing sample: " + str(fileCounter))
        indexCounter = 0
        split = text.split()
        for word in split:
            try:
                ids[fileCounter][indexCounter] = wordMap[word]
            except KeyError:
                ids[fileCounter][indexCounter] = mapLength - 1 #Vector for unkown words
            indexCounter = indexCounter + 1
            if indexCounter >= maxSeqLength:
                break
        fileCounter = fileCounter + 1 
    return ids


def createLTMLRNNGraph(numClasses, maxSeqLength, batchSize, lstmUnits, learningRate, wordVectors):
    LSTMRNNGraph = tf.Graph()
    with LSTMRNNGraph.as_default():
        # Create placeholders for the input data and their corresponding labels
        labels = tf.placeholder(tf.float32, [batchSize, numClasses])
        input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

        # Create embedding weights for word vectors
        numWordVectors, numDimensions = wordVectors.shape

        # Create a 3-D tensor by looking up the word vectors using IDs in the input data
        data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
        data = tf.nn.embedding_lookup(wordVectors, input_data)

        # Create the LSTM-RNN network
        lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

        # Perform final predictions using the LSTM-RNN network
        weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
        bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = (tf.matmul(last, weight) + bias)

        # Determine the accuracy of the predictions
        correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

        # Use ADAM (Gradient Descent) to optimize the model
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss)

    return LSTMRNNGraph, prediction, optimizer, accuracy, loss, input_data, labels

def trainLSTMRNNNetwork(graph, loss, accuracy, optimizer, input_data, labels, posIds, negIds, iterations, saveFrequency, batchSize, maxSeqLength, wordVectors, saveLocation):
    print("Use command tensorboard --logdir=tensorboard to view training at http://localhost:6006/")
    saveRate = saveFrequency
    with graph.as_default():
        # Setup TensorBoard to visualize the training procedure
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Accuracy', accuracy)
        merged = tf.summary.merge_all()
        logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir, graph)

        # Initialize the graph
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for i in range(iterations):
            # Next Batch of reviews
            nextBatch, nextBatchLabels = getBatch(posIds, negIds, batchSize, maxSeqLength);
            sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

            # Write summary to Tensorboard
            if (i % 50 == 0):
                summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
                writer.add_summary(summary, i)

            # Save the network every 10,000 training iterations
            if (i % saveRate == 0 and i != 0):
                save_path = saver.save(sess, saveLocation, global_step=i)
                print("saved to %s" % save_path)

        writer.close()

    return sess

class SentimentAnalyzer:
    def __init__(self, maxSeqLength, batchSize, lstmUnits, learningRate, wordMap, wordVectors):
        self.maxSeqLength = maxSeqLength
        self.batchSize = batchSize
        self.lstmUnits = lstmUnits
        self.wordMap = wordMap
        self.wordVectors = wordVectors
        numClasses = 2
        self.graph, self.prediction, self.optimizer, self.accuracy, self.loss, self.input_data, self.labels = createLTMLRNNGraph(numClasses, maxSeqLength, batchSize, lstmUnits, learningRate, wordVectors)

    def LoadModel(self, modelSaveLocation):
        with self.graph.as_default():
            sess = tf.InteractiveSession()
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(modelSaveLocation))
            self.session = sess

    def TrainModel(self, iterations, saveFrequency, positiveTextSamples, negativeTextSamples, modelSaveLocation):
        positiveTrainIDMatrix = generateWordIDMatrix(self.maxSeqLength, positiveTextSamples, self.wordMap, len(self.wordMap))
        negativeTrainIDMatrix = generateWordIDMatrix(self.maxSeqLength, negativeTextSamples, self.wordMap, len(self.wordMap))
        self.session = trainLSTMRNNNetwork(self.graph, self.loss, self.accuracy, self.optimizer, self.input_data, self.labels, positiveTrainIDMatrix, negativeTrainIDMatrix, iterations, saveFrequency, self.batchSize, self.maxSeqLength, self.wordVectors, modelSaveLocation)

    def TestModel(self, textSamples, textLabels):
        acc = 0
        iterations = math.floor(len(textSamples) / self.batchSize)
        labelMap = {1:[1, 0], 0:[0, 1]}
        for i in range(iterations):
            nextBatch = generateWordIDMatrix(self.maxSeqLength, textSamples[(i * self.batchSize):((i * self.batchSize) + self.batchSize)], self.wordMap, len(self.wordMap))
            nextBatchLabels = [labelMap[x] for x in textLabels[(i * self.batchSize):((i * self.batchSize) + self.batchSize)]]
            nextAcc = (self.session.run(self.accuracy, {self.input_data: nextBatch, self.labels: nextBatchLabels}))
            print("Batch " + str(i) + " Accuracy: " + str(nextAcc))
            acc = acc + nextAcc
        return acc / iterations
    
    # Returns 1 for positive; 0 for negative
    def Evaluate(self, textSamples):
        idMatrix = generateWordIDMatrix(self.maxSeqLength, textSamples, self.wordMap, len(self.wordMap))
        return np.argmin(self.session.run(self.prediction, {self.input_data: idMatrix}), 1)