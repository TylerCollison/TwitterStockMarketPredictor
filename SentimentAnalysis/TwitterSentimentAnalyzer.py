from SentimentAnalyzer import SentimentAnalyzer
import numpy as np

class TwitterSentimentAnalyzer:
    def __init__(self):
        wordVectors = np.load("TwitterSA/wordVectors.npy")
        wordMap = np.load("TwitterSA/wordMap.npy").item()
        parameterMap = np.load("TwitterSA/paramMap.npy").item()
        MAX_SEQUENCE_LENGTH = parameterMap["MAX_SEQUENCE_LENGTH"]
        BATCH_SIZE = parameterMap["BATCH_SIZE"]
        LSTM_UNITS = parameterMap["LSTM_UNITS"]
        LEARNING_RATE = parameterMap["LEARNING_RATE"]
        self.analyzer = SentimentAnalyzer(MAX_SEQUENCE_LENGTH, BATCH_SIZE, LSTM_UNITS, LEARNING_RATE, wordMap, wordVectors)
        self.analyzer.LoadModel("TwitterSA/model")
    
    # Returns an array of 1 for positive and 0 for negative in order of the samples array
    def Evaluate(self, textSamples):
        return self.analyzer.Evaluate(textSamples)