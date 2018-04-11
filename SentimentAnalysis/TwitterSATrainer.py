from PreProcessor import PreProcessor
from SentimentAnalyzer import SentimentAnalyzer
import gensim
import os
import numpy as np

# Create and save configuration
MAX_SEQUENCE_LENGTH = 30
BATCH_SIZE = 40
LSTM_UNITS = 128
ITERATIONS = 1000000
SAVE_FREQUENCY = 100000
LEARNING_RATE = 0.0001
np.save("TwitterSA/paramMap", {"MAX_SEQUENCE_LENGTH":MAX_SEQUENCE_LENGTH, "BATCH_SIZE":BATCH_SIZE, "LSTM_UNITS":LSTM_UNITS, "LEARNING_RATE":LEARNING_RATE})

# Create Gensim compatible word vectors from Glove files
# from gensim.scripts.glove2word2vec import glove2word2vec
# glove2word2vec(glove_input_file="../wordVec/glove/glove.twitter.27B.200d.txt", word2vec_output_file="WordVectors/gensim_glove_twitter_vectors.txt")

# Load and save the word vectors and index map
model = gensim.models.KeyedVectors.load_word2vec_format("WordVectors/gensim_glove_twitter_vectors.txt", binary=False)
wordVectors = model.syn0
wordsList = model.index2word
wordMap = {wordsList[i]: i for i in range(len(wordsList))}
np.save("TwitterSA/wordMap", wordMap)
np.save("TwitterSA/wordVectors", wordVectors)

# Find the training data
positiveFiles = ['./Data/TwitterData/pos/' + f for f in os.listdir('./Data/TwitterData/pos/') if os.path.isfile(os.path.join('./Data/TwitterData/pos/', f))]
negativeFiles = ['./Data/TwitterData/neg/' + f for f in os.listdir('./Data/TwitterData/neg/') if os.path.isfile(os.path.join('./Data/TwitterData/neg/', f))]

# Initialize the pre-processor and sentiment analyzer
processor = PreProcessor()
analyzer = SentimentAnalyzer(MAX_SEQUENCE_LENGTH, BATCH_SIZE, LSTM_UNITS, LEARNING_RATE, wordMap, wordVectors)

# Load and process the training data
negativeSamples = []
positiveSamples = []
for pf in positiveFiles:
    with open(pf, "r", encoding="utf8") as f:
       lines = f.readlines()
       positiveSamples.extend(processor.cleanTextList(lines))
       print("Cleaned positive document: " + pf)

for nf in negativeFiles:
    with open(nf, "r", encoding="utf8") as f:
       lines = f.readlines()
       negativeSamples.extend(processor.cleanTextList(lines))
       print("Cleaned negative document: " + nf)

# Train the analyzer
analyzer.TrainModel(ITERATIONS, SAVE_FREQUENCY, positiveSamples, negativeSamples, "TwitterSA/model/pretrained_lstm.ckpt")

# Load a trained analyzer
# analyzer.LoadModel("models")

# Test the analyzer
# analyzer.TestModel(positiveSamples, [1 for x in positiveSamples])