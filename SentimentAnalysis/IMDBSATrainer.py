from PreProcessor import PreProcessor
from SentimentAnalyzer import SentimentAnalyzer
import os
import gensim
import numpy as np

# Create and save configuration
MAX_SEQUENCE_LENGTH = 250
BATCH_SIZE = 24
LSTM_UNITS = 64
ITERATIONS = 100000
SAVE_FREQUENCY = 10000
LEARNING_RATE = 0.001
np.save("IMDBSA/paramMap", {"MAX_SEQUENCE_LENGTH":MAX_SEQUENCE_LENGTH, "BATCH_SIZE":BATCH_SIZE, "LSTM_UNITS":LSTM_UNITS, "LEARNING_RATE":LEARNING_RATE})

# Create Gensim compatible word vectors from Glove files
# from gensim.scripts.glove2word2vec import glove2word2vec
# glove2word2vec(glove_input_file="../wordVec/glove/glove.6B.50d.txt", word2vec_output_file="WordVectors/gensim_glove_wiki_vectors.txt")

# Load and save the word vectors and index map
model = gensim.models.KeyedVectors.load_word2vec_format("WordVectors/gensim_glove_wiki_vectors.txt", binary=False)
wordVectors = model.syn0
wordsList = model.index2word
wordMap = {wordsList[i]: i for i in range(len(wordsList))}
np.save("IMDBSA/wordMap", wordMap)
np.save("IMDBSA/wordVectors", wordVectors)

# Find the training data
positiveFiles = ['./Data/IMDBData/train/pos/' + f for f in os.listdir('./Data/IMDBData/train/pos/') if os.path.isfile(os.path.join('./Data/IMDBData/train/pos/', f))]
negativeFiles = ['./Data/IMDBData/train/neg/' + f for f in os.listdir('./Data/IMDBData/train/neg/') if os.path.isfile(os.path.join('./Data/IMDBData/train/neg/', f))]

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
analyzer.TrainModel(ITERATIONS, SAVE_FREQUENCY, positiveSamples, negativeSamples, "IMDBSA/model/pretrained_lstm.ckpt")

# Load a trained analyzer
# analyzer.LoadModel("models")

# Test the analyzer
# analyzer.TestModel(positiveSamples, [1 for x in positiveSamples])