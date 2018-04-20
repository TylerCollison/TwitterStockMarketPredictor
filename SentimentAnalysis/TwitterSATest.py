from TwitterSentimentAnalyzer import TwitterSentimentAnalyzer
from PreProcessor import PreProcessor

er = TwitterSentimentAnalyzer();
processor = PreProcessor();

tmp = "RT @Cj_Walker1: My family aand I have came to a great decision! With that being said I would like to say I have committed to The Ohio State"
clean = processor.preProcess(tmp, lowercase=True);
cleanString = "";
for s in clean:
    cleanString = cleanString + s + " "
cleanStringList = [cleanString]
for i in range(39):
    cleanStringList.append(" ")
print(cleanString);
print(processor.cleanText((tmp)));
result = er.Evaluate(cleanStringList);
print(result);