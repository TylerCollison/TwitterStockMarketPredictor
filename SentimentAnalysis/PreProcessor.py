# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
import tweepy,re
import datetime
# ConsumerKey = ''
# ConsumerSecrete = ''
# AccessToken = ''
# AccessTokenSecrete = ''

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    #r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

DicName="dic/normDic.txt"
class PreProcessor:
    def __init__(self):
        self.strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

    def establishConnection(self):
        self.auth = tweepy.OAuthHandler(ConsumerKey, ConsumerSecrete)
        self.auth.set_access_token(AccessToken, AccessTokenSecrete)
        self.api = tweepy.API(self.auth)

    # Process a given string s
    # @param s - input raw string
    # @return a list of processed words
    def preProcess(self,s, lowercase=False):
        tokens = tokens_re.findall(s)
        if lowercase:
            tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
        for i in range(0,len(tokens)):
            tokens[i] = "".join(tokens[i].split())
        #Normalization
        with open(DicName, "r") as f:
            line=f.readline()
            while line:
                line=line.strip('\n')
                index=line.find('\t')+1
                line=line[index:]
                for i in range(0,len(tokens)):
                    findIndex=line.find(tokens[i]);
                    if findIndex==0 and line[findIndex+len(tokens[i])+1]=='|':
                        print("Item Replaced (Dict): ", tokens[i], " # ", line)
                        tokens[i] = line[(len(tokens[i]) + 3):]
                        continue
                line=f.readline()
        #Remove hashTag, @, punctuations(not in the word), and url  (OPTIONAL)
        if "RT" in tokens:
            tokens.remove("RT")

        for item in tokens[:]:
            if ord(item[0])<65 or ord(item[0])>122 or (ord(item[0])<97 and ord(item[0])>90):
                print("Item Removed (Unnecessary char): ",item)
                tokens.remove(item)
            elif item.find("http://")!=-1 or item.find("https://")!=-1 or item.find("www.")!=-1:
                tokens.remove(item)

        return tokens
    # NOTE: This is only used for analyzing real time tweets
    def getProcessedTweets(self,keyword,numOfTweet=1):
        result=[]
        i = 0
        for status in tweepy.Cursor(self.api.search, q=keyword).items(numOfTweet):
            # Process a single status
            #fileName = "tweet" + str(i) + ".json"
            result.append(self.preProcess(status.text))
        return result

    def cleanText(self, string):
        # Original cleanText() content
        # string = string.lower().replace("<br />", " ")
        # return re.sub(self.strip_special_chars, "", string.lower())
        wordList=self.preProcess(string)
        result=' '.join(word for word in wordList)
        return result

    def cleanTextList(self, textList):
        return [self.cleanText(x) for x in textList]
        
