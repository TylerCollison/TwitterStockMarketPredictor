# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
import re

class PreProcessor:
    def __init__(self):
        self.strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

    def cleanText(self, string):
        string = string.lower().replace("<br />", " ")
        return re.sub(self.strip_special_chars, "", string.lower())

    def cleanTextList(self, textList):
        return [self.cleanText(x) for x in textList]
        