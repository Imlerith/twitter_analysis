# -*- coding: utf-8 -*-
"""
Created on Sat May 21 20:07:21 2016

@author: nasekins
"""

import re
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()

# Do tweets' processing
def processTweet(tweet):
    # process the tweets

    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet

# Deal with characters' repetitions
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


# Obtain feature vectors
def getFeatureVector(tweet,stopwords):
    featureVector = []
    #split tweet into words
    #words = tweet.split()
    words = re.split("[ '-]", tweet)
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.:;!')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #check if there are numbers in the word
        dig = re.search("([0-9].*)",w)
        #ignore if it is a stop word
        if(w in stopwords or val is None or dig is not None):
            continue
        else:
            featureVector.append(st.stem(w.lower()))
            #featureVector.append(w.lower())
        
        featureVector = map(str, featureVector)
    return featureVector
    
# Extract features
def extract_features(tweet,featureList):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features