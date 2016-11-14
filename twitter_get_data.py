# -*- coding: utf-8 -*-
"""
Created on Fri May 06 22:44:37 2016

@author: nasekin
"""

"""
Obtains tweets' data from the territory of Malaysia via Twitter API
Input: Twitter OAuth authorization data
Output: tweets' raw data in JSON format
"""

consumer_key = 	''
consumer_secret = ''
access_token = ""
access_secret = ""

import os
os.chdir('/Users/nasekins/Desktop')
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener


# file name that you want to open is the second argument
save_file = open('data.json', 'a')

class listener(StreamListener):

    def on_data(self, data):
        #print data
        with open('fetched_tweets22.txt','a') as tf:
            tf.write(data)
        return True

    def on_error(self, status):
        print status

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
twitterStream = Stream(auth, listener())
twitterStream.filter(locations=[99.96, 4.94, 100.80, 6.37,  100.84, 4.94, 103.37, 5.60,
                                                  100.46, 2.37, 104.02, 4.94,  101.96, 1.49, 104.29, 2.37,
                                                  110.09, 1.22, 111.02, 1.81,  110.14, 0.96, 111.02, 1.22,
                                                  111.02, 1.07, 111.90, 2.96,  111.90, 1.15, 112.15, 2.96,
                                                  112.15, 1.59, 114.62, 4.01,  113.73, 4.01, 114.25, 4.51,
                                                  114.62, 3.06, 115.52, 4.004,  115.33, 4.41, 119.28, 7.39,
                                                  117.46, 4.19, 118.81, 4.63])