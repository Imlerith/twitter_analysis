# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:40:14 2016

@author: nasekin
"""

import os
import json
import scipy.io as sio
from scipy.stats import gaussian_kde
#import urllib2
import re
#import pylab as pyl
import matplotlib.pyplot as plt
import numpy as np
import nltk
import pandas as pd
from sklearn.svm import SVC
clf = SVC()

os.chdir('/Users/nasekins/Desktop/python_twitter/final_tweets1416May')
from process_funcs import *

tweet_file = open('tweets1522.txt')
posfile    = open("poswords.txt")
negfile    = open("negwords.txt")


#Construct a dictionary for positive words
pscores = {} # initialize an empty dictionary
for line in posfile:
  term  = line.rstrip('\n')  # remove newline characters
  pscores[term] = 1.0  
  
#Construct a dictionary for negative words
nscores = {} # initialize an empty dictionary
for line in negfile:
  term  = line.rstrip('\n')  # remove newline characters
  nscores[term] = -1.0

#Merge two dictionaries together
scores = pscores.copy()
scores.update(nscores)
    
      
#KEYWORDS FOR THE AIRLINE INDUSTRY
indstr = 'air'
keystubsAir = ['air asi', 'Air asi', 'irasi', 'air As', 'Air As', 'irAs', 
               'lymoj', 'lindo Ai', 'lindo ai', 'lindoAi', 'lindoai', 'aysia Air',
               'aysiaAir', 'aysiaair', 'aya Air', 'ayaAir', 'ayaair']  
               
               
#KEYWORDS FOR THE AUTOMOTIVE INDUSTRY
indstr = 'auto'
keystubsAutoSmall = ['volvo', 'volvo','kia', 'peugeot', 'perodua', 'bmw', 'ford',
                     'land rover', 'land Rover', 'toyota', 'nissan', 'mitsubishi', 'subaru',
                     'isuzu', 'volkswagen', 'hyundai', 'proton satria', 'proton persona',
                     'proton saga', 'proton exora', 'proton inspira','proton savvy', 'proton waja']            
kscfrst      = [wrd.capitalize() for wrd in keystubsAutoSmall]
kscall       = [wrd.title() for wrd in keystubsAutoSmall]
kscallcap    = [wrd.upper() for wrd in keystubsAutoSmall]
keystubsAuto = keystubsAutoSmall + kscfrst + kscall + kscallcap
keystubsAuto = list(set(keystubsAuto))


#KEYWORDS FOR THE HOTELS & RESTAURANTS INDUSTRY (just list keywords here, not brands, as it is a
#monopolistic competition type industry)
indstr = 'hotel'
keystubsHotRestSmall  = ['hotel', 'motel','pub', 'club', 'bar', 'inn', 'resort',
                        'hostel', 'restaurant', 'cafe']
keystubsHotRestPlural = [wrd + 's' for wrd in keystubsHotRestSmall]
keystubsHotRestCap    = [wrd.capitalize() for wrd in keystubsHotRestSmall]
keystubsHotRest       = keystubsHotRestSmall + keystubsHotRestPlural + keystubsHotRestCap
keystubsHotRest       = list(set(keystubsHotRest))
            
#keystubs2 = ['ir asi', 'ir As','irAs', 'aya Air', 'ayaAir', 'ayaair', 'irefl',
#            'lymoj', 'lindo Ai', 'lindo ai', 'lindoAi', 'lindoai', 'aysia Air',
#            'aysiaAir', 'aysiaair']
            
#punc = "'.,/;:[]{}()=+-_*&^%$#!?~0123456789"
#albt = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
#           
#def translate(to_translate, to_langage="en", langage="auto"):
#    
#	'''Return the translation using google translate
#	you must shortcut the langage you define (French = fr, English = en, Spanish = es, etc...)
#	if you don't define anything it will detect it or use english by default
#	Example:
#	print(translate("salut tu vas bien?", "en"))
#	hello you alright?'''
#	agents       = {'User-Agent':"Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727; .NET CLR 3.0.04506.30)"}
#	before_trans = 'class="t0">'
#	link         = "http://translate.google.com/m?hl=%s&sl=%s&q=%s" % (to_langage, langage, to_translate.replace(" ", "+"))
#	request      = urllib2.Request(link, headers=agents)
#	page         = urllib2.urlopen(request).read()
#	result       = page[page.find(before_trans)+len(before_trans):]
#	result       = result.split("<")[0]
# 
#	return result

     
###############################################################################
#COLLECTING ALL TWEETS TO TRAIN SVM
###############################################################################
     
twtlistall = []
txtlistall = []
scresall   = []
countall   = 0
for line in tweet_file:
    #Read tweets, remove "bad" ones
    try:
        twt = json.loads(line)
    except:
        continue
    if (''u'delete''' in twt.keys()):
        continue
    try:
        txt = str(twt[''u'text'''])
        lng = str(twt[''u'lang'''])
    except:
        continue
    if lng == "en":
        txt = processTweet(txt)
        txtwrds = txt.split()
                
        scre = 0
        for wrd in txtwrds:
            if wrd in scores.keys():
                scre += scores[wrd]
        
        scresall.append(scre)
        twtlistall.append(twt)
        txtlistall.append(txt)
        
        countall += 1     
     
###############################################################################
#THIS SECTION IS FOR INDUSTRY-RELATED TWEETS' SELECTION
###############################################################################
twtlist = []
txtlist = []
scres   = []
wrdsusd = []
count   = 0
noneng  = 0
for line in tweet_file:
    #Read tweets, remove "bad" ones
    try:
        twt = json.loads(line)
    except:
        continue
    if (''u'delete''' in twt.keys()):
        continue
    try:
        txt = str(twt[''u'text'''])
        lng = str(twt[''u'lang'''])
    except:
        continue
    
    siml = 0
    at_detect = 0
    #Add the tweet if it contains at least one keyword
    if indstr == 'air':
        for kwrd in keystubsAir:
            restr  = '.+' + kwrd + '.+'
            simlst = re.findall(restr,txt)
            if len(simlst) > 0:
                siml += 1
    #            lstmrgd = ''.join(simlst)
    #            if '@' in lstmrgd:
    #                at_detect += 1
    elif indstr == 'auto':
        for kwrd in keystubsAuto:
            restr  = '(\s+' + kwrd + '\s+)'
            simlst = re.findall(restr,txt)
            #print(simlst)
            if len(simlst) > 0:
                siml += 1
                
    elif indstr == 'hotel':
        for kwrd in keystubsHotRest:
            restr  = '(\s+' + kwrd + '\s+)'
            simlst = re.findall(restr,txt)
            if len(simlst) > 0:
                siml += 1
                
                
    if siml > 0 and lng == "en" and re.match("^([iI]'m at)",txt) is None: 
        print txt
        txt = processTweet(txt)
        txtwrds = txt.split()
#        trwrds = []
#        for wrd in txtwrds:
#            trwrd = translate(wrd)
#            if all(x in albt for x in list(wrd)):
#                trwrds.append(trwrd)
#                
#        if trwrds != txtwrds:
#            txtnew = txt.replace("\n", " ")
#            trwrds = translate(txtnew)
#            txt = trwrds
#            txtwrds = txt.split()
            
        scre = 0
        for wrd in txtwrds:
            if wrd in scores.keys():
                scre += scores[wrd]
                wrdsusd.append(wrd)
                
        
        scres.append(scre)
        twtlist.append(twt)
        txtlist.append(txt)
        
    count += 1
    
len(list(set(wrdsusd)))
negind = [i for i,j in enumerate(scres) if j < 0]    
  
###############################################################################
#CREATING THE FEATURE VECTORS
###############################################################################  
#st = open('stopWords.txt', 'r')

stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", 
"almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  
"an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", 
"back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", 
"being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", 
"cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", 
"during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", 
"every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", 
"five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", 
"go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", 
"hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", 
"interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", 
"made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", 
"much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", 
"none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", 
"onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", 
"perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", 
"several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", 
"someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", 
"than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", 
"therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though", 
"three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", 
"twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", 
"whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", 
"wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", 
"with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

#cnt = 0
#stopWords = []
#for wrd in st:
#    stopWords.append(wrd.strip('\n'))
#    if wrd.strip('\n') in scores.keys():
#        cnt += 1 
#stopWords.append('AT_USER')
#stopWords.append('URL')

stopWords = nltk.corpus.stopwords.words('english')
stopWords.append("AT_USER")
stopWords.append("URL")
stopWords = list(set(stopWords+stopwords))
stopWords = map(str,stopWords)

#tweets_cl = []
#featureList = []
#for text, score in zip(txtlistall,scresall):
#    featVec = getFeatureVector(text,stopWords)
#    if score < 0:
#        tweets_cl.append((featVec,-1))
#    elif score > 0:
#        tweets_cl.append((featVec,1))
#    else:
#        tweets_cl.append((featVec,0))
#        
#    featureList += featVec

###############################################################################
#FEATURE VECTOR FOR ALL TWEETS
###############################################################################
tweets_cl = []
featureList = []
classCat = []
for text, score in zip(txtlistall,scresall):
    featVec = getFeatureVector(text,stopWords)
    if score < 0:
        tweets_cl.append(featVec)
        classCat.append(-1)
    elif score > 0:
        tweets_cl.append(featVec)
        classCat.append(1)
    else:
        tweets_cl.append(featVec)
        classCat.append(0)
        
    featureList += featVec
###############################################################################
    
###############################################################################
#FEATURE VECTOR ONLY FOR RELEVANT TWEETS
###############################################################################
tweets_uncl = []
testCat     = []
for text, score in zip(txtlist,scres):
    featVec = getFeatureVector(text,stopWords)
    if score < 0:
        tweets_uncl.append(featVec)
        testCat.append(-1)
    elif score > 0:
        tweets_uncl.append(featVec)
        testCat.append(1)
    else:
        tweets_uncl.append(featVec)
        testCat.append(0) 
###############################################################################
#SAVE TEST SAMPLE DATA TO MATLAB FORMAT
uncltwtList = np.array(tweets_uncl)
testCatList = np.array(testCat)
sio.savemat('uncltwtListAuto.mat', mdict={'uncltwtList': uncltwtList}) #save the classification labels to a mat-file
sio.savemat('testCatListAuto.mat', mdict={'testCatList': testCatList})


#SORT OUT "NOISE" WORDS AND EXPORT FEATURE VECTORS TO MATLAB FORMAT
len(featureList)
len(twtlistall)
from collections import Counter
counts = Counter(featureList)
len(counts)
print(counts)
countsDict = dict(counts)
countsLrg  = dict((k, v) for k, v in countsDict.items() if v >= 50)
len(countsLrg)
print(countsLrg)
setOrig      = set(featureList)
setFilt      = set(countsLrg.keys())
listFin      = list(setOrig.intersection(setFilt))
len(listFin)

#EXPORT TO MATLAB FOR PLOTTING
featfrqs = np.array(countsLrg.values())
featwrds = np.array(countsLrg.keys())
sio.savemat('featfrqs', mdict={'featfrqs': featfrqs})
sio.savemat('featwrds', mdict={'featwrds': featwrds}) 

##########################################################################################
#TF-IDF MATRIX AND CLUSTERING
##########################################################################################
def tf(term, document):
  return document.count(term)

term_matrix = []
tweets_nonem = []
for doc in tweets_cl:
    tf_vector = [tf(word, doc) for word in listFin]
    tf_vector_string = ', '.join(format(freq, 'd') for freq in tf_vector)
    print 'The tf vector for Document %d is [%s]' % ((tweets_cl.index(doc)+1), tf_vector_string)
    if sum(tf_vector) != 0:
        term_matrix.append(tf_vector)
        tweets_nonem.append(doc)

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(term_matrix)


#DO K-MEANS CLUSTERING ON TWEETS
from sklearn.cluster import KMeans
num_clusters = 3

from sklearn.decomposition import PCA           
reduced_data = PCA(n_components=2).fit_transform(tfidf.toarray())
kmeans       = KMeans(init='random', max_iter = 1000, n_clusters=num_clusters, n_init=10)
km           = kmeans.fit(reduced_data)
kmpred       = kmeans.predict(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .001     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy       = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z            = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])


from matplotlib import colors
my_cmap = colors.ListedColormap(['#99FF92', '#92D8FF', '#FFA292'])
  
#PLOT THE CALCULATED CLUSTERS 
%matplotlib qt
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = my_cmap,
           aspect = 'auto', origin = 'lower')

for i in range(0,reduced_data.shape[0]):
    if kmpred[i] == 0:
        cl1 = plt.plot(reduced_data[i, 0], reduced_data[i, 1], c='g', marker = 'o', markersize=3)
        
    elif kmpred[i] == 1:
        cl2 = plt.plot(reduced_data[i, 0], reduced_data[i, 1], c='b', marker = '+', markersize=3)
        
    elif kmpred[i] == 2:
        cl3 = plt.plot(reduced_data[i, 0], reduced_data[i, 1], c='r', marker = '*', markersize=3)
        

# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on tweets')
plt.xlim(reduced_data[:, 0].min(), reduced_data[:, 0].max()+0.1)
plt.ylim(reduced_data[:, 1].min()-0.1, reduced_data[:, 1].max())
plt.savefig('clusters4.png',dpi = 300, bbox_inches = 'tight', pad_inches = 0.05)


##########################################################################################
#COMPUTE COSINE SIMILARITY: POSITIVE SUB-LEXICON
##########################################################################################
poslist = map(st.stem,pscores.keys())
poslist = map(str,poslist)

Clust0 = []
Clust1 = []
Clust2 = []

for i in range(0,len(tweets_nonem)):
    if kmpred[i] == 0:
        Clust0.append(tweets_nonem[i])
    elif kmpred[i] == 1:
        Clust1.append(tweets_nonem[i])
    elif kmpred[i] == 2:
        Clust2.append(tweets_nonem[i])
   
   
simlist0 = []       
for doc in Clust0:
    doc_vals = Counter(doc)
    lex_vals = Counter(poslist)
    words    = list(set(doc_vals) | set(lex_vals))
    doc_vec  = [doc_vals.get(word, 0) for word in words]   
    lex_vec  = [lex_vals.get(word, 0) for word in words]    
    len_doc  = sum(av*av for av in doc_vec) ** 0.5    
    len_lex  = sum(bv*bv for bv in lex_vec) ** 0.5                
    dotprod  = sum(av*bv for av,bv in zip(doc_vec, lex_vec))    
    cossim   = dotprod / (len_doc * len_lex)
    simlist0.append(cossim)

AverCosSim0 = sum(simlist0)/len(simlist0)
print(AverCosSim0)

simlist1 = []       
for doc in Clust1:
    doc_vals = Counter(doc)
    lex_vals = Counter(poslist)
    words    = list(set(doc_vals) | set(lex_vals))
    doc_vec  = [doc_vals.get(word, 0) for word in words]   
    lex_vec  = [lex_vals.get(word, 0) for word in words]    
    len_doc  = sum(av*av for av in doc_vec) ** 0.5    
    len_lex  = sum(bv*bv for bv in lex_vec) ** 0.5                
    dotprod  = sum(av*bv for av,bv in zip(doc_vec, lex_vec))    
    cossim   = dotprod / (len_doc * len_lex)
    simlist1.append(cossim)

AverCosSim1 = sum(simlist1)/len(simlist1)
print(AverCosSim1)

simlist2 = []       
for doc in Clust2:
    doc_vals = Counter(doc)
    lex_vals = Counter(poslist)
    words    = list(set(doc_vals) | set(lex_vals))
    doc_vec  = [doc_vals.get(word, 0) for word in words]   
    lex_vec  = [lex_vals.get(word, 0) for word in words]    
    len_doc  = sum(av*av for av in doc_vec) ** 0.5    
    len_lex  = sum(bv*bv for bv in lex_vec) ** 0.5                
    dotprod  = sum(av*bv for av,bv in zip(doc_vec, lex_vec))    
    cossim   = dotprod / (len_doc * len_lex)
    simlist2.append(cossim)

AverCosSim2 = sum(simlist2)/len(simlist2)
print(AverCosSim2)
##########################################################################################

##########################################################################################
#COMPUTE COSINE SIMILARITY: NEGATIVE SUB-LEXICON
##########################################################################################
negwrds = nscores.keys()
neglist = []
for term in negwrds:
    try:
        stterm = st.stem(term)
    except UnicodeDecodeError:
        print "Fuck!"
        continue
    else:
        neglist.append(stterm)

neglist = map(str,neglist)
   
   
simlist0 = []       
for doc in Clust0:
    doc_vals = Counter(doc)
    lex_vals = Counter(neglist)
    words    = list(set(doc_vals) | set(lex_vals))
    lex_vec  = [lex_vals.get(word, 0) for word in words]
    len_lex  = sum(bv*bv for bv in lex_vec) ** 0.5 
    doc_vec  = [doc_vals.get(word, 0) for word in words]       
    len_doc  = sum(av*av for av in doc_vec) ** 0.5                   
    dotprod  = sum(av*bv for av,bv in zip(doc_vec, lex_vec))    
    cossim   = dotprod / (len_doc * len_lex)
    simlist0.append(cossim)

AverCosSim0 = sum(simlist0)/len(simlist0)
print(AverCosSim0)

simlist1 = []       
for doc in Clust1:
    doc_vals = Counter(doc)
    lex_vals = Counter(neglist)
    words    = list(set(doc_vals) | set(lex_vals))
    lex_vec  = [lex_vals.get(word, 0) for word in words]
    len_lex  = sum(bv*bv for bv in lex_vec) ** 0.5 
    doc_vec  = [doc_vals.get(word, 0) for word in words]       
    len_doc  = sum(av*av for av in doc_vec) ** 0.5                   
    dotprod  = sum(av*bv for av,bv in zip(doc_vec, lex_vec))    
    cossim   = dotprod / (len_doc * len_lex)
    simlist1.append(cossim)

AverCosSim1 = sum(simlist1)/len(simlist1)
print(AverCosSim1)

simlist2 = []       
for doc in Clust2:
    doc_vals = Counter(doc)
    lex_vals = Counter(neglist)
    words    = list(set(doc_vals) | set(lex_vals))
    lex_vec  = [lex_vals.get(word, 0) for word in words]
    len_lex  = sum(bv*bv for bv in lex_vec) ** 0.5 
    doc_vec  = [doc_vals.get(word, 0) for word in words]       
    len_doc  = sum(av*av for av in doc_vec) ** 0.5                   
    dotprod  = sum(av*bv for av,bv in zip(doc_vec, lex_vec))    
    cossim   = dotprod / (len_doc * len_lex)
    simlist2.append(cossim)

AverCosSim2 = sum(simlist2)/len(simlist2)
print(AverCosSim2)



#MOVE THE ARRAYS TO MATLAB FORMAT
featList     = np.array(listFin)
cltwtList    = np.array(tweets_cl)
classCatList = np.array(classCat)
sio.savemat('classCatList_tweets3.mat', mdict={'classCatList': classCatList}) #save the list of classified tweets to a mat-file
sio.savemat('cltwtList5_tweets3.mat', mdict={'cltwtList': cltwtList}) #save the classification labels to a mat-file
sio.savemat('featList5_tweets3.mat', mdict={'featList': featList}) #save the featureList to a mat-file

Xmat = []
for line in tweets_cl:
    ftrvec = []
    for wrd in featureList:
        if wrd in line:
            ftrvec.append(1)
        else:
            ftrvec.append(0)
    Xmat.append(ftrvec)
    
    
lst = []
arr = np.array(lst)
for i in range(9):
    for j in range(9):
        arr[i][j] = 0
        
############################################################################### 
#"""SVM FITTING ON THE TWEETS IN THE SAMPLE"""   
#       
#Y = np.array(Ymat)
#X = np.array(Xmat)
#clf.fit(X, Y)
#clf.predict(X)
#
#
#    
#negsc = 0
#possc = 0
#neutsc = 0
#for score in scresall:
#    if score < 0:
#        negsc += 1
#    elif score > 0:
#        possc += 1
#    else:
#        neutsc += 1
###############################################################################
    
def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)

#WORDS' HISTOGRAM
#featList = sio.loadmat('featList5.mat')
#featlist = map(str.strip,map(str,list(featList['featList'])))
#featureListDict = dict((x, featureList.count(x)) for x in featureList)
#pyl.hist(featureListDict.values(),bins=50,normed=1)

#SCORES' HISTOGRAM   
h         = plt.hist(scres, bins = 10, normed=True, color=(0,.5,0,1), 
                     align='left', label='Histogram')
sc_grid   = np.linspace(min(scres), max(scres), 100)
domain    = np.array(scres)
#bandwidth = 1.06*domain.std()*(len(scres)**(-0.2))
bandwidth = 0.3
kde       = kde_scipy(domain, sc_grid, bandwidth)
kd        = plt.plot(sc_grid,kde, lw=2)
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.savefig('autofreq.png',dpi = 600, bbox_inches = 'tight', pad_inches = 0.05)

plt.show()
      
#pyl.hist(featureList,bins=50,normed=1)
#pyl.title("Histogram of tweets' scores")
#pyl.xlabel('Score')
#pyl.ylabel('Frequency')
#pyl.plot(sc_grid,kde)



