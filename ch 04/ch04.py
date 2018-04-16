import bayes
from numpy import *

listpost,listClasses = bayes.loadDataSet()
myvocablist = bayes.createVocabList(listpost)
print(myvocablist)
print(bayes.setOfWords2Vec(myvocablist,listpost[0]))

trainMat = []
for postindoc in listpost:
	trainMat.append(bayes.setOfWords2Vec(myvocablist,postindoc))
p0v,p1v,pAb = bayes.trainNB0(trainMat,listClasses)
print(p0v)
print(p1v)
print(pAb)

bayes.testingNB()
testEntry = ['love', 'i', 'u']
thisDoc = array(bayes.setOfWords2Vec(myvocablist, testEntry))
print (testEntry,'classified as: ',bayes.classifyNB(thisDoc,p0v,p1v,pAb))

#bayes.spamTest()

import feedparser
ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
print(ny['entries'])
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
#bayes.localWords(ny,sf)