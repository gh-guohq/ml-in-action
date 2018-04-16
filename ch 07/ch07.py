import adaboost
datmat, classlabel=adaboost.loadSimpData()

from numpy import *
d = mat(ones((5,1))/5)
#print(adaboost.buildStump(datmat,classlabel,d))

#classifier,aggClassEst = adaboost.adaBoostTrainDS(datmat,classlabel,9)
#print(classifier)
#print(aggClassEst)

#print(adaboost.adaClassify([[0,0],[1,1]],classifier))

datarr,labelarr = adaboost.loadDataSet('horseColicTraining2.txt')
classifier,aggClassEst = adaboost.adaBoostTrainDS(datarr,labelarr,40)
testarr,testlabelarr = adaboost.loadDataSet('horseColicTest2.txt')
prediction = adaboost.adaClassify(testarr,classifier)
errarr = mat(ones((67,1)))
print(errarr[prediction != mat(testlabelarr).T].sum())
adaboost.plotROC(aggClassEst.T,labelarr)