import svmMLiA
dataArr,labelArr = svmMLiA.loadDataSet('testSet.txt')
print(labelArr)

#b,alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
#print(b,alphas[alphas>0])

b,alphas = svmMLiA.smoPK(dataArr, labelArr, 0.6, 0.001, 40)
ws = svmMLiA.calcWs(alphas,dataArr,labelArr)
from numpy import *
datMat = mat(dataArr)
for i in range(10):
    print(datMat[i]*mat(ws)+b,labelArr[i])

'''
svm的分类函数和 模型评估（参数优化）函数中都有向量点积。是针对线性可分问题的
那么，对于非线性数据的高维映射（在高维下是线性问题），就是要求高维向量的点积。
如果可以直接表达这些高维向量点积的结果，就不需要先将原始数据做映射再做点积这些步骤。
核函数，就是两个高维向量点积的结果。
'''

svmMLiA.testRbf()

svmMLiA.testDigits(('rbf',20))