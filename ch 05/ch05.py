from numpy import *
import logRegres

dataarr, labelmat = logRegres.loadDataSet()
weights = logRegres.gradAscent(dataarr,labelmat)
print(weights)

print(weights.getA())
#logRegres.plotBestFit(weights.getA())

weights = logRegres.stocGradAscent0(array(dataarr),labelmat)
print(weights)
#logRegres.plotBestFit(weights)

weights = logRegres.stocGradAscent1(array(dataarr),labelmat)
print(weights)
#logRegres.plotBestFit(weights)

logRegres.multiTest()