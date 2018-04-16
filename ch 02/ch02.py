import kNN
import matplotlib
import matplotlib.pyplot as plt
from numpy import array

features,labels = kNN.createDataSet()
features

kNN.classify0([0,0],features,labels,3)

datamat,labels = kNN.file2matrix('datingTestSet.txt')

'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datamat[:,1], datamat[:,2], 15.0*array(labels), 15.0*array(labels))
plt.show()
'''

normmat,ranges,minvals = kNN.autoNorm(datamat)
print(normmat)
print(ranges)
print(minvals)

# kNN.datingClassTest(0.2,7)

def classifyperson():
	result = ['not at all', 'small doses', 'large dose']
	
	ffmiles = float(input('frequent filter miles earned per year:'))
	gametimepercent = float(input('% of time spent on game:'))
	icecream = float(input('liters of ice cream consumed per year:'))
	datamat,labels = kNN.file2matrix('datingTestSet.txt')
	normmat,ranges,minvals = kNN.autoNorm(datamat)
	inarry = (array([ffmiles,gametimepercent,icecream]) - minvals) / ranges
	classifyresult = kNN.classify0(inarry,normmat,labels,3)
	print ("you like this person:", result[classifyresult-1])
	return

# classifyperson()

testvector = kNN.img2vector('testDigits/0_13.txt')
kNN.handwritingClassTest()