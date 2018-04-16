import trees

mydata,features = trees.createDataSet()
print(mydata)
print(features)
print(trees.calcShannonEnt(mydata))
'''
mydata[0][-1] = 'maybe'
print(trees.calcShannonEnt(mydata))
'''
#print(trees.splitDataSet(mydata,0,1))

index = trees.chooseBestFeatureToSplit(mydata)
#print(index)
'''
mytree = trees.createTree(mydata,features)
print(mytree)
'''
import treePlotter
'''
mytree = treePlotter.retrieveTree(0)
treePlotter.createPlot(mytree)
mytree['no surfacing'][3] = 'maybe'
treePlotter.createPlot(mytree)
'''

mytree = treePlotter.retrieveTree(0)
print(trees.classify(mytree,features,[0,0]))
print(trees.classify(mytree,features,[1,1]))

trees.storeTree(mytree, 'classifier.txt')
grabtree = trees.grabTree('classifier.txt')
print(grabtree)


fr = open('lenses.txt')
lense =[inst.strip().split('\t') for inst in fr.readlines()]
lensefeatures = ['age', 'prescript', 'astigmatic', 'tearrate']
lensetree = trees.createTree(lense,lensefeatures)
print(lensetree)
treePlotter.createPlot(lensetree)