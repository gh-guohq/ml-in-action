import regression
from numpy import *
xarr,yarr = regression.loadDataSet('ex0.txt')
#print(xarr)

ws = regression.standRegres(xarr,yarr)
#print(ws)

xmat = mat(xarr)
ymat = mat(yarr)
yhat = xmat*ws

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xmat[:,1].flatten().A[0],ymat.T[:,0].flatten().A[0])
xcopy = xmat.copy()
xcopy.sort(0)
yhat = xcopy*ws
ax.plot(xcopy[:,1],yhat)
#plt.show()

yhat = xmat*ws
print(corrcoef(yhat.T,ymat))