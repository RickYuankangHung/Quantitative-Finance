import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
bodyfat_data = sio.loadmat('C:\\Users\\Ric\\Desktop\\545HW3\\bodyfat_data.mat')

mseList=list()
np.random.seed(0)
x = bodyfat_data['X']
y = bodyfat_data['y']
n,d= x.shape
ytrain=y[:150]
Xmatrix=np.asmatrix(np.concatenate((np.ones((n,1)),x),axis=1)[0:150,:])


ite=0
while (True):
    if ite==0:
###################Forward Pass###########################
        z0=Xmatrix
        inputNum=150
        w1=np.random.randn(64,3)
        w1[:,0]=0
        a1=z0.dot(w1.T)
        a1[a1<=0]=0
        z1=np.asmatrix(np.concatenate((np.ones((inputNum,1)),a1),axis=1))
        w2=np.random.randn(16,65)
        w2[:,0]=0
        a2=z1.dot(w2.T)
        a2[a2<=0]=0
        z2=np.asmatrix(np.concatenate((np.ones((inputNum,1)),a2),axis=1))
        w3=np.random.randn(1,17)
        w3[:,0]=0
        a3=z2.dot(w3.T)
    
    
    
    else:
        
        z0=Xmatrix
        inputNum=150
        
        a1=z0.dot(w1.T)
        a1[a1<=0]=0
        z1=np.asmatrix(np.concatenate((np.ones((inputNum,1)),a1),axis=1))
        
        
        a2=z1.dot(w2.T)
        a2[a2<=0]=0
        z2=np.asmatrix(np.concatenate((np.ones((inputNum,1)),a2),axis=1))
        
        a3=z2.dot(w3.T)
########################################################################

#############Loss Metric################################################        
    mse = (np.square(a3-ytrain)).mean(axis=0)
    print(mse)
    print(ite)
    mseList.append(np.asscalar(mse))
########################################################################    
    
#####################Backward Pass######################################    
    delta3=-2*(ytrain-a3)
    sigmaPrime2=a2.copy()
    
    sigmaPrime2[sigmaPrime2>0]=1
    delta2=np.multiply(delta3.dot(w3[:,1:]),sigmaPrime2)
    for j in range(16):
        for i in range(65):
            partial=0
            for iteN in range(inputNum):
                partial=partial+delta2[iteN,j]*z1[iteN,i]
            partial=partial/inputNum
            w2[j,i]=w2[j,i]-10**(-7)*partial
    
    sigmaPrime1=a1.copy()
    sigmaPrime1[sigmaPrime1>0]=1
    
    delta1=np.multiply(delta2.dot(w2[:,1:]),sigmaPrime1)
    for j in range(64):
        for i in range(3):
            partial=0
            for iteN in range(inputNum):
                partial=partial+delta1[iteN,j]*z0[iteN,i]
            partial=partial/inputNum
            w1[j,i]=w1[j,i]-10**(-7)*partial
########################################################################    
    ite=ite+1
    if ite>1:
        if (abs(mseList[ite-1]-mseList[ite-2])<10**(-4)):
            break

print("Stopping criterion: the absolute value of the change of MSE < 1e-4")
print("After "+str(ite)+" iterations,"+"training data MSE: "+str(mseList[-1]))
plt.title("log of MSE vs iterations")
plt.plot(np.log(mseList))
plt.ylabel("log of MSE")
plt.show()

# For test data
################################Forward Pass#################################
ytest=y[150:]
XMatrixTest=np.asmatrix(np.concatenate((np.ones((n,1)),x),axis=1)[150:,:])


z0=XMatrixTest
inputNum=n-150


a1=z0.dot(w1.T)
a1[a1<=0]=0
z1=np.asmatrix(np.concatenate((np.ones((inputNum,1)),a1),axis=1))


a2=z1.dot(w2.T)
a2[a2<=0]=0
z2=np.asmatrix(np.concatenate((np.ones((inputNum,1)),a2),axis=1))

a3=z2.dot(w3.T)
############################################################################

#########################Loss Metric#######################################
mse = (np.square(a3-ytest)).mean(axis=0)
############################################################################
print("Test data MSE: "+str(np.asscalar(mse)))
