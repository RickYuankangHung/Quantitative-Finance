#Import the data
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
mnist_49_3000 = sio.loadmat('C:\\Users\\Ric\\Downloads\\mnist_49_3000.mat')
x = mnist_49_3000['x']
y = mnist_49_3000['y']
d,n= x.shape
#Remap y
y[y==-1]=0
#Set lamda value
lamda=10
#Concatenate X features matrix with constant
Xmatrix=np.asmatrix(np.concatenate((np.ones((1,3000)),x),axis=0)[:,0:2000])
#Define Gradient function with vectorizing techniques
def gradient(theta,xMatrix,y,lamda): 
    e_neg_thx=np.exp(-theta.T*xMatrix)
    e_pos_thx=np.exp(theta.T*xMatrix)
    coefvector=np.multiply(np.reciprocal(1+e_neg_thx),e_neg_thx)
    coefvector2=np.multiply(np.reciprocal(1+e_pos_thx),e_pos_thx)
    coefmatrix=np.diagflat(coefvector)
    coefmatrix2=np.diagflat(coefvector2)
    mtx=-xMatrix*coefmatrix
    mtx2=xMatrix*coefmatrix2
    grad=mtx*(y.T)+mtx2*((1-y).T)+2*lamda*theta
    return grad
#Define Hessian function
def Hessian(theta,Xmatrix,y,lamda):
    [d,n]=Xmatrix.shape
    hessian=np.zeros((d,d))
    for i in range(n):
        xTilta=np.asmatrix(Xmatrix[:,i])
        e_pos_thx=np.asscalar(np.exp(theta.T*xTilta))
        iteMatrix=(1+e_pos_thx)**(-2)*e_pos_thx*xTilta*xTilta.T
        hessian=hessian+iteMatrix
    hessian=hessian+2*lamda*np.identity(d)
    return hessian
#Define ObjFunc
def objFunc(theta,xMatrix,y,lamda):
    e_neg_thx=np.exp(-theta.T*xMatrix)
    e_pos_thx=np.exp(theta.T*xMatrix)
    coefvector=np.log(1+e_neg_thx)
    coefvector2=np.log(1+e_pos_thx)
    obj1=np.asscalar(y[0:1,0:2000]*coefvector.T)
    obj2=np.asscalar((1-y[0:1,0:2000])*coefvector2.T)
    obj=obj1+obj2+lamda*np.asscalar(theta.T*theta)
    return obj
#Ready to iterate
theta=np.zeros((785,1))
iteTime=0
condition = True
#Starting iterating
while condition:
    beforetheta=theta.copy() #Record the theta_(t-1)
    theta=theta-np.linalg.inv(Hessian(theta,Xmatrix,y[0:1,0:2000],lamda))*gradient(theta,Xmatrix,y[0:1,0:2000],lamda)
    e_neg_thx=np.exp(-theta.T*Xmatrix)
    etavec=np.reciprocal(1+e_neg_thx)
    yhat=np.copy(etavec)
    yhat[yhat>=0.5]=1
    yhat[yhat<0.5]=0
    aftertrainError=1-np.equal(yhat,y[0:1,0:2000]).sum()/y[0:1,0:2000].shape[1]
    condition = (np.linalg.norm(beforetheta-theta)/np.linalg.norm(theta)>0.01)
    iteTime=iteTime+1
    print(str(iteTime)+' iteration(s)')
    objOpt=objFunc(theta,Xmatrix,y[0:1,0:2000],lamda)
#Test set
e_neg_thxTest=np.exp(-theta.T*np.asmatrix(np.concatenate((np.ones((1,3000)),x),axis=0)[:,2000:3000]))
etavecTest=np.reciprocal(1+e_neg_thxTest)
yhatTest=np.copy(etavecTest)
yhatTest[yhatTest>=0.5]=1
yhatTest[yhatTest<0.5]=0
testError=1-np.equal(yhatTest,y[0:1,2000:3000]).sum()/y[0:1,2000:3000].shape[1]
print('Test error: '+str(testError))
print('Terminal condition: '+'The norm of difference of theta_t and theta_(t-1) is less than or equal to 1% of norm of theta_t')
print('Objective function at the optimum: '+str(objOpt))
#Confidence evaluation
ConfetavecTest=abs(etavecTest-0.5)
# The confidence can be the distance between eta and 0.5.
# Because eta and 1-eta denote the conditional pmf of y=1 or y=0 given x.
# The greater |eta-0.5| is, the higher max(P(Y=1|X),P(Y=0|X)f) we will have as estimated. 
# For example, if eta is greater than or equal to 0.5, this data point will be classified as label 1. Therefore, the greater eta-0.5 is, the more confident we are to classify it to be label 1.
ConfetavecTest[np.equal(yhatTest,y[0:1,2000:3000])==True]=0
N=20
confiWrong=np.asarray(ConfetavecTest).argsort()[0:1,-N:][::-1]
#Plot the wrong confident images
print('label 0 stands for digit 4\n'+'label 1 stands for digit 9')
for i in range(20):  
    plt.subplot(4,5,i+1)
    plt.title('True: '+str(y[0,2000+confiWrong[0,i]]))
    plt.imshow( np.reshape(x[:,2000+confiWrong[0,i]], (int(np.sqrt(d)),int(np.sqrt(d)))))
plt.tight_layout()
plt.show()
print('label 0 stands for digit 4\n'+'label 1 stands for digit 9')
print('The confidence I defined is the distance between eta and 0.5. Because eta and 1-eta denote the conditional pmf of y=1 or y=0 given x. The greater |eta-0.5| is, the higher max(P(Y=1|X),P(Y=0|X)f) we will have as estimated. For example, if eta is greater than or equal to 0.5, this data point will be classified as label 1. Therefore, the greater eta-0.5 is, the more confident we are to classify it to be label 1.')
