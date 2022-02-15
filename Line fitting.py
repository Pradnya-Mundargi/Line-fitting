"""
@author: Pradnya Mundargi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# Read data and create X and Y coordinates
data = pd.read_csv(r'*insert path*')

x = np.array(data['age'])
x=(x-np.min(x))/(np.max(x)-np.min(x))
y = np.array(data['charges'])
y=(y-np.min(y))/(np.max(y)-np.min(y))
x_mean= np.mean(x)
y_mean= np.mean(y)

# Least square
def leastsq(m,n):
    X = []
    for i in range(len(m)):
        X.append([1,m[i]])
    Xdata = np.array(X)
    Y = np.array(n)
    P = np.dot(np.linalg.inv(np.dot(np.transpose(Xdata),Xdata)),(np.dot(np.transpose(Xdata),Y)))
    return P

def points(m,n,P):
    y_pred = np.array([P[1]])*m+np.array([P[0]])
    return y_pred



# Total Least square
def Total_LS(x,y):
    x=x.reshape(len(x),1)
    y=y.reshape(len(x),1)
    d=-1*np.ones(len(x)).reshape(len(x),1)
    A=np.hstack((x,d,y))  
    
    #SVD
    S,U=np.linalg.eig(A@A.T)
    S,V=np.linalg.eig(A.T@A)
    S=np.sqrt(S)
    V=-V.T
    result=V[-1,:]

    #Reorganizing the matrix to compute Y
    result[1]=-result[1]
    result=result/result[2]
    return result

A=Total_LS(x,y)
X=np.hstack((x.reshape(len(x),1),-1*np.ones(len(x)).reshape(len(x),1)))

    

#Ransac
def ransac(m,n,threshold):
    #Calculate the inliers and give the best equation out        
    max_iter=np.inf
    no_of_iter=0
    max_inlier=0
    p=0.99
    e=0.5
    sample=np.zeros(2)
    while max_iter>no_of_iter:
        A= leastsq(m,n)
        y_pred= np.array([A[1]])*m+np.array([A[0]])
        y_act= n 
        error_y=np.square(y_pred- y_act)
        
        error_y[error_y>threshold]=0
        num_inlier = np.count_nonzero(error_y)
        
        e=1-(num_inlier/len(m))
        if max_inlier<num_inlier:
            max_inlier=num_inlier
            sample=A
       
        """  adaptive thresholding"""
        max_iter=np.log(1-p)/np.log(1-(1-e)**(2)+0.0000000001)
        no_of_iter+=1
     
    return sample

K=ransac(x,y,2)

#plots
plt.scatter(x,y)
plt.plot(x,points(x,y,leastsq(x,y)), label='Least square', color='red')
plt.plot(x,(-X.dot(A[:2].reshape(2,1))),label='Total Least square', color="orange")
plt.plot(K, label='Ransac', color="green")
plt.legend()   
