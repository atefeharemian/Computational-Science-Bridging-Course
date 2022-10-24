from operator import inv
import numpy as np
import math
from numpy.linalg import qr, norm
from math import sqrt
from pprint import pprint
import rogues

def QRcpivot(A):
    def GivensPar(x,y):
            if abs(x) > abs(y):
                t = y/x; c = 1/np.sqrt(1+t**2); s = c*t
            else:
                t = x/y; s = 1/np.sqrt(1+t**2); c = s*t
            return c,s

    def GivensProd(c,s,j,k,A):
        A[[j,k],:] = np.matmul([[c,s],[-s,c]],A[[j,k],:])
        return A

    def Q_R(A, col):
        m = A.shape[0]
        n = A.shape[1]
        Q = np.eye(m)
        A1 = A.copy()

        for k in range(m - 1,col,-1):
            c,s = GivensPar(A[col,col],A[k,col])
            A=GivensProd(c, s, col, k, A)
            R = A
            G = np.eye(m)
            G[col, col] = c
            G[col, k] = s
            G[k, col] = -s
            G[k, k] = c
            P=np.transpose(G)
            Q = Q @ P
        return Q.T


    m = A.shape[0]
    n = A.shape[1]
    A1 = A.copy()
    PP = []
    QQ = []
    for col in range(n):
        Q = np.eye(m)
        P=np.eye(n)
        Amax = A[col:, col:]
        max_col = np.argmax([np.linalg.norm(Amax[:, c]) for  c in range(Amax.shape[1])]) + col
        P.T[[col, max_col]] = P.T[[max_col, col]]
        A = A @ P
        Q = Q_R(A.copy(), col)
        A = Q @ A
        PP.append(P)
        QQ.append(Q)

    QT = QQ[0]
    for i in range(1, len(QQ)):
        QT = QQ[i] @ QT

    PT = PP[0]
    for i in range(1, len(PP)):
        PT = PP[i] @ PT   

    R=QT@A1@PT
    N=np.linalg.norm(A1@PT-QT@R)

    return QT.T, R, PT, len(QQ)


def LeastSquaresQRcPivot(A, b):
    Q, R, P, _ = QRcpivot(A)
    
    r = np.linalg.matrix_rank(R)
    m, n = P.shape
    R11 = R[:r,:r]
    ytilde = np.linalg.inv(R11) @ Q.T[:r] @ b 
    yhat = np.zeros((m - r, 1))
    y = np.concatenate((ytilde, yhat), axis = 0)
    x = P @ y
    return x

def LeastSquaresQRcPivot2(A, b):
    Q, R, P, _ = QRcpivot(A)
    r = np.linalg.matrix_rank(R)
    m, n = P.shape
    yhat = np.ones((m - r, 1))

    R11 = R[:r,:r]
    R12 = R[:r, r:]

    ytilde = np.linalg.inv(R11) @ (Q.T[:r] @ b - R12 @ yhat) 
    
    y = np.concatenate((ytilde, yhat), axis = 0)
    x = P @ y
    return x

A = np.array(
[[1, -1, 2, 0],
[1, 2, -1, 3],
[1 ,1, 0, 2],
[1, -1, 2, 0],[1,3,-1,4]])

b = np.array([1,-1,0,1,0]).reshape(-1, 1)

n = 4
B = rogues.neumann(n)
A = B[0]
b = np.zeros((A.shape[0], 1))

x = LeastSquaresQRcPivot2(A, b)


print(1)













