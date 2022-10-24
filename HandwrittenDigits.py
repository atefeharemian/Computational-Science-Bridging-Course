import numpy as np
import matplotlib.pyplot as plt

TrainDigits = np.load("TrainDigits.npy")
TrainLabels=np.load("TrainLabels.npy")
TestDigits =np.load('TestDigits.npy')
TestLabels =np.load('TestLabels.npy')
#Training


debn = round(TrainDigits.shape[1]/10)
V = np.zeros((10, TestDigits.shape[1]))
m = TrainDigits.shape[0]
for i in range(10):
    print(i)
    ind = [k for k in range(TrainLabels.shape[1]) if TrainLabels[0][k] == i]
    A = TrainDigits[:, ind[:debn]]
    ll = TrainLabels[0, ind[:debn]]
    (U,_,_)=np.linalg.svd(A)
    Ur = U[:, :10]
    d = TestDigits
    v = (np.eye(m) - Ur @ Ur.T) @ d
    #v = Ur.T @TestDigits[:, :30]
    vn = np.linalg.norm(v, axis = 0)
    V[i, :] = vn
predict = np.argmin(V, axis = 0)
label = TestLabels[0, :]
accuracy = np.sum(predict == label)/len(predict)*100
print(accuracy)




