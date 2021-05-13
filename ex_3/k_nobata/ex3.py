import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#2次元散布図
def pdscatter(df,x,y):
    plt.scatter(df["x1"], df["x2"])
    plt.plot(x,y,color="green")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid()
    plt.show()

#3次元散布図
def pdscatter3(df):
    fig = plt.figure()
    ax = Axes3D(fig)
    
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")

    ax.plot(df["x1"], df["x2"], df["x3"] ,marker="o",linestyle='None')

    plt.grid()
    plt.show()

#理論値作る
def ideal(df,K,w):
    """
    paramerters
    --
    df:pandas.core.frame.DataFrame
       dataframe of csv file
    K:int
      degree of regression equation
    w:numpy.ndarray
    
    returns
    ide_1:np.ndarray
          x
    ide_2:np.ndarray
          y
    """
    ide_1 = np.arange(min(df["x1"]), max(df["x1"]), 0.01)
    ide_2 = np.zeros(len(ide_1))
    for i in range(len(ide_1)):       
        for j in range(K+1):
            ide_2[i] += w[j] * (ide_1[i] ** j)
    return ide_1, ide_2

def regression(ar_x,ar_y,K):
    """
    paramerters
    df:pandas.core.frame.DataFrame
       dataframe of csv file
    K:int
      degree of regression equatio
      
    returns
    w:
    """
    #基底関数行列Φ
    Len_x = len(ar_x)
    phi = np.zeros([Len_x, K+1])
    
    for i in range(Len_x):
        for k in range(K+1):
            phi[i, k] = ar_x[i] ** k
    
    #最小二乗法
    w = np.linalg.inv(phi.T@phi)@phi.T@ar_y   #linalg:逆行列　@:内積
    
    return w

def main():

    df1 = pd.read_csv("/Users/nobatakoki/B4輪行/B4Lecture-2021/ex_3/data1.csv")
    df2 = pd.read_csv("/Users/nobatakoki/B4輪行/B4Lecture-2021/ex_3/data2.csv")
    df3 = pd.read_csv("/Users/nobatakoki/B4輪行/B4Lecture-2021/ex_3/data3.csv")
    
    ar1_x = np.array(df1["x1"])
    ar1_y = np.array(df1["x2"])
    ar2_x = np.array(df2["x1"])
    ar2_y = np.array(df2["x2"])
    ar3_x = np.array(df3["x1"])
    ar3_y = np.array(df3["x2"])
    ar3_z = np.array(df3["x3"])

    w1 = regression(ar1_x,ar1_y,1)
    x1, y1 = ideal(df1,1,w1)
    w2 = regression(ar2_x,ar2_y,3)
    x2, y2 = ideal(df2, 3, w2)

    pdscatter(df1, x1, y1)
    pdscatter(df2, x2, y2)
    pdscatter3(df3)


if __name__ == "__main__":
    main()