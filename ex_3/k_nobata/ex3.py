import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#2次元散布図
def pdscatter2(df, x, y):
    """
    paramerters
    --
    df:pandas.core.frame.DataFrame
       dataframe of csv file
    x:numpy.ndarray
      explanatory variable
    y:numpy.ndarray
      explained variable
    """
    plt.scatter(df["x1"], df["x2"])
    plt.plot(x,y,color="pink")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid()
    plt.show()

#3次元散布図
def pdscatter3(df, x, y, z):
    """
    paramerters
    --
    df:pandas.core.frame.DataFrame
       dataframe of csv file
    x:numpy.ndarray
      explanatory variable
    y:numpy.ndarray
      explanatory variable
    z:numpy.ndarray
      explained variable
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")

    ax.plot(df["x1"], df["x2"], df["x3"] ,marker="o",linestyle='None')
    ax.plot_wireframe(x, y, z,color="pink")

    plt.grid()
    plt.show()

#2次元の理論値作る
def ideal2(df, K):
    """
    paramerters
    --
    df:pandas.core.frame.DataFrame
       dataframe of csv file
    K:int
      degree of regression equation
      
    returns
    --
    ide_x:np.ndarray
          explanatory variable
    ide_y:np.ndarray
          explained variable
    """
    w = regression2(df, K)
    ide_x = np.arange(min(df["x1"]), max(df["x1"]), 0.01)
    ide_y = np.zeros(len(ide_x))
    
    for i in range(len(ide_x)):       
        for j in range(K+1):
            ide_y[i] += w[j] * (ide_x[i] ** j)
            
    return ide_x, ide_y

#3次元の理論値作る
def ideal3(df, K1, K2):
    """
    paramerters
    --
    df:pandas.core.frame.DataFrame
       dataframe of csv file
    K1 and K2:int
              degree of regression equation
      
    returns
    --
    ide_z:np.ndarray
          explanatory variable
    ide_y:np.ndarray
          explanatory variable
    ide_z:np.ndarray
          explained variable
    """
    
    w = regression3(df, K1, K2)
    ide_x = np.arange(min(df["x1"]), max(df["x1"]), 0.01)    
    ide_y = np.linspace(min(df["x2"]), max(df["x2"]), len(ide_x))
    ide_z = np.zeros([len(ide_y),len(ide_x)])
    
    mesh_x, mesh_y = np.meshgrid(ide_x, ide_y)
    
    for i in range(len(ide_y)):
        for m in range(len(ide_x)):
            for j in range(K1+K2+1):
                if j <= K1:
                    ide_z[i,m] += (w[j] * (ide_x[m] ** j))
                else:
                    ide_z[i,m] += (w[j] * (ide_y[i] ** (j-K1)))
                    
    return mesh_x, mesh_y, ide_z

#2次元の回帰係数
def regression2(df,K):
    """
    paramerters
    --
    df:pandas.core.frame.DataFrame
       dataframe of csv file
    K:int
      degree of regression equatio
      
    returns
    --
    w:numpy.ndarray
      regression coefficient
    """
    ar_x = np.array(df["x1"])
    ar_y = np.array(df["x2"])
    
    #基底関数行列Φ
    Len_x = len(ar_x)
    phi = np.zeros([Len_x, K+1])
    
    for i in range(Len_x):
        for k in range(K+1):
            phi[i, k] = ar_x[i] ** k
    
    #最小二乗法
    w = np.linalg.inv(phi.T@phi)@phi.T@ar_y   #linalg.inv:逆行列　@:内積
    
    return w

#3次元の回帰係数
def regression3(df, K1, K2):
    """
    paramerters
    --
    df:pandas.core.frame.DataFrame
       dataframe of csv file
    K1 and K2:int
              degree of regression equatio
      
    returns
    --
    w:numpy.ndarray
      regression coefficient
    """
    ar_x = np.array(df["x1"])
    ar_y = np.array(df["x2"])
    ar_z = np.array(df["x3"])
    
    #基底関数行列Φ
    Len_x = len(ar_x)
    phi = np.zeros([Len_x, K1 + K2 +1]) #定数項は一つ
    
    for i in range(Len_x):
        for j in range(K1+1):
            phi[i, j] = ar_x[i] ** j
    for l in range(Len_x):
        for m in range(K2):
            phi[l, K1 + m + 1] = ar_y[l] ** (m+1)
    
    
    #最小二乗法
    w = np.linalg.inv(phi.T@phi)@phi.T@ar_z   #linalg.inv:逆行列　@:内積
    
    return w

def main():
    df1 = pd.read_csv("/Users/nobatakoki/B4輪行/B4Lecture-2021/ex_3/data1.csv")
    df2 = pd.read_csv("/Users/nobatakoki/B4輪行/B4Lecture-2021/ex_3/data2.csv")
    df3 = pd.read_csv("/Users/nobatakoki/B4輪行/B4Lecture-2021/ex_3/data3.csv")

    x1, y1 = ideal2(df1, 1)
    x2, y2 = ideal2(df2, 3)
    x3, y3, z3 = ideal3(df3,3,3)

    pdscatter2(df1, x1, y1)
    pdscatter2(df2, x2, y2)
    pdscatter3(df3, x3, y3, z3)

if __name__ =="__main__":
      main()
