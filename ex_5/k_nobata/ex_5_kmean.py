#!/usr/bin/env python
# coding: utf-8

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 

#ファイル読み込み
def file(num):
    """
    paramerter
    ---
    num:int
        file number
    
    return
    ---
    data:numpy.ndarray
        csv data
    """
    f_name = "/Users/nobatakoki/B4輪行/B4Lecture-2021/ex_5/data"+str(num)+".csv"
    df = pd.read_csv(f_name)
    data = df.values
    return data

#2次元散布図
def scatter_2d(data, save, K, clu):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    save:str
         save file name
    K:int
      the number of cluster
    clu:numpy.ndarray
        cluster
    """
    fig = plt.figure(figsize=(8,6))
    
    cmap = plt.get_cmap("tab10")
    for i in range(K):
        cdata = data[clu==i]
        x = cdata[:,0]
        y = cdata[:,1]
        #計算値とデータをプロット
        plt.scatter(x, y,color=cmap(i))
    
    #ラベル作成
    plt.xlabel("x",fontsize=18)
    plt.ylabel("y",fontsize=18)
    
    plt.grid() 
    #plt.legend() #凡例の追加
    plt.tight_layout()
    #plt.savefig(save)
    plt.show()

#3次元散布図
def scatter_3d(data, save, K, clu):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    save:str
         save file name
    K:int
      the number of cluster
    clu:numpy.ndarray
        cluster
    """
    fig = plt.figure(figsize=(32,24))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    cmap = plt.get_cmap("tab10")

    #クラスタごとで描画
    for i in range(K):
        cdata = data[clu==i]
        x = cdata[:,0]
        y = cdata[:,1]
        z = cdata[:,2]
        
        ax1.plot(x, y, z, marker="o", linestyle='None', color=cmap(i))
        ax2.plot(x, y, z, marker="o", linestyle='None', color=cmap(i))
        ax3.plot(x, y, z, marker="o", linestyle='None', color=cmap(i))

    
    #ラベル作成
    ax1.set_xlabel("x",fontsize=24)
    ax1.set_ylabel("y",fontsize=24)
    ax1.set_zlabel("z",fontsize=24)
    ax2.set_xlabel("x",fontsize=24)
    ax2.set_ylabel("y",fontsize=24)
    ax2.set_zlabel("z",fontsize=24)
    ax3.set_xlabel("x",fontsize=24)
    ax3.set_ylabel("y",fontsize=24)
    ax3.set_zlabel("z",fontsize=24)

    plt.grid()
    #plt.legend()  #凡例の追加
    ax2.view_init(elev=0, azim=-90) #回転
    ax3.view_init(elev=0, azim=0)
    plt.tight_layout()
    #plt.savefig(save)
    plt.show()

#ミニマックス法
def minimax(data, K):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    K:int
      the number of cluster
      
    return
    ----
    cent:numpy.ndarray
         center of cluster
    clu:numpy.ndarray
        cluster
    """
    num, dim = data.shape
    cen = [] #中心のインデックス
    cen = np.append(cen, random.randint(0, num-1))
    dis = np.zeros((K, num))
    cent = np.zeros((K, dim))
    for k in range(K):
        cent[k] = data[int(cen[k])]
        r = np.sum((data - data[int(cen[k])])**2, axis = 1) #距離計算
        dis[k] =  r #距離保存
        
        cen = np.append(cen, np.argmax(np.min(dis[:k+1], axis = 0)))  #距離最大の次の中心
    clu = np.argmin(dis, axis=0)
    return cent, clu

#kmeans++
def kplus(data, K):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    K:int
      the number of cluster
      
    return
    ----
    cent:numpy.ndarray
         center of cluster
    clu:numpy.ndarray
        cluster
    """
    num, dim = data.shape
    cen = [] #中心のインデックス
    cen = np.append(cen, random.randint(0, num-1))
    dis = np.zeros((K, num))
    cent = np.zeros((K, dim))
    pr = np.zeros(num)
    for k in range(K):
        #距離計算
        cent[k] = data[int(cen[k])]
        r = np.sum((data - data[int(cen[k])])**2, axis = 1) #距離計算
        dis[k] =  r #距離保存
        
        #確率作成
        pr = np.min(dis[:k+1], axis = 0)
        pr = pr / np.sum(pr)
        
        #次の中心
        x = np.random.choice(np.arange(num), 1, p=pr)
        cen = np.append(cen, x)

    clu = np.argmin(dis, axis=0)
    return cent, clu

#初期値決定方法
def method(data, K, mname):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    K:int
      the number of cluster
    mname:str
      method name
      
    return
    ----
    cent:numpy.ndarray
         center of cluster
    clu:numpy.ndarray
        cluster
    """
    if mname == "minimax":
        cen, clu = minimax(data, K)
    elif mname == "kplus":
        cen, clu = kplus(data, K)
    elif mname == "LGB":
        cen, clu = LBG(data, K)
    else:
        print("error:method name")
        
    return cen, clu

#kmeanアルゴリズム
def kmean(data, K, cen, clu):
    """
    paramerters
    --
    data:numpy.ndarray
         csv data
    K:int
      the number of cluster
    cen:numpy.ndarray
         center of cluster
    clu:numpy.ndarray
        cluster
      
    return
    ----
    ncen:numpy.ndarray
         center of cluster
    clu:numpy.ndarray
        cluster
    """
    num, dim = data.shape
    clusters = []
    dis = np.zeros((K, num))
    while (True):
        ncen = np.zeros((K, dim))
        for k in range(K):
            ncen[k] = np.mean(data[clu == k]) #重心計算
            r = np.sum((data - ncen[k])**2, axis = 1) #距離計算
            dis[k] = r #距離保存
                       
        clu = np.argmin(dis, axis=0)
        if np.allclose(cen,ncen) is True:#変化がないなら終了
            break
        cen = ncen
    return ncen, clu

def main(args):
    n = args.fnum
    c = args.clo
    save = args.save
    data = file(n)
    
    #クラスタ計算&描画
    cen, clu_m = method(data, c)
    ncen, clu_mk = kmean(data, c, cen, clu_m)
    if data.shape[1] == 2:
        scatter_2d(data, save, c, clu_mk)
    
    elif data.shape[1] == 3:
        scatter_3d(data, save, c, clu_mk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fnum", default="1")
    parser.add_argument("--clo", default=4)
    parser.add_argument("--save", default="data1.png")
    parser.add_argument("--method", default="minimax")
    args = parser.parse_args()

    main(args)