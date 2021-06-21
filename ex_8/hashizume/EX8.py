import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import time



def load_pickledata(data):
    """
    answer_models : answer label of model (the model that generated the output series)
    output : output series
    PI : initial probability
    A : transition probability matrix
    B : output probability
    """
    answer_models = np.array(data["answer_models"])
    output = np.array(data["output"])
    models = data["models"]
    PI = np.array(models["PI"])
    A = np.array(models["A"])
    B = np.array(models["B"])
    return answer_models, output, PI, A, B


def forward_alg(answer_models, output, PI, A, B):
    """
    n_series : number of times
    len_series : length of series
    c : number of state
    m : number of output symbol
    q_i : i-th state
    o_k : k-th output symbol

    input
    ----------------------------------------------------------------------------------
    answer_models : answer label of model (the model that generated the output series)
    output : output series (number of observations,len_series : length of series)
    PI : initial probability (m,c,1)
    A : transition probability matrix (m,c,c) (a_i_j : probablity q_i => q_j)
    B : output probability (m,c,m) (b_i_k : probablity output o_k in q_i)

    output
    ---------------------------------
    array of HMM predicted by Forward algorithm
    """

    n_times, len_series = output.shape
    model_predict = np.empty(n_times)
    for i in range(len_series):
        alpha = PI[:, :, 0] * B[:, :, output[0, i]]
        for t in range(1, n_times):
            alpha = np.sum(A.T * alpha.T, axis=1).T * B[:, :, output[t, i]]
        model_predict[i] = np.argmax(np.sum(alpha, axis=1))
    return model_predict

