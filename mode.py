import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


import pickle
import sklearn
def training(x,y):
    lr=sklearn.linear_model.LinearRegression()
    # lr.fit(df['X'].values,df['Y'].values)
    lr.fit(x,y)
    pickle.dump(lr,open("myseries1.pkl","wb"))
    
def plotting(x,y):
    print("Plotting..!!!!")
    print("x=",x)
    print("Y=",y)
    plt.clf()
    plt.plot(x,y,marker='*')
    plt.title("Graph for your series")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    filename="static/graph.png"
    if os.path.exists(filename):
        print("file deleted")
        os.remove(filename)
    plt.savefig(filename)
    plt.show()
