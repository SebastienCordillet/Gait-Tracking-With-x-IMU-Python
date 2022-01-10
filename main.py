# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 13:06:06 2021

@author: cordillet
"""

def getIMU(filename, chanelName, acc=True, gyr=True, mag=True, name=""):
    from pyomeca import Analogs
    #list des channels à importer
    channels=[]
    if acc:
        channels.append(chanelName+'_ACC_X')
        channels.append(chanelName+'_ACC_Y')
        channels.append(chanelName+'_ACC_Z')
    if gyr:
        channels.append(chanelName+'_GYRO_X')
        channels.append(chanelName+'_GYRO_Y')
        channels.append(chanelName+'_GYRO_Z')
    if mag:
        channels.append(chanelName+'_MAG_X')
        channels.append(chanelName+'_MAG_Y')
        channels.append(chanelName+'_MAG_Z')       
   
    #import du c3d à partir du nom du fichier
    analogs=Analogs.from_c3d(filename, usecols=channels)
    
    #set channels names    
    # if name is not None:
    newChannels = [w.replace(chanelName+"_", name) for w in channels]
    analogs.coords["channel"]=newChannels
        
    return(analogs)


def selTime(times, rate):
    import numpy as np
    time = np.arange(start=times[0], stop=times[1], step=1 / rate)
    
    return(time)

def getPCAaxis(gyr):
    from sklearn.decomposition import PCA
    pca=PCA(n_components=1)
    pca.fit(gyr.T)
    return(pca.components_[0])
    