import Particle as particule
import Vnet3d as vnet3d
import pandas as pd
import numpy as np


class PSOEngine :
    
    def __init__(self,numParticles,c1,c2,w):
        
        self.numParticles=numParticles
        
        self.c1=c1
        self.c2=c2
        self.w=w

    def evaluateFitness(position_i):
        # Read  data set (Train data from CSV file)
        csvmaskdata = pd.read_csv('dataprocess\\data\Segmentation3dMask.csv')
        csvimagedata = pd.read_csv('dataprocess\\data\Segmentation3dImage.csv')
        maskdata = csvmaskdata.iloc[:, :].values
        imagedata = csvimagedata.iloc[:, :].values
        # shuffle imagedata and maskdata together
        perm = np.arange(len(csvimagedata))
        np.random.shuffle(perm)
        imagedata = imagedata[perm]
        maskdata = maskdata[perm]
        Vnet3d = vnet3d.Vnet3dModule(96, 96, 16, channels=1)
        return Vnet3d.train(imagedata, maskdata,particule,6)

    
    def initParticles(self,particles):
        """
            Method to initialize the particles for PSO
        """
        for i in range(0,self.numParticles):
            p=particule(vnet3d.lunch)            
            particles.append(p)
    
    