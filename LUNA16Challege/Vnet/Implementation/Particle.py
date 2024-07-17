
import numpy as np


class Particle:

    def __init__(self,data):
        self.position=np.empty(len(data),dtype=object) # particle position
        self.velocity=np.empty(len(data),dtype=object) # particle velocity
        self.best_pos=np.empty(len(data),dtype=object) # best position individual
        self.partial_derivative=np.empty(len(data),dtype=object) # particle position
        self.fitness=0
        self.dot_derivate=0
        self.pre_activations=[]
        self.activations=[]
        
        for i in range(0,len(data)):
            self.position[i]=data[i]
            self.velocity[i]=np.zeros(shape=data[i].shape())
            self.partial_derivative[i]=np.zeros(shape=data[i].shape())
        self.best_pos=self.position.copy()
    
    
        

        

