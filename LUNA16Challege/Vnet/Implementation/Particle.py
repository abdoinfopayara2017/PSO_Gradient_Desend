
import numpy as np


class Particle:

    def __init__(self,data):
        self.position=np.empty(len(data),dtype=object) # particle position
        self.velocity=np.empty(len(data),dtype=object) # particle velocity
        self.pos_best=np.empty(len(data),dtype=object) # best position individual
        self.fitness=0
        
        for i in range(0,len(data)):
            self.position[i]=data[i]
            self.velocity[i]=np.zeros(shape=data[i].shape())
        self.pos_best=self.position
    
    # evaluate current fitness
        

        

