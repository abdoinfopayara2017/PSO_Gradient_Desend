
import numpy as np


class Particle:

    def __init__(self,data):
        self.position_i=np.empty(len(data),dtype=object) # particle position
        self.velocity_i=np.empty(len(data),dtype=object) # particle velocity
        self.pos_best_i=np.empty(len(data),dtype=object) # best position individual
        self.fitness=0
        
        for i in range(0,len(data)):
            self.position_i[i]=data[i]
            self.velocity_i[i]=np.zeros(shape=data[i].shape())
        self.pos_best_i=self.position_i
    
    # evaluate current fitness
        

        

