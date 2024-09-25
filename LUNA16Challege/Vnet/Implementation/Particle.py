
import numpy as np
import tensorflow as tf


class Particle:

    def __init__(self,data):
        self.position = np.empty(len(data),dtype=object) # particle position
        self.velocity = np.empty(len(data),dtype=object) # particle velocity
        self.best_pos = np.empty(len(data),dtype=object) # best position individual
        self.partial_derivative = np.empty(len(data),dtype=object) # particle position
        self.fitness = tf.Variable(0, name="fitness")
        self.fitness_best_pos = tf.Variable(0, name="fitness_best_pos")
        self.dot_derivate = tf.Variable([0.0,0.0,0.0,0.0,0.0],name="dot_derivate")
        self.pre_activations = []
        self.activations = []
        
        for i in range(0,len(data)):            
            self.position[i]=data[i]
            self.velocity[i]=np.zeros(shape=data[i].get_shape())
            self.partial_derivative[i]=np.zeros(shape=data[i].get_shape())
            self.velocity[i]=tf.convert_to_tensor(self.velocity[i],dtype=tf.float32)
            self.partial_derivative[i]=tf.convert_to_tensor(self.partial_derivative[i],dtype=tf.float32)
        self.best_pos=self.position.copy()
        self.fitness_best_pos=self.fitness
        
    
    
        

        

