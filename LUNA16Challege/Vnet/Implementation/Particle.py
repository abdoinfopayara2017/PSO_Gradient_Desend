
import numpy as np
import tensorflow as tf


class Particle:

    def __init__(self,data):
        self.position = np.empty(len(data),dtype=object) # particle position
        self.velocity = np.empty(len(data),dtype=object) # particle velocity
        self.best_pos = np.empty(len(data),dtype=object) # best position individual
        self.partial_derivative = [] #np.empty(len(data),dtype=object) # particle position
        self.fitness = tf.Variable(0,dtype=tf.float32)
        self.fitness_best_pos = tf.Variable(0,dtype=tf.float32)
        self.dot_derivate = tf.Variable([0.0,0.0,0.0,0.0,0.0],name="dot_derivate")
        self.pre_activations = []
        self.activations = []
        
        for i in range(0,len(data)):            
            self.position[i]=tf.Variable(initial_value=tf.constant(0.0,shape=data[i].get_shape()),
                                         shape=data[i].get_shape(),dtype=tf.float32)            
            self.position[i].assign(data[i])
            self.best_pos[i]=tf.Variable(initial_value=tf.constant(0.0,shape=data[i].get_shape()),
                                         shape=data[i].get_shape(),dtype=tf.float32)
            self.velocity[i]=tf.Variable(shape=data[i].get_shape(),dtype=tf.float32,
                                             initial_value=tf.constant(0.001,shape=data[i].get_shape()))
             #np.zeros(shape=data[i].get_shape())

            #self.partial_derivative[i]=np.zeros(shape=data[i].get_shape())
            #self.velocity[i]=tf.convert_to_tensor(self.velocity[i],dtype=tf.float32)
            #self.partial_derivative[i]=tf.convert_to_tensor(self.partial_derivative[i],dtype=tf.float32)
        #self.best_pos=self.position.copy()
        #self.fitness_best_pos=self.fitness
        
    
    
        

        

