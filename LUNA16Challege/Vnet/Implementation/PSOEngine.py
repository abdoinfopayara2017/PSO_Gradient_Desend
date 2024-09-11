from Particle import Particle
import Vnet3d as vnet3d
from Vnet3d import lunch
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path

import sys

sys.path.insert(0, 'E:/LUNA 16/PSOGD/LUNA16Challege/Vnet')

from layer import (full_conv3d , valid_conv3d)


class PSOEngine :
    
    def __init__(self,num_particles,c1,c2,w):
        
        self.num_particles=num_particles        
        self.c1=c1
        self.c2=c2
        self.w=w

    def evaluate_fitness(self,position):
        path_mask = Path(__file__).parent / "..\..\dataprocess\data\Segmentation3dMask.csv"
        path_data = Path(__file__).parent / "..\..\dataprocess\data\Segmentation3dImage.csv"
        with path_mask.open() as f_m:
            with path_data.open() as f_d:

                # Read  data set (Train data from CSV file)
                csvmaskdata = pd.read_csv(f_m)
                csvimagedata = pd.read_csv(f_d)
                maskdata = csvmaskdata.iloc[:, :].values
                imagedata = csvimagedata.iloc[:, :].values
                # shuffle imagedata and maskdata together
                perm = np.arange(len(csvimagedata))
                np.random.shuffle(perm)
                imagedata = imagedata[perm]
                maskdata = maskdata[perm]
                Vnet3d = vnet3d.Vnet3dModule(96, 96, 16, channels=1)
                return Vnet3d.train(imagedata, maskdata,position,6)

    
    def init_particles(self,list_particules):
        """
            Method to initialize the particles for PSO
        """      
        
        for i in range(0,self.num_particles):
            
            list_weights_all_layers = lunch()
            
            p=Particle(list_weights_all_layers)                      
            list_particules.append(p)
        return list_particules
    
    
    def find_gbest(self,particles):
       particles = sorted(particles, key=lambda Particle: Particle.fitness.eval())   # sort by fitness
       return particles[0].position , particles[0].fitness.eval() 

    def update_velocity(self,particule,gbest,r1,r2):
       inertia_term = np.empty(len(particule.velocity),dtype=object)
       
       for i in range (0,len(particule.velocity)) : 
        inertia_term[i] = tf.multiply(particule.velocity[i] , self.w) 
        
       difference1 = np.empty(len(particule.best_pos),dtype=object)
       for i in range (0,len(particule.best_pos)) :
        difference1[i] = tf.subtract (particule.best_pos[i] , particule.position[i])
        
       c1_timesr1 = np.empty(len(r1),dtype=object)
       for i in range (0,len(r1)) : 
        c1_timesr1[i] = tf.multiply(tf.convert_to_tensor(r1[i],dtype=tf.float32) , self.c1)        
        
       cognitive_term = np.empty(len(difference1),dtype=object)
       
       for i in range (0,len(difference1)) :
        cognitive_term[i] = tf.multiply(c1_timesr1[i] , difference1[i])
        
       difference2 = np.empty(len(particule.position),dtype=object)
       for i in range (0,len(particule.position)) : 
        difference2[i] = tf.subtract(gbest[i] , particule.position[i])
        
       c2_timesr2 = np.empty(len(r2),dtype=object) 
       for i in range (0,len(r2)) :
        c2_timesr2[i] = tf.multiply(tf.convert_to_tensor(r2[i],dtype=tf.float32) , self.c2)
        
       social_term = np.empty(len(difference2),dtype=object)
       for i in range (0,len(difference2)) : 
         social_term[i] = tf.multiply(difference2[i] , c2_timesr2[i])
        
       for i in range (0,len(particule.velocity)) : 
         particule.velocity[i] = tf.add(tf.add(inertia_term[i] , cognitive_term[i]) , social_term[i])
         particule.velocity[i]=tf.abs(particule.velocity[i])
       
       return particule 

    def update_partial_derivatives_v0(self,particule) :
        #chain rule
        index = 0
        #for i in range(0,len(particule.partial_derivative)//2):  
        for i in range(0,1):              
            
            # Weight matrix layer k
            Ωk= particule.position[-(2*(i+1))]                       
            # derivative Bias Bk or Dli/Df(k)
            particule.partial_derivative[index]=particule.dot_derivate            
            particule.partial_derivative[index] =tf.where(tf.greater_equal(particule.partial_derivative[index],tf.constant(0,dtype=tf.float32))\
                                                    ,tf.ones_like(particule.partial_derivative[index]),\
                                                         tf.zeros_like(particule.partial_derivative[index]))
            # derivative Ωk
            # Hk activation layer k                        
            index = + 1
            matrix = particule.activations[-(i+2)]
            particule.partial_derivative[index] = tf.matmul(particule.dot_derivate,\
                                                            tf.transpose(matrix))
            particule.partial_derivative[index]=tf.where(tf.greater_equal(particule.partial_derivative[index],0)\
                                                    ,tf.constant(1),tf.constant(-1))
            # Calculate Dli/Df(k-1)
            if( (i+2) <= len(particule.pre_activations)) : 
             matrix=particule.pre_activations[-(i+2)]
             matrix=tf.where(tf.greater(matrix,0)\
                    ,tf.constant(1),tf.constant(0))
             particule.dot_derivate= tf.math.multiply(
                 matrix,tf.math.multiply(particule.dot_derivate ,\
                                tf.transpose(Ωk)))
            index = + 1
        
        particule.partial_derivative=particule.partial_derivative.reverse()
        return particule
    
    def update_partial_derivatives(self,particule) :
        #chain rule
        index = 0
        for i in range(0,len(particule.partial_derivative)//2):  
          # Weight matrix layer k
          Ωk= particule.position[-(2*(i+1))]                       
          # derivative Bias Dli/Bk 
          particule.partial_derivative[index]=particule.dot_derivate            
          particule.partial_derivative[index] =tf.where(tf.greater_equal(particule.partial_derivative[index],tf.constant(0,dtype=tf.float32))\
                ,tf.ones_like(particule.partial_derivative[index]),\
               - tf.ones_like(particule.partial_derivative[index]))
          # derivative Ωk
          # Hk activation layer k                        
          index = + 1
          Hk = particule.activations[-(i+2)]
          particule.partial_derivative[index] = valid_conv3d(Hk , particule.dot_derivate)
          particule.partial_derivative[index]=tf.where(tf.greater_equal(particule.partial_derivative[index],tf.constant(0,dtype=tf.float32))\
                ,tf.ones_like(particule.partial_derivative[index]),\
               - tf.ones_like(particule.partial_derivative[index]))
          # Calculate Dli/DH(k-1)
          if((i+2) <= len(particule.pre_activations)) :
            # 180-degree rotated Filter
            Ωk = tf.image.rot90(Ωk, k=-2)
            filter_shape = Ωk.shape.as_list()
            pad_depth = filter_shape[0] - 1
            pad_height = filter_shape[1] - 1
            pad_width = filter_shape[2] - 1
            padding_tensor = tf.pad(particule.dot_derivate,[[0,0],[pad_depth,pad_depth],\
                                    [pad_height,pad_height],[pad_width,pad_width],[0,0]])  
            padding_tensor = full_conv3d(padding_tensor,Ωk)
            matrix=particule.pre_activations[-(i+2)]
            matrix=tf.where(tf.greater(matrix,tf.constant(0,dtype=tf.float32))\
                ,tf.ones_like(matrix),\
                 tf.zeros_like(matrix))
            
            # Calculate Dli/DF(k-1)
            particule.dot_derivate= tf.math.multiply(\
                 matrix,padding_tensor)
          index = + 1
        
        particule.partial_derivative=particule.partial_derivative.reverse()
        return particule
   
    def updatePosition(particule):
        particule.position = particule.position - tf.math.multiply(
            particule.velocity,particule.partial_derivative)

    def padding_same(tensor , filter) :
      input_shape = tensor.shape.as_list()
      filter_shape = filter.shape.as_list() 
      #stride equal 1 in each dimension
      out_size =  input_shape
      # Total Padding Required
      P_depth = filter_shape[0] - 1
      P_height = filter_shape[1] - 1
      P_width = filter_shape[2] - 1
      padding = [[P_depth // 2 , P_depth - (P_depth//2)] , \
                 [P_height // 2 , P_height - (P_height//2)] , \
                 [P_width // 2 , P_width - (P_width//2)] ]
      tensor = tf.pad(tensor , padding)
      return tensor

       









    
    