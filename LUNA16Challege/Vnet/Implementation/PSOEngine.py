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
    
    def __init__(self,num_particles,c1,c2,w,index_in_epoch):
        
        self.num_particles=num_particles        
        self.c1=c1
        self.c2=c2
        self.w=w
        self.index_in_epoch = index_in_epoch

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
                return Vnet3d.train(imagedata, maskdata,position,6,self.index_in_epoch)

    
    def init_particles(self,list_particules):
        """
            Method to initialize the particles for PSO
        """      
        
        for i in range(0,self.num_particles):
            
            list_weights_all_layers = lunch()
            
            p=Particle(list_weights_all_layers)                      
            
            list_particules.append(p)
        return list_particules
    
    
    def find_gbest(self,particles,gbest,gbest_fitness):
       
       particles = sorted(particles, key=lambda Particle: Particle.fitness_best_pos.numpy())   # sort by fitness
       
       gbest_fitness.assign(particles[0].fitness_best_pos)
       
       for w in range(0,len(particles[0].position)):
        gbest[w].assign (particles[0].position[w])
       
       return gbest , gbest_fitness 

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
         particule.velocity[i].assign(tf.add(tf.add(inertia_term[i] , cognitive_term[i]) , social_term[i]))
         particule.velocity[i].assign(tf.abs(particule.velocity[i]))
       
       return particule 
        
    def padding_same(tensor , filter) :
      input_shape = tensor.shape.as_list()
      filter_shape = filter.shape.as_list() 
      #stride equal 1 in each dimension
      out_size =  input_shape
      # Total Padding Required
      P_depth = filter_shape[0] - 1
      P_height = filter_shape[1] - 1
      P_width = filter_shape[2] - 1
      padding = [[0,0],
        [P_depth // 2 , P_depth - (P_depth//2)] , \
                 [P_height // 2 , P_height - (P_height//2)] , \
                 [P_width // 2 , P_width - (P_width//2)],
                 [0,0] ]
      tensor = tf.pad(tensor , padding)
      return tensor


    def update_partial_derivatives(self,particule) :
                
        # update D L/D B33
        index = -1
        d_b33=tf.reduce_sum(particule.dot_derivate)
        particule.partial_derivative[index]= tf.where(
            tf.greater_equal(d_b33,tf.constant(0,dtype=tf.float32))\
           ,tf.ones_like(d_b33),- tf.ones_like(d_b33))  

        # update D L/D ω33
        index = -2
        H45 = particule.activations[index]
        ω33 = tf.zeros(shape = particule.position[index].shape,dtype=tf.dtypes.float32)
        for batch in range (0,6) :
            par_deriv = particule.dot_derivate[batch:batch+1,:,:,:,0:1]            
            input = H45[batch:batch+1,:,:,:,:]
            for channel in range(0,32) :
                input_ch = input[:,:,:,:,channel:channel+1]
                filter = particule.position[index]
                input_ch = PSOEngine.padding_same(input_ch,filter)
                par_deriv = tf.reshape(par_deriv,(16,96,96,1,1))
                outpout = valid_conv3d (input_ch,par_deriv)              
                indices = tf.constant([[0,0,0,channel,0]])                
                ω33 = tf.tensor_scatter_nd_add(ω33,indices,
                         tf.reduce_sum(outpout,[0,1,2,3]))
        particule.partial_derivative[index]=tf.where(
            tf.greater_equal(ω33,tf.constant(0,dtype=tf.float32))\
           ,tf.ones_like(ω33),- tf.ones_like(ω33))

        # update D L/D H45        
        filter =  particule.position[index]
        list_batch = []
        for batch in range (0,6) :
            par_deriv = particule.dot_derivate[batch:batch+1,:,:,:,0:1]            
            list_channel = []
            for channel in range(0,32) :                
                filter_channel = filter[:,:,:,channel:channel+1,:]
                outpout = valid_conv3d (par_deriv,filter_channel)
                list_channel.append(outpout[0,:,:,:,0])                
            tensor_stacked_channel = tf.stack(list_channel, axis=3)
            list_batch.append(tensor_stacked_channel)       
        d_h45 = tf.stack(list_batch, axis=0)        
        
        # update D L/D H44
        d_h44 = d_h45
       
        # update D L/D F32
        index = -3
        H44 = particule.activations[index]
        H44 = tf.where(
            tf.greater(H44,tf.constant(0,dtype=tf.float32))\
           ,tf.ones_like(H44), tf.zeros_like(H44))
        d_f32 = tf.multiply(H44 , d_h44)
        # update D L/D B32
        d_b32 = tf.reduce_sum(d_f32,[0,1,2,3])
        particule.partial_derivative[index] = d_b32
           

        return particule
   
    def update_position(self,particule):
        for i in range(0,len(particule.position)) :
         particule.position[i].assign (tf.subtract(particule.position[i], tf.math.multiply(
            particule.velocity[i],particule.partial_derivative[i])))
        return particule

    
       









    
    