import Particle as particule
import Vnet3d as vnet3d
import pandas as pd
import numpy as np
import tensorflow as tf


class PSOEngine :
    
    def __init__(self,num_particles,c1,c2,w):
        
        self.num_particles=num_particles        
        self.c1=c1
        self.c2=c2
        self.w=w

    def evaluate_fitness(position):
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
        return Vnet3d.train(imagedata, maskdata,position,6)

    
    def init_particles(self,particles):
        """
            Method to initialize the particles for PSO
        """
        for i in range(0,self.num_particles):
            p=particule(vnet3d.lunch)            
            particles.append(p)
    
    
    def find_gbest(particles):
       sorted(particles, key=lambda Particle: Particle.fitness)   # sort by fitness
       return particles[0].position

    def update_velocity(self,particule,gbest,r1,r2):
        inertia_term = particule.velocity * self.w 
        difference1 = particule.best_pos - particule.position
        c1_timesr1 = r1 * self.c1        
        cognitive_term = c1_timesr1 * difference1
        difference2 = gbest - particule.position
        c2_timesr2 = r2 * self.c2
        social_term = difference2 * c2_timesr2

        particule.velocity = inertia_term + cognitive_term + social_term
        particule.velocity=np.absolute(particule.velocity)

    def update_partial_derivatives(particule) :
        #chain rule
        for i in range(0,len(particule.partial_derivative)/2):            
            
            # Weight matrix layer k
            Ωk= particule.position[-(2*(i+1))]                       
            # derivative Bias Bk or Dli/Df(k)
            particule.partial_derivative.append(particule.dot_derivate)
            particule.partial_derivative[i][particule.partial_derivative[i] >= 0] = 1
            particule.partial_derivative[i][particule.partial_derivative[i] < 0] = -1
            # derivative Ωk
            # Hk activation layer k                        
            if( i <len(particule.partial_derivative)/2) :
                matrix = particule.activations[-(i+2)]
                
            else : matrix =  particule.activations[-(i+1)]    
            matrix = np.array(matrix)
            particule.partial_derivative.append(
                particule.dot_derivate * matrix.transpose())
            particule.partial_derivative[i+1][particule.partial_derivative[i+1] >= 0] = 1
            particule.partial_derivative[i+1][particule.partial_derivative[i+1] < 0] = -1
            
            # Calculate Dli/Df(k-1)
            if( i <len(particule.partial_derivative)/2) : 
                matrix=np.array(particule.pre_activations[-(i+2)])
                matrix[matrix > 0] = 1
                matrix[matrix < 0] = 0
                particule.dot_derivate= tf.math.multiply(
                    matrix,particule.dot_derivate * Ωk.transpose())
            
        particule.partial_derivative.reverse()
    
    def updatePosition(particule):
        particule.position = particule.position - tf.math.multiply(
            particule.velocity,particule.partial_derivative)











    
    