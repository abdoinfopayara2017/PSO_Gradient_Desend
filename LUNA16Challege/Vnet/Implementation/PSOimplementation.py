import numpy as np


from PSOEngine import PSOEngine
import tensorflow as tf
import pickle
import os


class PSOimplemntation :

    
    def __init__(self,nb_iteration,swarm_size,cognitive,social,weight):
        self.nb_iteration=nb_iteration
        self.swarm_size=swarm_size
        self.cognitive=cognitive
        self.social=social
        self.weight=weight
    
    def saveVariables(self, path ,variables): #where 'variables' is a list of variables
        with open(path + "model.txt", 'wb+') as file:
           pickle.dump(variables, file)
    
    def retrieveVariables(self, filename):
        variables = []
        with open(str(filename), 'rb') as file:
            variables = pickle.load(file)
        return variables
     
    def lunch(self,retrive):     

     PSO=PSOEngine(self.swarm_size,self.cognitive,self.social,self.weight,0)
     list_particules=[]
     path = "log\\segmeation\\" + "model\\" 
     if not os.path.exists(path) :
        os.makedirs(path)
     
     if retrive :
        list_particules=self.retrieveVariables(path+"model.txt")
        gbest=np.empty(len(list_particules[0].position),dtype=object)
        for w in range(0,len(gbest)):            
          gbest[w]=tf.Variable(initial_value=tf.constant(0.0,shape=list_particules[0].position[w].get_shape()),
                    shape=list_particules[0].position[w].get_shape(),dtype=tf.float32)
        gbest_fitness=tf.Variable(0,dtype=tf.float32)
     else :
        list_particules=[]    
        list_particules=PSO.init_particles(list_particules)
        gbest=np.empty(len(list_particules[0].position),dtype=object)
        for w in range(0,len(gbest)):            
          gbest[w]=tf.Variable(initial_value=tf.constant(0.0,shape=list_particules[0].position[w].get_shape()),
                    shape=list_particules[0].position[w].get_shape(),dtype=tf.float32)      
        gbest_fitness=tf.Variable(0,dtype=tf.float32)
     
        # initialisation des parametres 
        for p in range(0,len(list_particules)) :      
          list_particules[p].fitness , PSO.index_in_epoch = \
           PSO.evaluate_fitness(list_particules[p].position)
        
          """
          with tf.device('/gpu:0'):
            for w in range(0,len(list_particules[p].partial_derivative)) :         
              list_particules[p].partial_derivative[w]=tf.where(
              tf.greater_equal(list_particules[p].partial_derivative[w],tf.constant(0,dtype=tf.float32))\
              ,tf.ones_like(list_particules[p].partial_derivative[w]),- tf.ones_like(list_particules[p].partial_derivative[w]))
          """      
          # for each particle i do Pbesti = xi;
          list_particules[p].fitness_best_pos.assign( list_particules[p].fitness)     
          for w in range(0,len(list_particules[p].position)):
           list_particules[p].best_pos[w].assign (list_particules[p].position[w])
          
     # Find best particle in set
     gbest , gbest_fitness=PSO.find_gbest(list_particules,gbest,gbest_fitness)
     
     last_fitness = gbest_fitness.numpy()
            
     # PSO boucle
     # for each iteration do
     with tf.device('/gpu:0'):
      for epoch in range(0,20) : 
        for i in range(0,self.nb_iteration) :
        # for each particle p do
          for j in range(0,len(list_particules)):
            #update the velocity and the position
            # Initialize the random vectors for updates
            r1=np.empty(len(list_particules[0].position),dtype=object) 
            r2=np.empty(len(list_particules[0].position),dtype=object) 
            for r in range(0,len(list_particules[0].position)):
                r1[r]=np.random.rand(*list_particules[0].position[r].get_shape())
                r2[r]=np.random.rand(*list_particules[0].position[r].get_shape())           
            
            #print('fitness for  particule %d is %.5f and best is %.5f' % (j,list_particules[j].fitness.numpy(),\
                                                                          #list_particules[j].fitness_best_pos.numpy()))
            list_particules[j] = PSO.update_velocity(list_particules[j],gbest,r1,r2)            
            #if j==0 :
              #print('vilocity for particule %d is  ' , j,list_particules[j].velocity[0][0,0,0,0,:8].numpy())
            list_particules[j] = PSO.update_position(list_particules[j])
            #print('position after for particule %d is %.5f ' % (j,list_particules[j].position[0][0,0,0,0,5]))
            
            # move the particle and evaluate its fitness
            list_particules[j].fitness , PSO.index_in_epoch= \
                PSO.evaluate_fitness(list_particules[j].position)
           
            #print('partial derivate for j',j,list_particules[j].partial_derivative[0][0,0,0,0,:8])
            """
            for w in range(0,len(list_particules[j].partial_derivative)) : 
              list_particules[j].partial_derivative[w]=tf.where(
                tf.greater_equal(list_particules[j].partial_derivative[w],tf.constant(0,dtype=tf.float32))\
             ,tf.ones_like(list_particules[j].partial_derivative[w]),- tf.ones_like(list_particules[j].partial_derivative[w]))
            """ 
            #update Pbest
           
            bool = tf.less(list_particules[j].fitness,
                        list_particules[j].fitness_best_pos).numpy()
            if (bool):
             list_particules[j].fitness_best_pos =  list_particules[j].fitness
             for w in range(0,len(list_particules[j].position)):
              list_particules[j].best_pos[w].assign (list_particules[j].position[w])
                     
          #update Gbest 
          gbest , gbest_fitness=PSO.find_gbest(list_particules,gbest , gbest_fitness)
          
          PSO.w = 1 - abs(gbest_fitness)
          PSO.c1 = PSO.w * 2
          PSO.c2 = 2 - PSO.c1
          """if i % 10 ==0 : """ 
          #PSO.w = PSO.w / 100000
          PSO.c1 = PSO.c1 / 1000000
          PSO.c2 = PSO.c2  / 100000    
          if i % 100 == 0 :
            print('iteration %d in epoch %d the  Gbest solution is %5f ' \
                  %(i, epoch,gbest_fitness.numpy(),))
          if(gbest_fitness.numpy() < last_fitness) : 
             last_fitness = gbest_fitness.numpy()
             self.saveVariables(path = path ,variables = list_particules)
                
      #print(' Gbest solution %.5f ' \
        #      % (gbest_fitness.numpy()))                    
       

def launch_pso(retrive):
   
     psoimplemntation = PSOimplemntation(nb_iteration=2717,
                          swarm_size=20,cognitive=0.000018,social=0.000002,weight=0.9)
     psoimplemntation.lunch(retrive)

launch_pso(True)    


           


