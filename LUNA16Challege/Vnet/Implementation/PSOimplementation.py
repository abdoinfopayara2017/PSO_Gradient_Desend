import numpy as np
from PSOEngine import PSOEngine
import tensorflow as tf


class PSOimplemntation :

    
    def __init__(self,nb_iteration,swarm_size,cognitive,social,weight):
        self.nb_iteration=nb_iteration
        self.swarm_size=swarm_size
        self.cognitive=cognitive
        self.social=social
        self.weight=weight

     
    def lunch(self):
     
     PSO=PSOEngine(self.swarm_size,self.cognitive,self.social,self.weight)
     list_particules=[]    
     list_particules=PSO.init_particles(list_particules)
     init = tf.global_variables_initializer()
     with tf.Session() as sess:
      sess.run(init)
      # initialisation des parametres 
      for p in range(0,len(list_particules)) :
        for j in range(0,len(list_particules[p].position)) :
          sess.run(list_particules[p].position[j])
          # Evaluate the particles positions;
          list_particules[p].fitness , list_particules[p].dot_derivate ,\
          list_particules[p].pre_activations , list_particules[p].activations \
          = PSO.evaluate_fitness(list_particules[p].position)
          # for each particle i do Pbesti = xi;
          list_particules[p].fitness_best_pos =  list_particules[p].fitness
          list_particules[p].best_pos =  list_particules[p].position.copy()
          
      # Find best particle in set
      gbest , gbest_fitness=PSO.find_gbest(list_particules)
           
      # PSO boucle
      for i in range(0,self.nb_iteration):
       # Find the best for each particle
       for j in range(0,len(list_particules)):
        list_particules[j].fitness , list_particules[j].dot_derivate ,\
        list_particules[j].pre_activations , list_particules[j].activations \
        = PSO.evaluate_fitness(list_particules[j].position)
        accuracy=-sess.run(list_particules[j].fitness)
        print('accuracy swarm number %d is %f' % (j,accuracy))
        sess.run(list_particules[j].dot_derivate)        
        for w in range(0,len(list_particules[j].pre_activations)): 
         sess.run(list_particules[j].pre_activations[w])         
        for w in range(0,len(list_particules[j].activations)): 
         sess.run(list_particules[j].activations[w])      
      
        #best_pos = PSO.evaluate_fitness(list_particules[j].best_pos)[0]       
        #sess.run(best_pos)
        #if(list_particules[j].fitness.eval() < best_pos.eval() ) :
        # list_particules[j].best_pos=list_particules[j].position.copy()
        
       # Find best particle in set
       gbest , gbest_fitness=PSO.find_gbest(list_particules)
       print('the best solution : ', gbest_fitness )
       # Initialize the random vectors for updates
       r1=np.empty(len(list_particules[0].position),dtype=object) 
       r2=np.empty(len(list_particules[0].position),dtype=object) 
       for j in range(0,len(list_particules[0].position)):
        r1[j]=np.random.rand(*list_particules[0].position[j].shape)
        r2[j]=np.random.rand(*list_particules[0].position[j].shape)
        
       # Update the velocity and position vectors
       for j in range(0,len(list_particules)):
        
        list_particules[j] = PSO.update_velocity(list_particules[j],gbest,r1,r2)
        for w in range(0,len(list_particules[j].velocity)) :
         sess.run(list_particules[j].velocity[w])
        list_particules[j] = PSO.update_partial_derivatives(list_particules[j])
        for w in range(0,len(list_particules[j].partial_derivative)) :
         if w == len(list_particules[j].partial_derivative) - 3 :
          print(sess.run(list_particules[j].partial_derivative[w]))
        #for w in range(0,len(list_particules[j])) :        
         #PSO.updatePosition(list_particules[j])
        
        #print("loss :",gbest.fitness)'''    
       

def launch_pso():
   
     psoimplemntation = PSOimplemntation(nb_iteration=1,
                          swarm_size=1,cognitive=2,social=2,weight=0.9)
     psoimplemntation.lunch()

launch_pso()    


           


