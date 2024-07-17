from PSOEngine import PSOEngine
import numpy as np

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
     PSO.init_particles(list_particules)
     # PSO boucle
     for i in range(0,self.nb_iteration):
        
        # Find the best for each particle
        for j in range(0,len(list_particules)):
          list_particules[j].fitness , list_particules[j].dot_derivate , list_particules[j].pre_activations , 
          list_particules[j].activations = PSO.evaluate_fitness(list_particules[j].position)
        
          if(list_particules[j].fitness < PSO.evaluate_fitness(list_particules[j].best_pos)[0]) :
           list_particules[j].best_pos=list_particules[j].position.copy()
        
        # Find best particle in set
        gbest=PSO.find_gbest(list_particules)
        # Initialize the random vectors for updates
        r1=np.empty(len(list_particules[0].position),dtype=object) 
        r2=np.empty(len(list_particules[0].position),dtype=object) 
        for j in range(0,len(list_particules[0].position)):
            r1[j]=np.random.rand(list_particules[0].position[j].shape())
            r2[j]=np.random.rand(list_particules[0].position[j].shape())
        
        # Update the velocity and position vectors
        for j in range(0,len(list_particules)):
           PSO.update_velocity(list_particules[j],gbest,r1,r2)
           PSO.update_partial_derivatives(list_particules[j])
           PSO.updatePosition(list_particules[j])
        
        print("loss :",gbest.fitness)

def launch_pso():
   p=PSOimplemntation(5,2,2,2,0.9)    

launch_pso()    


           


