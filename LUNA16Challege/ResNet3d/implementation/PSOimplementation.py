import numpy as np


from PSOEngine import PSOEngine
import tensorflow as tf
import pickle
import os
import cv2
from layer import (dense_to_one_hot)
from pathlib import Path
import pandas as pd
import Resnet3d as resNet3d 


#sys.path.insert(0, 'D:/FELIOUNE/PSO_GD/PSO_Gradient_Desend/LUNA16Challege/Vnet')
#sys.path.insert(0, 'E:/LUNA 16/PSOGD v1/PSO_Gradient_Desend/LUNA16Challege/Vnet')



#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
 

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
     
    def lunch(self,retrive,imagedata,labeldata,batch_size):     

     PSO=PSOEngine(self.swarm_size,self.cognitive,self.social,self.weight,0,batch_size)
     list_particules=[]
     path = "log\\classification\\" + "model\\" 
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
        imagedata_batch , maskdata_batch , PSO.index_in_epoch = \
             PSO._next_batch(imagedata , labeldata , PSO.batch_size ,PSO.index_in_epoch )
        # initialisation des parametres 
        for p in range(0,len(list_particules)) :      
          list_particules[p].fitness , list_particules[p].partial_derivative , list_particules[p].lost = \
           PSO.evaluate_fitness(list_particules[p].position,imagedata_batch,maskdata_batch)
        
          with tf.device('/gpu:0'):
            for w in range(0,len(list_particules[p].partial_derivative)) :         
              list_particules[p].partial_derivative[w]=tf.where(
              tf.greater_equal(list_particules[p].partial_derivative[w],tf.constant(0,dtype=tf.float32))\
              ,tf.ones_like(list_particules[p].partial_derivative[w]),- tf.ones_like(list_particules[p].partial_derivative[w]))
                
          # for each particle i do Pbesti = xi;
          list_particules[p].fitness_best_pos.assign( list_particules[p].fitness)     
          for w in range(0,len(list_particules[p].position)):
           list_particules[p].best_pos[w].assign (list_particules[p].position[w])
          
     # Find best particle in set
     gbest , gbest_fitness=PSO.find_gbest(list_particules,gbest,gbest_fitness)
     
     last_fitness = gbest_fitness.numpy()

     #print('position after for particule is %.5f ',  (list_particules[0].position[0][0,0,0,0,:8].numpy()))
            
            
     # PSO boucle
     # for each iteration do
     with tf.device('/gpu:0'):
      #for epoch in range(0,20) : 
        for i in range(0,self.nb_iteration) :
          imagedata_batch , maskdata_batch , PSO.index_in_epoch = \
             PSO._next_batch( imagedata , labeldata , PSO.batch_size ,PSO.index_in_epoch  )
          
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
            list_particules[j].fitness , list_particules[j].partial_derivative ,list_particules[j].lost = \
                PSO.evaluate_fitness(list_particules[j].position,imagedata_batch , maskdata_batch)
           
            #print('partial derivate for j',j,list_particules[j].partial_derivative[0][0,0,0,0,:8])
            
            for w in range(0,len(list_particules[j].partial_derivative)) : 
              list_particules[j].partial_derivative[w]=tf.where(
                tf.greater_equal(list_particules[j].partial_derivative[w],tf.constant(0,dtype=tf.float32))\
             ,tf.ones_like(list_particules[j].partial_derivative[w]),- tf.ones_like(list_particules[j].partial_derivative[w]))
             
            #update Pbest
           
            bool = tf.less(list_particules[j].fitness,
                        list_particules[j].fitness_best_pos).numpy()
            if (bool):
             list_particules[j].fitness_best_pos =  list_particules[j].fitness
             for w in range(0,len(list_particules[j].position)):
              list_particules[j].best_pos[w].assign (list_particules[j].position[w])
                     
          #update Gbest 
          gbest , gbest_fitness =PSO.find_gbest(list_particules,gbest , gbest_fitness)
          best_lost , best_acc = PSO.find_best(list_particules)
          with open('myLog.txt', 'a') as f:
               print('best of iteration %d is %5f with acc of %5f' \
                     %(i,best_lost,best_acc), file=f)
          
          PSO.w = 1 - abs(gbest_fitness)
          PSO.c1 = PSO.w * 2
          PSO.c2 = 2 - PSO.c1
          """if i % 10 ==0 : """ 
          #PSO.w = PSO.w / 100000
          PSO.c1 = PSO.c1 / 10000
          PSO.c2 = PSO.c2  / 1000    
          
          if(gbest_fitness.numpy() < last_fitness) : 
             last_fitness = gbest_fitness.numpy()
             self.saveVariables(path = path ,variables = list_particules)
             #with open('myLog.txt', 'a') as f:
              # print('iteration %d the  Gbest solution is %5f ' \
               #          %(i,gbest_fitness.numpy(),), file=f)
    
    def predict(self,batch_size):
        path_test = Path(__file__).parent / "..\..\dataprocess\data\\test.csv"
                
        with path_test.open() as file:
         # Read  data set (Train data from CSV file)
         csvimagedata = pd.read_csv(file,delimiter=',')
         data = csvimagedata.iloc[:, :].values
         # For Image
         images = data[:, 1:]
         # For Labels
         labels = data[:, 0]
         path = "log\\classification\\" + "model\\" 
         list_particules=self.retrieveVariables(path+"model.txt")
         gbest=np.empty(len(list_particules[0].position),dtype=object)
         for w in range(0,len(gbest)):            
          gbest[w]=tf.Variable(initial_value=tf.constant(0.0,shape=list_particules[0].position[w].get_shape()),
                    shape=list_particules[0].position[w].get_shape(),dtype=tf.float32)
         gbest_fitness=tf.Variable(0,dtype=tf.float32)
         PSO=PSOEngine(self.swarm_size,self.cognitive,self.social,self.weight,0,batch_size)
         gbest , gbest_fitness=PSO.find_gbest(list_particules,gbest , gbest_fitness)
         
         predictvalues = []
         predict_probs = []
         ResVGGnet3d = resNet3d.RestNet3dModule(48, 48, 48, channels=1, n_class=2)
         
         with tf.device('/gpu:0') :
          for num in range(np.shape(images)[0]):
            batchimage = np.reshape(np.load(images[num][0]), (1, 48, 48, 48, 1))
            predictvalue, predict_prob = ResVGGnet3d.prediction(batchimage,gbest)
            predictvalues.append(predictvalue)
            predict_probs.append(predict_prob)
          name = 'classify_metrics.csv'
          out = open(name, 'w')
          out.writelines("y_predict" + "," + "y_score" + "," + "y_true" + "\n")
          labels = labels.tolist()
          for index in range(np.shape(images)[0]):
           
           out.writelines(
            str(predictvalues[index][0]) + "," + str(predict_probs[index][0]) + "," + str(labels[index]) + "\n")
            
def predict_test():
       psoimplemntation = PSOimplemntation(nb_iteration=5435,
                          swarm_size=20,cognitive=0.00018,social=0.002,weight=0.9)
       psoimplemntation.predict(1)
def launch_pso(retrive):
     
     path_data = Path(__file__).parent / "..\..\dataprocess\data\\training.csv"
     with path_data.open() as file:
      # Read  data set (Train data from CSV file)
      csvimagedata = pd.read_csv(file,delimiter=',')
      data = csvimagedata.iloc[:, :].values        
      np.random.shuffle(data)
      # For Image
      images = data[:, 1:]
      # For Labels
      labels = data[:, 0]
     psoimplemntation = PSOimplemntation(nb_iteration=754 * 10,
                          swarm_size=20,cognitive=0.00018,social=0.002,weight=0.9)
     # label one_hot coding
     label_counts = np.unique(labels).shape[0]
     train_labels_onehot = dense_to_one_hot(labels, label_counts)
     train_labels_onehot = train_labels_onehot.astype(np.float)
     psoimplemntation.lunch(retrive,images,train_labels_onehot,32)
launch_pso(False)      
#predict_test()      


           


