import numpy as np


from PSOEngine import PSOEngine
import tensorflow as tf
import pickle
import os
import cv2
import Vnet3d as vnet3d
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, 'D:/FELIOUNE/PSO_GD/PSO_Gradient_Desend/LUNA16Challege/Vnet')
#sys.path.insert(0, 'E:/LUNA 16/PSOGD v1/PSO_Gradient_Desend/LUNA16Challege/Vnet')

from layer import save_images

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
     
    def lunch(self,retrive,magedata,maskdata,batch_size):     

     PSO=PSOEngine(self.swarm_size,self.cognitive,self.social,self.weight,0)
     list_particules=[]
     path = "log\\segmentation\\" + "model\\" 
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
             PSO._next_batch( magedata , maskdata , batch_size , PSO.index_in_epoch )
        # initialisation des parametres 
        for p in range(0,len(list_particules)) :      
          list_particules[p].fitness , list_particules[p].partial_derivative = \
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
             PSO._next_batch( magedata , maskdata , batch_size , PSO.index_in_epoch )
          
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
            list_particules[j].fitness , list_particules[j].partial_derivative = \
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
          gbest , gbest_fitness=PSO.find_gbest(list_particules,gbest , gbest_fitness)
          
          with open('myLog.txt', 'a') as f:
               print('best of iteration %d is %5f' %(i,PSO.find_best(list_particules)), file=f)
          PSO.w = 1 - abs(gbest_fitness)
          PSO.c1 = PSO.w * 2
          PSO.c2 = 2 - PSO.c1
          """if i % 10 ==0 : """ 
          #PSO.w = PSO.w / 100000
          PSO.c1 = PSO.c1 / 10000
          PSO.c2 = PSO.c2  / 1000    
          
          if(gbest_fitness.numpy() < last_fitness) : 
             last_fitness = gbest_fitness.numpy()
             #self.saveVariables(path = path ,variables = list_particules)
             with open('myLog.txt', 'a') as f:
               print('iteration %d the  Gbest solution is %5f ' \
                         %(i,gbest_fitness.numpy(),), file=f)
    
    def predict(self):
        DSC = 0
        path_mask = Path(__file__).parent / "..\..\dataprocess\data\Segmentation3dMaskTest.csv"
        path_data = Path(__file__).parent / "..\..\dataprocess\data\Segmentation3dImageTest.csv"
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
        path = "log\\segmentation\\" + "model\\" 
        list_particules=self.retrieveVariables(path+"model.txt")
        gbest=np.empty(len(list_particules[0].position),dtype=object)
        for w in range(0,len(gbest)):            
          gbest[w]=tf.Variable(initial_value=tf.constant(0.0,shape=list_particules[0].position[w].get_shape()),
                    shape=list_particules[0].position[w].get_shape(),dtype=tf.float32)
        gbest_fitness=tf.Variable(0,dtype=tf.float32)
        PSO=PSOEngine(self.swarm_size,self.cognitive,self.social,self.weight,0)
        gbest , gbest_fitness=PSO.find_gbest(list_particules,gbest , gbest_fitness)
        
        with tf.device('/gpu:0') :
          for num in range(imagedata.shape[0]) :
           src_path = imagedata[num][0]
           mask_path = maskdata[num][0]
           imges = []
           masks = []
          
           for z in range(16):
             img = cv2.imread(src_path + "/" + str(z) + ".bmp", cv2.IMREAD_GRAYSCALE)
             mask = cv2.imread(mask_path + "/" + str(z) + ".bmp", cv2.IMREAD_GRAYSCALE)
             imges.append(img)
             masks.append(mask)
             #print(src_path + "/" + str(z) + ".bmp"+ " -- " +mask_path + "/" + str(z) + ".bmp")
           test_imges = np.array(imges)
           test_imges = np.reshape(test_imges, (16, 96, 96))

           test_masks = np.array(masks)
           test_masks = np.reshape(test_masks, (16, 96, 96))
           Vnet3d = vnet3d.Vnet3dModule(96, 96, 16,channels=1)        
           predict , train_loss = Vnet3d.prediction(test_imges,gbest,test_masks)
           test_images = np.multiply(test_imges, 1.0 / 255.0)
           test_masks = np.multiply(test_masks, 1.0 / 255.0)
           DSC = train_loss.numpy() #+ DSC
           if(-DSC < 0.50) :
            #print ('avrage of DSC %5f on iteration %d' %(-DSC/(num + 1),num))
            with open('results.txt', 'a') as f:
               print('for image %s DSC %5f' %(src_path,-DSC), file=f)         
          """path_test = path + "DSC %5f\\" %(-DSC)
          if not os.path.exists(path_test) :
             os.makedirs(path_test)
          save_images(test_images, [4, 4],path_test + "test_%d_src.bmp" %(num))        
          save_images(test_masks, [4, 4], path + "DSC %5f" %(-DSC) + "\\" + "test_%d_mask.bmp" %(num))
          save_images(predict, [4, 4], path + "DSC %5f" %(-DSC) + "\\" + "test_%d_predict.bmp" %(num))
          """
   
def predict_test():
       psoimplemntation = PSOimplemntation(nb_iteration=5435,
                          swarm_size=20,cognitive=0.00018,social=0.002,weight=0.9)
       psoimplemntation.predict()
def launch_pso(retrive):
     
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
     psoimplemntation = PSOimplemntation(nb_iteration=5435,
                          swarm_size=20,cognitive=0.00018,social=0.002,weight=0.9)
     psoimplemntation.lunch(retrive,imagedata,maskdata,3)
launch_pso(False)      
#predict_test()      


           


