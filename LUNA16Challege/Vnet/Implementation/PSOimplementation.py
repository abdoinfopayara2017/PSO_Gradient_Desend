from PSOEngine import PSOEngine

class PSOimplemntation :
    
    def __init__(self,numIteration):
        self.numIteration=numIteration

        
    def lunch(self):
     PSO=PSOEngine(10,3,1,0.9)
     listPrticules=[]
     PSO.initParticles(listPrticules)
     for i in range(0,self.numIteration):
        listPrticules[i].fitness=PSO.evaluateFitness(listPrticules[i].position_i)
        
        if(listPrticules[i].fitness < PSO.evaluateFitness(listPrticules[i].pos_best)) :
           listPrticules[i].pos_best=listPrticules[i].position


