

class PSOEngine :
    def __init__(self,numDimensions,numParticles,maxIterations,c1,c2,w):
        self.numDimensions=numDimensions
        self.numParticles=numParticles
        self.maxIterations=maxIterations
        self.c1=c1
        self.c2=c2
        self.w=w

    def initParticles(particles):
        """
            Method to initialize the particles for PSO
        """
        for i in range(0,len(particles)):
            

