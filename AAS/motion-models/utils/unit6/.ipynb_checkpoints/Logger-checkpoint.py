import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Logger():
    def __init__(self, n_steps, n_features):
        self.n_features = n_features
        self.log_error = np.empty((n_steps,n_features))
        self.log_det = np.empty((n_steps,n_features))
            
    def log(self, k, robot, Map):
        for idx in range(self.n_features):
            tid = robot.MappedFeatures[idx,0]
            if tid <= -1:
                self.log_error[k,idx]= np.Inf
                self.log_det[k,idx]=np.Inf
            else:
                self.log_det[k,idx] = np.linalg.det(robot.PEst[tid:tid+2,tid:tid+2])
                self.log_error[k,idx] = np.sqrt(np.sum((robot.xEst[tid:tid+2,0] - Map[:,idx])**2))
                
    def plot(self):
        fig1 , ax1 =plt.subplots(1,1,sharex=True)
        fig2 , ax2 =plt.subplots(1,1,sharex=True)
        fig2.tight_layout()
        fig1.tight_layout()

        df1 = pd.DataFrame(data= self.log_error, columns = ['Landmark {}'.format(i) for i in range(self.n_features)])
        #ax1.set_title('Error between map and est')
        df1.plot(ax = ax1)
        df2 = pd.DataFrame(data=np.log(self.log_det), columns=['Landmark {}'.format(i) for i in range(self.n_features)])
        #ax2.set_title('Det. of covar.')
        df2.plot(ax = ax2)
        #fig1.savefig('img/err-{}.png'.format(nFeatures))