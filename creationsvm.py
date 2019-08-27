import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

class svmachine:
    def _init_(self, visualization=True):
        self.visualization = visualization
        self.color = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
   #train
    def fit(self, data):
        self.data = data
        #{ ||W||:[w,b]}
        opt_dict = {}
        transform = [[1,1],
                     [-1,1],
                     [1,-1],
                     [-1,-1]]
        all_data = []
        for y in self.data:
            for featureset in self.data[y]:
                for feature in featureset:

                     all_data.append(feature)

        self.max_feature = max(all_data)
        self.min_feature = min(all_data)
        all_data = None

        step_sizes = [ self.max_feature *0.1,
                      self.max_feature *0.01,
                      self.max_feature *0.001,]
     #extremly expensive
        b_range = 5
        b_multiple = 5

        latest_optimun =  self.max_feature*10

        for step in step_sizes:
           w = np.array([latest_optimun,latest_optimun])
           #we can do this bcz convex
           optimized = False
           while not optimized:
               for b in np.arange(-1*( self.max_feature*b_range),
                                  self.max_feature*b_range,
                                  step* b_multiple):
                   for transformation in transform:
                       w_t = w*transformation
                       found = True
#weakest link in svm smo fix this
                       for i in self.data:
                           for y in self.data[i]:
                               yi=i;
                               if not yi*(np.dot(w_t,y)+b) >= 1:
                                   found = False
                                    
                       if found:
                           opt_dict[np.linalg.norm(w_t)] = [w_t,b]
               if w[0] < 0:
                    optimized = True
                    print('optimized a step')
               else:
                     w = w - step

           norms = sorted([n for n in opt_dict])
           opt_choice = opt_dict[norms[0]]
           self.w = opt_choice[0]
           self.b = opt_choice[1]
           latest_optimum = opt_choice[0][0]+step*2
                            
                           
                                  
                   
                                  
           
        
    def predict(self, features):
        #sign(x.w+b)
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification !=0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors(classification))
        return classification

    def visual(self):
            [(self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]) for i in data_dict]
            def hyperplane(x,w,b,v):
                return (-w[0]*x-b+v) / w[1]
            datarange = (self.min_feature*0.9,self.max_feature*1.1)
            xmin = datarange[0]
            xmax = datarange[1]

            #(w.x+b)= 1
            psv1 = hyperplane(xmin, self.w, self.b, 1)
            psv2 = hyperplane(xmax, self.w, self.b, 1)
            self.ax.plot([xmin,xmax],[psv1,psv2])
        
            #(w.x+b)= -1
            nsv1 = hyperplane(xmin, self.w, self.b, 1)
            nsv2 = hyperplane(xmax, self.w, self.b, 1)
            self.ax.plot([xmin,xmax],[nsv1,nsv2])

            #(w.x+b)= 0
            tsv1 = hyperplane(xmin, self.w, self.b, 1)
            tsv2 = hyperplane(xmax, self.w, self.b, 1)
            self.ax.plot([xmin,xmax],[tsv1,tsv2])

            plt.show()

data_dict = {-1:np.array([[1,7],
                          [2,8],
                         [ 3,8],]),


             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}

svm = svmachine()
svm.fit(data=data_dict)
svm.visualize()
