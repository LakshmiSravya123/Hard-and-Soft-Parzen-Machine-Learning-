import numpy as np
#import matplotlib.pyplot as plt
iris = np.loadtxt('http://www.iro.umontreal.ca/~dift3395/files/iris.txt')


def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    #print(np.random.choice(label_list))
    return np.random.choice(label_list)


#RBF function 
#kernel function (k-means)
 



class Q1:
    
    ### feature means ###########
    def feature_means(self, iris):
        iris_array=np.array(iris[:,:4],  dtype=np.float32)
        #print(np.around(np.mean(iris_array, axis=0),decimals=2))
        return (np.around((np.mean(iris_array, axis=0)),decimals=2))

    ########covariance matrix ###########
    def covariance_matrix(self, iris):
        iris_array=np.array(iris[:,:4],  dtype=np.float32)
        #print(np.around(np.cov(iris_array,rowvar=False),decimals=1))
        return (np.around(np.cov(iris_array,rowvar=False),decimals=1))
        
    def feature_means_class_1(self, iris):
        iris_array=[]
        for ix,iy in np.ndindex(iris.shape):
          if ((iris[ix,4])==1):
             iris_array.append(np.array(iris[ix,:4]))
             #iris_array[ix]= (np.array(iris[:,:4]))
        #print(np.around(np.mean(iris_array, axis=0),decimals=2))
        return (np.around((np.mean(iris_array, axis=0)),decimals=2))

    def covariance_matrix_class_1(self, iris):
        iris_array=[]
        for ix,iy in np.ndindex(iris.shape):
          if ((iris[ix,4])==1):
             iris_array.append(np.array(iris[ix,:4]))
        #print(iris_array)
        #print(np.around(np.cov(iris_array,rowvar=False),decimals=1))
        return (np.around(np.cov(iris_array,rowvar=False),decimals=1))

###hard parzeb
class HardParzen:
    def __init__(self, h):
        self.h = h
        
    ###training ##############3
    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs      
        self.train_labels= np.array(train_labels)
        self.label_listcount =len(np.unique(train_labels))
        self.label_list = (np.unique(train_labels))
        
    ###predictions ############3
    def compute_predictions(self, test_data):
        
        test_array_size = test_data.shape[0]
        #print(test_array)
        count = np.ones((test_array_size, self.label_listcount))    
        predict_classes = np.zeros(test_array_size)
        
        for(rowIndex, row) in enumerate(test_data):
        #distance finding
          #print(rowIndex)
          distance= (np.sum((np.abs(row - self.train_inputs)) ** 2, axis=1)) ** (1.0 / 2.0)
        #finding the neighnbours
         
          neighbour_array =[]
          h=self.h
          while len(neighbour_array) == 0:
             neighbour_array = np.array([j for j in range(len(distance)) if distance[j] < h ] )
             h*=2
             if len(neighbour_array) == 0:
               neighbour_array=np.array([ draw_rand_label(row, self.label_list).astype(int)])
            
          
         
          
          calulated_neighbours = list((self.train_labels.astype(int))[neighbour_array]-1)
        
          for j in range(min(len(calulated_neighbours),self.train_inputs.shape[0])):   
              count[rowIndex, calulated_neighbours[j]] += 1
              
          predict_classes[rowIndex] = (np.argmax(count[rowIndex, :]) + 1)
        #print(predict_classes)
        return predict_classes
                
           

#########Soft parzen #############3 

class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma
        

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels= train_labels
        self.label_listcount =len(np.unique(train_labels))
        self.label_list = (np.unique(train_labels))
    def compute_predictions(self, test_data):
        test_array_size = test_data.shape[0]
        count = np.zeros((test_array_size, self.label_listcount))
        
        predict_classes = np.zeros(test_array_size)
        
  
        for(rowIndex, row) in enumerate(test_data):
            
              sig=np.square(self.sigma)
          
              dis=np.sum((row - self.train_inputs) ** 2, axis=1)
             
              
              distance= np.exp(-dis/(2*sig))
         
                
              for j in range(len(self.train_inputs)):
                
                count[rowIndex, self.train_labels.astype(int)[j]-1 ] += distance[j]
               
              predict_classes[rowIndex] = np.argmax(count[rowIndex, :]) + 1
        
       
        return predict_classes
    

def split_dataset(iris):
        train =[]
        validation =[]
        test =[]
        for(rowIndex, row) in enumerate(iris):
       
            if(rowIndex%5 == 0 or rowIndex%5 == 1 or rowIndex%5 == 2):
#                if (training_set == []):
#                    training_set = row
#                else:
                     train.append(np.array(iris[rowIndex,:5]))
            elif(rowIndex%5 == 3):
#                if (validation_set  == []):
#                    validation_set = row
#                else:
                  validation.append(np.array(iris[rowIndex,:5]))
            elif(rowIndex%5 == 4):
#                if (test_set == []):
#                    test_set = row
#                else:
                  test.append(np.array(iris[rowIndex,:5]))
       
        
        return (np.array(train),np.array(validation),np.array(test))

####Error Rate############3
class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        
       hp = HardParzen(h)
       
       hp.train(self.x_train, self.y_train)
       
       classes_pred = hp.compute_predictions(self.x_val)
       #print(classes_pred)
       nclass = int(max(self.y_val))
       new_matrix = np.zeros((nclass, nclass))

       for (test, pred) in zip(self.y_val, classes_pred):
           new_matrix[int(test - 1), int(pred - 1)] += 1

       
       prediction_sum = np.sum(new_matrix)
       correct_sum = np.sum(np.diag(new_matrix))
     
       
       return np.round((1.0 - (float(correct_sum) / float(prediction_sum))),5)


        
    def soft_parzen(self, sigma):
       sp = SoftRBFParzen(sigma)
       
       sp.train(self.x_train, self.y_train)

       classes_pred = sp.compute_predictions(self.x_val)
       #print(classes_pred)
       nclass = int(max(self.y_val))
       new_matrix = np.zeros((nclass, nclass))

       for (test, pred) in zip(self.y_val, classes_pred):
           new_matrix[int(test - 1), int(pred - 1)] += 1

       #print(new_matrix)
       sum_preds = np.sum(new_matrix)
       sum_correct = np.sum(np.diag(new_matrix))
      
       return np.round((1.0 - (float(sum_correct) / float(sum_preds))),5)
   
#########get errors###############333
def get_test_errors(iris):
    data=split_dataset(iris)
    train_set=(data[0])
    x_train=np.array(train_set[:,:4])
    y_train= np.array(train_set[:,4])
    validation_set = data[1]
    x_val= np.array(validation_set[:,:4])
    y_val= np.array(validation_set[:,4])
    hp=ErrorRate(x_train,y_train,x_val,y_val)
    sp=ErrorRate(x_train,y_train,x_val,y_val)
    hp.hard_parzen(1.0)
    sp.soft_parzen(0.3)
    test_set=data[2]
    x_test= np.array(test_set[:,:4])
    y_test= np.array(test_set[:,4])
    hp1=ErrorRate(x_train,y_train,x_test,y_test)
    sp1=ErrorRate(x_train,y_train,x_test,y_test)
    sp4 = sp1.soft_parzen(0.3)
    hp5 = hp1.hard_parzen(1.0)

    return (hp5,sp4)

#############random projections################33
def random_projections(X, A):
    return np.transpose(np.multiply(1.0/np.sqrt(2.0),np.dot(np.transpose(A),np.transpose(X))))
     
    
     

###plotting
#plt.plot(range(1, 100), [get_test_error(k) for k in range(1, 100)], label='test error')
