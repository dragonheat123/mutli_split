import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM

import matplotlib.pyplot as plt

np.random.seed(10)

data = pd.read_csv('E://AC2016//a0804.csv',header=0)

####one hot encoding for categorical variables

power_status = (data.ix[:,'power_status_LVr'])

power_status_1hot = np.array(pd.get_dummies(power_status))

####some editing to shape arrays for proper inputs

selectionx = data.ix[:,['set_point_LVr','indoortemp_LVr','TemperatureC']]
selectionx = np.array(selectionx) 
selectionx = np.concatenate((selectionx,power_status_1hot),axis=1)

selectiony = np.array(data.ix[:,['power_LVr']])

frequency_mins = 10
points = frequency_mins*2

power_10min = []
selectionx_10min = np.zeros([int(len(selectiony)/points),len(selectionx[0,:])])

for i in range(points,len(selectiony)+points,points):
    if np.isnan(np.mean(selectiony[i-points:i]))==True:
        power_10min.append(1)
    else:
        try:
            power_10min.append((np.mean(selectiony[i-points:i])))
        except ValueError:
            power_10min.append(1)

        

for j in range(0,len(selectionx[0,:])):
    holder = []
    for i in range(points,len(selectiony)+points,points):
        if np.isnan(np.mean(selectionx[i-points:i,j]))==True:
            holder.append(0)
        else:
            try:
                holder.append(np.mean(selectionx[i-points:i,j]))
            except ValueError:
                holder.append(0)
    selectionx_10min[:,j]=holder  

#######normalizing

def stand_mat(data):
    return (data-np.mean(data,axis=0))/np.std(data,axis=0)

def ret_power(data):
    return data*np.std(power_10min,axis=0)+np.mean(power_10min,axis=0)

def ret_selection(data):
    return data*np.std(selectionx_10min,axis=0)+np.mean(selectionx_10min,axis=0)
          
        
power_10min_std = stand_mat(power_10min)
selectionx_10min_std = stand_mat(selectionx_10min)
        

######breaking into timesteps
ts = 96
selectionx_10min_std_array = []
power_10min_std_array = []

for i in range(0,len(power_10min_std)-ts):
    selectionx_10min_std_array.append(selectionx_10min_std[i:i+ts,:])
    power_10min_std_array.append(power_10min_std[i+ts])                 #### at time T
   

selectionx_10min_std_array = np.array(selectionx_10min_std_array) 
power_10min_std_array = np.array(power_10min_std_array) 


xtrain, ytrain = selectionx_10min_std_array[0:40000,:,:], power_10min_std_array[0:40000]

xtest, ytest = selectionx_10min_std_array[40000:50000,:], power_10min_std_array[40000:50000]



#### building network

model = Sequential()
model.add(LSTM(50, input_shape=(96, 5),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

### 

model.fit(xtrain, ytrain,verbose=1,batch_size=1000,epochs=20)
predicted = model.predict(xtest)

plt.figure(figsize=(16,8))
plt.plot(ytest[1:10000], color='black', linewidth=0.5)
plt.plot(predicted[1:10000], color='blue', linewidth=0.5)
plt.show()

plt.figure(figsize=(16,8))
plt.plot(ret_power(ytest[2000:2200]), color='black', linewidth=0.5)
plt.plot(ret_power(predicted[2000:2200]), color='blue', linewidth=0.5)
plt.show()


real_pred=ret_power(predicted).astype(np.float64)
#real_pred[real_pred<100]=7
real_power= ret_power(ytest).reshape(10000,1)

c = np.abs(np.abs(real_pred-real_power)/real_power)
mape = np.mean(c)
rmse = np.sqrt(np.mean((real_pred-real_power)**2))
print(mape)
print(rmse)



