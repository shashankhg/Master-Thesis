
"""
Master Thesis
Project: Energy Efficient and Low Cost Localization in Wireless Sensor Networks
@author: Shashank Harigopal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
import time
from matplotlib.ticker import (MultipleLocator)

start_time = time.time()

# Bell shaped kernel
def kernel(point, xmat, k):
    m,n = np.shape(xmat)
    weights = np.mat(np.eye((m)))
    
    for j in range(m):
        diff = point - X[j]
        weights[j, j] = np.exp(diff * diff.T / (-2.0 * k**2))
    
    return weights

#Assign weights to the points using the Kernel Function
def localWeight(point, xmat, ymat, k):
    wt = kernel(point, xmat, k)
    W = (X.T * (wt*X)).I * (X.T * wt * ymat.T)
    return W

#Local Weighted Regression 
def localWeightRegression(xmat, ymat, k):
    m,n = np.shape(xmat)
    ypred = np.zeros(m)
    
    for i in range(m):
        ypred[i] = xmat[i] * localWeight(xmat[i], xmat, ymat, k)
        
    return ypred

def process(rssi, time, k):
    mtime = np.mat(time)
    mrssi = np.mat(rssi)
    m = np.shape(mrssi)[1]
    one = np.ones((1, m), dtype = int)
    global X
     # horizontal stacking
    X = np.hstack((one.T, mtime.T))
    rssi_smooth = localWeightRegression(X, mrssi, k)
    return rssi_smooth

# Function to calculate RSSI for all Nodes
def calcrssi(X, Y):
    
   d = [sqrt(X[i]**2 + Y[i]**2) for i in range(len(X))]
   ld = [np.log10(k) for k in d]
   rss = [(-20*ld[i])-45 for i in range(len(X))]
   return rss

def plotrssi(rssi, rssis, time):
    plt.plot(time, rssi, color='red', linewidth=2)
    plt.plot(time, rssis, color='green', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('RSSI of Node ')
    plt.show()
    return 

def calcposition(rssi1, rssi2, rssi3):
    r1 = [(10**((-rssi1[i]-45)/20)) for i in range(len(rssi1))]
    r2 = [(10**((-rssi2[i]-45)/20)) for i in range(len(rssi2))]
    r3 = [(10**((-rssi3[i]-45)/20)) for i in range(len(rssi3))]
    x1 = 1
    y1 = 1
    x2 = 1
    y2 = 10
    x3 = 10
    y3 = 1
    #Trilateration Formula
    a = -2*x1 + 2*x2
    b = -2*y1 + 2*y2
    c = [(r1[i]**2) - (r2[i]**2) - (x1**2) + (x2**2) - (y1**2) + (y2**2) for i in range(len(rssi1))]
    d = -2*x2 + 2*x3
    e = -2*y2 + 2*y3
    f = [(r2[i]**2) - (r3[i]**2) - (x2**2) + (x3**2) - (y2**2) + (y3**2) for i in range(len(rssi1))]
    xpos = [((c[i]*e-f[i]*b)/(e*a-b*d)) for i in range(len(rssi1))]
    ypos = [((c[i]*d-a*f[i])/(b*d-a*e)) for i in range(len(rssi1))]
    return xpos, ypos

def calcaccuracy(posX, posY):
    #Calculating error and accuracy for X Position
    errorX = abs(posX - posXa_test)
    meanex = round(np.mean(errorX), 2)
    print('Mean Absolute Error X:', meanex, 'meters')
    mapex = 100 * (errorX / posXa_test)
    # Calculate and display accuracy
    accuracyx = 100 - np.mean(mapex)
    print('Accuracy for X Position:', round(accuracyx, 2), '%')
    
    #Calculating error and accuracy for Y Position
    errorY = abs(posY - posYa_test)
    meaney = round(np.mean(errorY), 2)
    print('Mean Absolute Error Y:', meaney, 'meters')
    mapey = 100 * (errorY / posYa_test)
    # Calculate and display accuracy
    accuracyy = 100 - np.mean(mapey)
    print('Accuracy for Y Position:', round(accuracyy, 2), '%')
    print('Mean Absolute Error:', round((meanex+meaney)/2, 2), 'meters')
    accuracy = (accuracyx + accuracyy)/2
    return accuracy


data_test = pd.read_csv('Test_Data_Scenario10x10_Path_Outdoor.csv')
print(data_test.info())
rssin1_test = np.array(data_test.Node_1e)
rssin2_test = np.array(data_test.Node_2e)
rssin3_test = np.array(data_test.Node_3e)
rssin4_test = np.array(data_test.Node_4e)
posXa_test = np.array(data_test['PositionX_actual'])
posYa_test = np.array(data_test['PositionY_actual'])
ti = np.array(data_test.Time_Instant)

rssin1_smooth = process(rssin1_test, ti, 5)
with open('rssin1_smooth.txt', 'w') as f:
    for item in rssin1_smooth:
        f.write("%s\n" % item)
rssin2_smooth = process(rssin2_test, ti, 5)
with open('rssin2_smooth.txt', 'w') as f:
    for item in rssin2_smooth:
        f.write("%s\n" % item)
rssin3_smooth = process(rssin3_test, ti, 5)
with open('rssin3_smooth.txt', 'w') as f:
    for item in rssin3_smooth:
        f.write("%s\n" % item)
rssin4_smooth = process(rssin4_test, ti, 5)
with open('rssin4_smooth.txt', 'w') as f:
    for item in rssin4_smooth:
        f.write("%s\n" % item)

plotrssi(rssin1_test, rssin1_smooth, ti)
plotrssi(rssin2_test, rssin2_smooth, ti)
plotrssi(rssin3_test, rssin3_smooth, ti)
plotrssi(rssin4_test, rssin4_smooth, ti)
posXe_test, posYe_test = calcposition(rssin1_test, rssin2_test, rssin4_test)
posXs, posYs = calcposition(rssin1_smooth, rssin2_smooth, rssin4_smooth)
with open('posXs.txt', 'w') as f:
    for item in posXs:
        f.write("%s\n" % item)
        
with open('posYs.txt', 'w') as f:
    for item in posYs:
        f.write("%s\n" % item)

staticXs = round(np.mean(posXs), 2)
staticYs = round(np.mean(posYs), 2)
        
fig, ax = plt.subplots(figsize=(10, 10))
#ax.set_xlim(1, 10)
#ax.set_ylim(1, 10)
#ax.xaxis.set_major_locator(MultipleLocator(1))
#ax.yaxis.set_major_locator(MultipleLocator(1))
ax.grid(which='major', color='b', linestyle='--')
#plt.plot(posXa_test, posYa_test, color='green', linewidth=5)
#plt.plot(posXs, posYs, color='red', linewidth=5)
plt.scatter(posXa_test, posYa_test, color='green', linewidth=5)
plt.scatter(staticXs, staticYs, color='red', linewidth=5)
plt.xlabel('X Position', fontsize=18); plt.ylabel('Y Position', fontsize=18); plt.title('Actual Experiment Position vs Smoothened RSSI Reading Position', fontsize=20);
plt.show()

data_train = pd.read_csv('Training_Data_Scenario10x10.csv')
posXa_train = np.array(data_train['PositionX_actual'])
posYa_train = np.array(data_train['PositionY_actual'])

rssin1_train = calcrssi(posXa_train, posYa_train)
with open('rssin1_train.txt', 'w') as f:
    for item in rssin1_train:
        f.write("%s\n" % item)
        
rssin2_train = calcrssi(posXa_train, 10-posYa_train)
with open('rssin2_train.txt', 'w') as f:
    for item in rssin2_train:
        f.write("%s\n" % item)
        
rssin3_train = calcrssi(10-posXa_train, 10-posYa_train)
with open('rssin3_train.txt', 'w') as f:
    for item in rssin3_train:
        f.write("%s\n" % item)
        
rssin4_train = calcrssi(10-posXa_train, posYa_train)
with open('rssin4_train.txt', 'w') as f:
    for item in rssin4_train:
        f.write("%s\n" % item)

posXc_train, posYc_train = calcposition(rssin1_train, rssin2_train, rssin4_train)
with open('posXc_train.txt', 'w') as f:
    for item in posXc_train:
        f.write("%s\n" % item)

with open('posYc_train.txt', 'w') as f:
    for item in posYc_train:
        f.write("%s\n" % item)
        
#fig, ax = plt.subplots(figsize=(10, 10))
#ax.set_xlim(1, 10)
#ax.set_ylim(1, 10)
#ax.xaxis.set_major_locator(MultipleLocator(1))
#ax.yaxis.set_major_locator(MultipleLocator(1))
#ax.grid(which='major', color='b', linestyle='--')
#plt.scatter(posXc_train, posYc_train, color='yellow', linewidth=2)
#plt.scatter(posXa_train, posYa_train, color='green', linewidth=2)


"""Random Forest Regression Begins"""

features_train = pd.read_csv('Training_Data_Scenario10x10.csv')
features_train.head(6)
features_train= features_train.drop('PositionX_calc', axis = 1)
features_train= features_train.drop('PositionY_calc', axis = 1)
features_train= features_train.drop('PositionX_actual', axis = 1)
features_train= features_train.drop('PositionY_actual', axis = 1)
features_train= features_train.drop('Time_Instant', axis = 1)
print('The shape of the training features is:', features_train.shape)
features_train_list = list(features_train.columns)
features_train = np.array(features_train)

features_test = pd.read_csv('Test_Data_Scenario10x10_Path_Outdoor.csv')
features_test.head(6)
posXf_test = np.array(features_test['PositionX_fil'])
posYf_test = np.array(features_test['PositionY_fil'])
posXa_test = np.array(features_test['PositionX_actual'])
posYa_test = np.array(features_test['PositionY_actual'])
features_test= features_test.drop('PositionX_fil', axis = 1)
features_test= features_test.drop('PositionY_fil', axis = 1)
features_test= features_test.drop('Node_1e', axis = 1)
features_test= features_test.drop('Node_2e', axis = 1)
features_test= features_test.drop('Node_3e', axis = 1)
features_test= features_test.drop('Node_4e', axis = 1)
features_test= features_test.drop('PositionX_actual', axis = 1)
features_test= features_test.drop('PositionY_actual', axis = 1)
features_test= features_test.drop('Time_Instant', axis = 1)
print('The shape of the testing features is:', features_test.shape)
features_test_list = list(features_test.columns)
features_test = np.array(features_test)

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0.5, 10.5)
ax.set_ylim(0.5, 10.5)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.grid(which='major', color='b', linestyle='--')
#plt.plot(posXa_test , posYa_test , color='green', linewidth=5)
plt.scatter(posXa_test, posYa_test, color='green', linewidth=5)
plt.xlabel('X Position', fontsize=18); plt.ylabel('Y Position', fontsize=18); plt.title('Actual Position in Experiment', fontsize=20);
plt.show()

# Baseline predictions based on Experimental RSSI values
exp_accuracy = calcaccuracy(posXe_test, posYe_test)
print('Average accuracy from the experiment: ', round(exp_accuracy, 2), '%')

# Baseline predictions based on Filtered RSSI values
smooth_accuracy = calcaccuracy(posXf_test, posYf_test)
print('Average accuracy after filtering RSSI: ', round(smooth_accuracy, 2), '%')

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data for X Position
trainingtimeX = time.time()
rf.fit(features_train, posXa_train);
print("X Position Training time: %s seconds" % (time.time() - trainingtimeX))

# Use the forest's predict X Position on the test data
predictiontimeX = time.time()
predictionX = rf.predict(features_test)
print("X Position Prediction time: %s seconds" % (time.time() - predictiontimeX))

# Train the model on training data for Y Position
trainingtimeY = time.time()
rf.fit(features_train, posYa_train);
print("Y Position Training time: %s seconds" % (time.time() - trainingtimeY))

# Use the forest's predict Y Position on the test data
predictiontimeY = time.time()
predictionY = rf.predict(features_test)
print("Y Position Prediction time: %s seconds" % (time.time() - predictiontimeY))

pred_accuracy = calcaccuracy(predictionX, predictionY)
print('Random Forest prediction accuracy: ', round(pred_accuracy, 2), '%')

staticpredX = round(np.mean(predictionX), 2)
staticpredY = round(np.mean(predictionY), 2)

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0.5, 10.5)
ax.set_ylim(0.5, 10.5)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.grid(which='major', color='b', linestyle='--')
#plt.plot(posXa_test , posYa_test , color='green', linewidth=5)
#plt.plot(predictionX, predictionY, color='orange', linewidth=5)
plt.scatter(posXa_test, posYa_test, color='green', linewidth=5)
plt.scatter(staticpredX, staticpredY, color='orange', linewidth=5)
plt.grid(color='b', linestyle='--', linewidth=1)
plt.xlabel('X Position', fontsize=18); plt.ylabel('Y Position', fontsize=18); plt.title('Predicted Position using Random Forest Regression', fontsize=20);
#plt.savefig('Scenario10x10_Outdoor_Path.png')
plt.show()

Xposition = process(predictionX, ti, 1)
Yposition = process(predictionY, ti, 1)

staticpredXs = round(np.mean(Xposition), 2)
staticpredYs = round(np.mean(Yposition), 2)

final_accuracy = calcaccuracy(Xposition, Yposition)
print('Final smoothened accuracy: ', round(final_accuracy, 2), '%')

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0.5, 10.5)
ax.set_ylim(0.5, 10.5)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.grid(which='major', color='b', linestyle='--')

#plt.plot(posXa_test , posYa_test , color='green', linewidth=5)
#plt.plot(Xposition, Yposition, color='blue', linewidth=5)
#plt.plot(predictionX, predictionY, color='yellow', linewidth=5)
plt.scatter(posXa_test, posYa_test, color='green', linewidth=5)
plt.scatter(staticpredX, staticpredY, color='orange', linewidth=5)
plt.scatter(staticpredXs, staticpredYs, color='blue', linewidth=5)
plt.grid(color='b', linestyle='--', linewidth=1)
plt.xlabel('X Position', fontsize=18); plt.ylabel('Y Position', fontsize=18); plt.title('Predicted Position filtered using LOESS', fontsize=20);
#plt.savefig('Scenario10x10_Outdoor_Path.png')
plt.show()

print("Total Runtime: %s seconds" % (time.time() - start_time))