#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 18:36:57 2018

@author: gevans
"""
import os
import numpy as np
import pandas as pd
import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
#Importing local libraries
root = '/Users/gevans/Documents/Python/Coursera/ex4/'
os.chdir(root)

#class classifier(object):
#    def __init__(self):
#        pass

class handwriting_images(object):
    def __init__(self,Xshort,y):
        self.Xshort = Xshort
        handwriting_images.add_bias_unit_to_input(self)
        self.y = y
        
    def display_multiple_images(self,width):
        pad = 1
        Xshuffled = self.X
        Xshuffled = self.X[np.random.rand(self.X.shape[0]).argsort(),:]
        
        examples,allpixels = Xshuffled.shape
        pixel_width = int(np.sqrt(allpixels-1))
        all_images = Xshuffled[:,:allpixels-1].reshape(examples,pixel_width,pixel_width).transpose([0,2,1])
        multiplot = np.zeros(((pad+pixel_width)*width+pad,(pad+pixel_width)*width+pad))
        for i in range(width):
            for j in range(width):
    #            i=0;j=1;width=10
                startx = j*pixel_width+j*pad+1
                starty = i*pixel_width+i*pad+1
                multiplot[startx:startx+pixel_width,starty:starty+pixel_width] = all_images[width*i+j,:,:]
        plt.imshow(multiplot);plt.axis('off');plt.title('Visualising the handwriting training data');plt.show()
    
    def add_bias_unit_to_input(self):
        self.X = np.append(np.ones((self.Xshort.shape[0],1)),self.Xshort,axis=1)  







class Neural_Net_2layers(object):
    def __init__(self,structure,reg):
        self.structure = structure
        self.reg = reg
        Neural_Net_2layers.randomly_initialise_theta(self)
    
    def set_epsilon(L_in,L_out):
        return np.sqrt(6)/np.sqrt(L_in+L_out)

    def randomly_initialise_theta(self):
        init_epsilon1 = Neural_Net_2layers.set_epsilon(self.structure[0],self.structure[1])
        init_epsilon2 = Neural_Net_2layers.set_epsilon(self.structure[1],self.structure[2])
        theta1_init = np.random.rand(self.structure[1],self.structure[0]+1)*(2*init_epsilon1)-init_epsilon1
        theta2_init = np.random.rand(self.structure[2],self.structure[1]+1)*(2*init_epsilon2)-init_epsilon2
        self.theta = np.r_[theta1_init.ravel(), theta2_init.ravel()]
        
    def sigmoid_func (input_numeric):
        return 1/(1+np.exp(-input_numeric))
    
    def sigmoidGrad_func(z):
        return(Neural_Net_2layers.sigmoid_func(z)*(1-Neural_Net_2layers.sigmoid_func(z)))
        
    def cost_func(self,training_data):
        #Set params
        m = training_data.X.shape[0]    
        input_layer_size = self.structure[0]
        hidden_layer_size = self.structure[1]
        labels_size = self.structure[2]
    
        #Unravel Theta values
        theta1 = self.theta[0:(hidden_layer_size*(input_layer_size+1))].reshape(hidden_layer_size,(input_layer_size+1))
        theta2 = self.theta[(hidden_layer_size*(input_layer_size+1)):].reshape(labels_size,(hidden_layer_size+1))
    
        #Forward prop
        z2 = theta1.dot(training_data.X.T)
        a2 = np.append(np.ones((z2.shape[1],1)),Neural_Net_2layers.sigmoid_func(z2.T),axis=1)
        z3 = theta2.dot(a2.T)
        a3 = Neural_Net_2layers.sigmoid_func(z3).T
    #    h = np.argmax(a3, axis=1)+1
    
        ywide = pd.get_dummies(training_data.y.ravel()).as_matrix()
    #    hwide = pd.get_dummies(h.ravel()).as_matrix()
        
        J = -1*(1/m)*np.sum((np.log(a3)*(ywide)+np.log(1-a3)*(1-ywide))) + \
            (reg/(2*m))*(np.sum(np.square(theta1[:,1:])) + np.sum(np.square(theta2[:,1:])))
            
        #Back prop
        d3 = (a3 - ywide).T
        d2 = (theta2[:,1:].T.dot(d3))*Neural_Net_2layers.sigmoidGrad_func(z2) # 25x10 *10x5000 * 25x5000 = 25x5000
    
        delta1 = d2.dot(training_data.X) # 25x5000 * 5000x401 = 25x401
        delta2 = d3.dot(a2) # 10x5000 *5000x26 = 10x26
        
        theta1_ = np.c_[np.ones((theta1.shape[0],1)),theta1[:,1:]]
        theta2_ = np.c_[np.ones((theta2.shape[0],1)),theta2[:,1:]]
        
        theta1_grad = delta1/m + (theta1_*reg)/m
        theta2_grad = delta2/m + (theta2_*reg)/m
        
        return(J, np.append(theta1_grad,theta2_grad))
        
    def minimise_theta(self,iterations,training_data):
        cost,grad = Neural_Net_2layers.cost_func(self,training_data)
        for i in range(iterations):
            multiplier = 1
            new_cost,grad = Neural_Net_2layers.cost_func(self,training_data)
            self.theta = self.theta - multiplier * grad

        self.cost = Neural_Net_2layers.cost_func(self,training_data)[0]
        
    def predict(self,training_data):
    
        m = training_data.X.shape[0]
        input_layer_size = self.structure[0]
        hidden_layer_size = self.structure[1]
        labels_size = self.structure[2]
        
        #Unravel Theta values
        theta1 = self.theta[0:(hidden_layer_size*(input_layer_size+1))].reshape(hidden_layer_size,(input_layer_size+1))
        theta2 = self.theta[(hidden_layer_size*(input_layer_size+1)):].reshape(labels_size,(hidden_layer_size+1))
        
        #Forward prop
        z2 = theta1.dot(training_data.X.T)
        a2 = np.append(np.ones((z2.shape[1],1)),Neural_Net_2layers.sigmoid_func(z2.T),axis=1)
        z3 = theta2.dot(a2.T)
        a3 = Neural_Net_2layers.sigmoid_func(z3).T    
        h = np.argmax(a3, axis=1)+1
        h.shape = training_data.y.shape
        percent = np.sum(h==training_data.y)*100/m
        return round(percent,4)


#%%

NN_structure = [400,25,10]
reg = 1
testing_net = Neural_Net_2layers(NN_structure,reg)

data = loadmat('ex4data1.mat')
handwriting_photos = handwriting_images(data['X'],data['y'])

handwriting_photos.display_multiple_images(10)
#%%
testing_net.randomly_initialise_theta()
testing_net.cost_func(handwriting_photos)[0]
testing_net.minimise_theta(10000,handwriting_photos)
testing_net.predict(handwriting_photos)   #99.26
#%%

example_reg_terms   = [0,0.01,0.03,0.1,0.3,1,3,10,30]
iterations          = 1000
structure           = [400,25,10]
output              = []

for i in example_reg_terms:
    looping_neural_network =  Neural_Net_2layers(structure,i)
    testing_net.randomly_initialise_theta()
    testing_net.minimise_theta(iterations,handwriting_photos)
    accuracy = testing_net.predict(handwriting_photos)   

    output.append(i)
    output.append(accuracy)


output = np.array(output).reshape(9,2)
plt.semilogx(output[:,0],output[:,1],'.b')
plt.ylabel('Accuracy on training set')
plt.xlabel('Regularisation term')
plt.title("Visualising the effect of various regularisation terms")


#%%















