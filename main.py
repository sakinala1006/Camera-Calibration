#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:01:03 2020

@author: saisrinijasakinala
"""

import numpy as np

import calibrator

def find_p(P_img1,K_img1,x_img1,P_img2,K_img2,x_img2):
    #method to estimate 3D world coordinates
    
    I = np.identity(3)
    
    #Projection matrix is product of all matrices => P = KRt
    #We compute Rt as we already know K => Rt = (K^-1)P for both the images.
    Rt_img1 = np.matmul(np.linalg.inv(K_img1),P_img1)

    Rt_img2 = np.matmul(np.linalg.inv(K_img2),P_img2)
    
    #The first three columns in the previous 3x4 matrix represent the rotation matrix R.
    #The last column represents the translation matrix t.
    x  = np.hsplit(Rt_img1,[3,3]) 
    
    y  = np.hsplit(Rt_img2,[3,3])
    
    #As hsplit returns a list of arrays, we assign the respective list items to arrays.
    R_img1 , t_img1 , R_img2 , t_img2 = x[0] , x[2] , y[0] , y[2]
    
    #Cj represents vector from origin of world coordinates to camera coordinates.
    #Cj is formulated as => Cj = -(Rj.T)tj
    c_img1 , c_img2 = (-1)*np.matmul(R_img1.T,t_img1) , (-1)*np.matmul(R_img2.T,t_img2)

    #Vj represents vector of pixels wrt world
    #Vj is formulated as => Vj = (Rj.T)(Kj^-1)Xj
    v_img1 , v_img2 =np.matmul(np.matmul(np.linalg.inv(R_img1),np.linalg.inv(K_img1)),x_img1), np.matmul(np.matmul(np.linalg.inv(R_img2),np.linalg.inv(K_img2)),x_img2)
    
    v_img1 , v_img2 =v_img1/np.linalg.norm(v_img1) , v_img2/np.linalg.norm(v_img2) #It is a normalized vector

    #p is the world coordinates matrix.
    #p is formulated as p = ((sum(I-Vj Vj.T))^-1)(sum(I-Vj Vj.T) Cj)
    #first represents the first part of the calculation.
    #second represents the second part of the calculation.
    first = np.linalg.inv((I-np.matmul(v_img1,v_img1.T)) + (I-np.matmul(v_img2,v_img2.T)))
    
    second = np.matmul(I-np.matmul(v_img1, v_img1.T),c_img1)+np.matmul(I-np.matmul(v_img2, v_img2.T),c_img2)
    
    p = 1000*np.matmul(first,second) 
    
    #returning world coordinates matrix
    return p

def main():
    
    #input training data for both the cameras
    file_1 = input("Training test for image 1:")
    
    file_2 = input("Training test for image 2:")
    
    #call calibrate function in Tsai calibrator.
    #This function takes the file name of the training set and returns 
    #Projection matrix, residual and Calibration matrix
    P_img1,r_img1,K_img1 = calibrator.calibrate(file_1)
    
    P_img2,r_img2,K_img2 = calibrator.calibrate(file_2)
    
    #input the file name consisting of test cases.
    f=open(input("Test cases:"))
    
    #input the line number of the test case you want to test.
    l = int(input("Which test case?:"))
    
    lines=f.readlines()
    
    x=lines[l-1].split(' ')
    
    #form two arrays with pixel lpcations in two images.
    x_img1 , x_img2 = np.array([[int(x[0])],[int(x[1])],[1]]) , np.array([[int(x[2])],[int(x[3])],[1]])
    
    #call the find_p method to get the matrix consisting of 3D world coordinates.
    p = find_p(P_img1,K_img1,x_img1,P_img2,K_img2,x_img2)
    
    #print the details.
    print('\n3D cordinates for \n')
    
    print('{}'.format(x_img1))
    
    print('\n and\n{}'.format(x_img2))
    
    print('is \n{}'.format(p))

if __name__=='__main__':
    
    main()