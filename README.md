# Camera-Calibration
----File structure----
In the main folder, the code is written in main.py
Tsai calibration code is written in calibrator.py
/input images- source images adn their edge detected images.
/train and test/data1.txt- (X,Y,Z,x,y) tuples of the first image.
			   /data2.txt- (X,Y,Z,x,y) tuples of the second image.
			   /test.txt- (x1,y1,x2,y2) tuples which are used as test cases.

----Functions----
-> /main/main()- has code for input interactions, calling necessary functions and printing the final output.
-> /main/find_p()- has code for estimating final 3D coordinate matrix.
-> /calibrator/calibrate()- has code for estimating calibration matrix and projection matrix.

----Note----
-> Console asks the user to enter the path to training set of image1, image2, test set and the line number in which the desired test case is.
-> In the end, it prints the 2D coordinates and repsective 3D locations.

----Packages Used----
numpy - for basic matrix manipulations.

----Sources----
-> Calibrator used is taken from an open source Github. Contributor - Alex
	https://github.com/alexprz/tsai-camera-calibration
-> The edges are detected using an online edge detection tool, Pinetools
	https://pinetools.com/image-edge-detection
