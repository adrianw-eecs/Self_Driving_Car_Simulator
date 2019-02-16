# Self Driving Car Simulation
I was inspired to do this project by Siraj Raval from Youtube. 

## Overview

In this project I will be using a Neural Network to drive a car around a  track in a simulator. The simulator we're going to use Udacity's [self driving car simulator](https://github.com/udacity/self-driving-car-sim) as a testbed for training an autonomous car. Along with the simulator I was provided a pre-trained model, I will be attempting to outperform this model. 

## Usage
The models that I will be posting can be run by running the command python drive.py "modelname.h5"

## Model Versions
1.	**model-Pre-Trained** - Pre-trained model created
This model performs fairly well from the beginning, it stays on the track the entire time.
2.	**model-V1.00** - Model trained using 2 laps of test data and 2 epochs and learning rate of 1.0e-4
This model performed poorly, did mild amounts of steering but encountered issues with sharp turns and swayed within the lane. 
3.	**model-V1.01** - Model trained using 2 laps of test data and 10 epochs and learning rate of 1.0e-4
This model performed poorly, but was able to complete half a lap, did not expereince the swaying as much as the V1.00 model.
4.	**model-V1.02** - Model trained using 2 laps of test data and 10 epochs and learning rate of 2.0e-4
This model performed extremely well, the faster learning rate helped the algorithim tune its self much better and was able to perform on p ar with the Pre-trained model. I was fairly surprised that such a small change could affect the outcome of the model so much.
5.	**model-V1.03** - Model trained using 2 laps of test data and 10 epochs and learning rate of 1.0e-3
For this model I had the idea that since increasing the training rate in my last example improved the output model by a lot. I researched other training rates, many suggested to increment by a factor of 10. 
The results were quite surprising to me since the model performed extremenly bad. It could not turn properly, and appeard to have learned nothing. This is most likely to the fact that when using gradient decent if the learning rate is to high the steps it takes are to large and thus does not learn quickly
6.	**model-V1.04** - Model trained using 2 laps of test data and 10 epochs and learning rate of 5.0e-4
As expected the training rate was way to high and the model moved forward with a slight steer to the right. I found this interesting because most of the track consists of the model steering to the left. Similarly to model-V1.03, the model moved straight with a slight steer to the right. This could mean there are issues in the testing data.   

## Future Steps:
1.	Add Lane Detection
2.	Prevent Swaying




