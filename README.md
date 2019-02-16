# Self Driving Car Simulation
I was inspired to do this project by Siraj Raval from Youtube. 

## Overview

In this project I will be using a Neural Network to drive a car around a  track in a simulator. The simulator we're going to use Udacity's [self driving car simulator](https://github.com/udacity/self-driving-car-sim) as a testbed for training an autonomous car. Along with the simulator I was provided a pre-trained model, I will be attempting to outperform this model. 

## Usage
The models that I will be posting can be run by running the command python drive.py "modelname.h5"

## Model Versions
1.	**model-Pre-Trained** - Pre-trained model created
This model performs fairly well from the beginning, it stays on the track the entire time.
2.	**model-V1.00** - Model trained using 2 laps of desert test data and 2 epochs and learning rate of 1.0e-4
This model performed poorly, did mild amounts of steering but encountered issues with sharp turns and swayed within the lane. 
3.	**model-V1.01** - Model trained using 2 laps of desert test data and 10 epochs and learning rate of 1.0e-4
This model performed poorly, but was able to complete half a lap, did not expereince the swaying as much as the V1.00 model.
4.	**model-V1.02** - Model trained using 2 laps of desert test data and 10 epochs and learning rate of 2.0e-4
This model performed extremely well, the faster learning rate helped the algorithim tune its self much better and was able to perform on p ar with the Pre-trained model. I was fairly surprised that such a small change could affect the outcome of the model so much.
Testing model-V1.02 on a track that it has never seen before yielding unexpected results the neural network was able to manage part of the course which is surprising since many other models couldnt make it past the first turn.
5.	**model-V1.03** - Model trained using 2 laps of desert test data and 10 epochs and learning rate of 1.0e-3
For this model I had the idea that since increasing the training rate in my last example improved the output model by a lot. I researched other training rates, many suggested to increment by a factor of 10. 
The results were quite surprising to me since the model performed extremenly bad. It could not turn properly, and appeard to have learned nothing. This is most likely to the fact that when using gradient decent if the learning rate is to high the steps it takes are to large and thus does not learn quickly
6.	**model-V1.04** - Model trained using 2 laps of desert test data and 10 epochs and learning rate of 5.0e-4
As expected the training rate was way to high and the model moved forward with a slight steer to the right. I found this interesting because most of the track consists of the model steering to the left. Similarly to model-V1.03, the model moved straight with a slight steer to the right. This could mean there are issues in the testing data.
7.	**model-V1.05** - Model trained using 5 laps of desert test data and 10 epochs and learning rate of 2.0e-4
This model was trained based on 5 laps from the desert track meaning it has a larger dataset to learn from and hopefully from that we will be able to see some improvements. The model turned out to be a big disappointment, I thought that with more data it would be able to perform better. Testing on another track yielded similar results, this model performed poorly. Model was unable to make it around the first turn under different conditions. 

## Lessons Learned
### Learning rate: 
Learning rates have large effects on the outcome of the models, in model-V1.01 I used a Learning rate of 1.0e-4 which resulted in a model that was able to complete half a lap in the simulator. In the next model-V1.02 I doubled the Learning rate and the model was able to complete laps flawlessly which was a huge improvement from for a minor change. Upon analyzing the epochs I noticed that when the Learning rate was ideal a steady decline in the loss and val_loss functions can be seen below:
Training Model with learning rate 2.0e-4
20000/20000 [==============================] - 135s - loss: 0.0407 - val_loss: 0.0233
Epoch 2/10
20000/20000 [==============================] - 114s - loss: 0.0381 - val_loss: 0.0221
Epoch 3/10
20000/20000 [==============================] - 111s - loss: 0.0366 - val_loss: 0.0228
Epoch 4/10
20000/20000 [==============================] - 112s - loss: 0.0335 - val_loss: 0.0238
Epoch 5/10
20000/20000 [==============================] - 111s - loss: 0.0320 - val_loss: 0.0255
Epoch 6/10
20000/20000 [==============================] - 111s - loss: 0.0305 - val_loss: 0.0234
Epoch 7/10
20000/20000 [==============================] - 111s - loss: 0.0294 - val_loss: 0.0238
Epoch 8/10
20000/20000 [==============================] - 112s - loss: 0.0288 - val_loss: 0.0230
Epoch 9/10
20000/20000 [==============================] - 112s - loss: 0.0287 - val_loss: 0.0172
Epoch 10/10
20000/20000 [==============================] - 112s - loss: 0.0262 - val_loss: 0.0182

### Size of Data
From my current experiment their were not any improved results from training the networks on a larger data set. In fact the models trained on the larger data sets performed worse than those trained on the smaller datasets. This most likely has to do with the complexity of the neural network, if its a larger network then more data will be required to adjust the weights.

More experimentation is requried.

## Future Steps:
1.	Add Lane Detection
2.	Prevent Swaying






