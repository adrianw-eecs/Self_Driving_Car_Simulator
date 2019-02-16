# Self Driving Car Simulation
I was inspired to do this project by Siraj Raval from Youtube. 

## Overview

In this project I will be using a Neural Network to drive a car around a  track in a simulator. The simulator we're going to use Udacity's [self driving car simulator](https://github.com/udacity/self-driving-car-sim) as a testbed for training an autonomous car. Along with the simulator I was provided a pre-trained model, I will be attempting to outperform this model. 

## Usage
The models that I will be posting can be run by running the command python drive.py "modelname.h5"

## Model Versions
### Current Best Model: **model-V1.02**
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

8. **model-V1.06** - Model trained using 5 laps of desert test data and 10 epochs and learning rate of 2.0e-4

This model was used to see how the neural network changes after each Epoch and helped me understand the difference between loss and val_loss. More results can be seen in the section "Evaluation of how Epochs change the model".

## Lessons Learned
### Learning rate: 
Learning rates have large effects on the outcome of the models, in model-V1.01 I used a Learning rate of 1.0e-4 which resulted in a model that was able to complete half a lap in the simulator. In the next model-V1.02 I doubled the Learning rate and the model was able to complete laps flawlessly which was a huge improvement from for a minor change. Upon analyzing the epochs I noticed that when the Learning rate was ideal a steady decline in the loss and val_loss functions can be seen below:

Training Model with learning rate 2.0e-4
```
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
```
### Size of Data
From my current experiment their were not any improved results from training the networks on a larger data set. In fact the models trained on the larger data sets performed worse than those trained on the smaller datasets which I did not expect. The models trained on larger datasets would have had more expereince and should expereince the overfitting problem less. This most likely has to do with the complexity of the neural network, if its a larger network then more data will be required to adjust the weights.

More experimentation is requried.

### Evaluation of how Epochs change the model
20000/20000 [==============================] - 117s - loss: 0.0313 - val_loss: 0.0134
Using *model-V1.06* I used keras.callbacks to create checkpoints of how every epoch changes the behavior of the model. Looking at the training stats below one may assume that after the 4th Epoch(loss: 0.0313 - val_loss: 0.0134) would yield the highest performance due to the extremely low val_loss of 0.0134 which was the lowest out of the batch however when testing this model it was unable to make sharp turns and was swaying back and fourth in the lane.

When testing the after the 6th Epoch the model fixed the swaying issue noticed in the previous versions and was able to make it 3/4ths the way through the track, getting stuck on the sharp right turn. Which I noticed was an issue with all the other version. None of these models were able to complete an entire lap. 
```
20000/20000 [==============================] - 118s - loss: 0.0357 - val_loss: 0.0169
Epoch 2/10
20000/20000 [==============================] - 117s - loss: 0.0336 - val_loss: 0.0194
Epoch 3/10
20000/20000 [==============================] - 116s - loss: 0.0332 - val_loss: 0.0205
Epoch 4/10
20000/20000 [==============================] - 117s - loss: 0.0313 - val_loss: 0.0134
Epoch 5/10
20000/20000 [==============================] - 117s - loss: 0.0305 - val_loss: 0.0147
Epoch 6/10
20000/20000 [==============================] - 117s - loss: 0.0292 - val_loss: 0.0176
Epoch 7/10
20000/20000 [==============================] - 117s - loss: 0.0293 - val_loss: 0.0147
Epoch 8/10
20000/20000 [==============================] - 117s - loss: 0.0293 - val_loss: 0.0145
Epoch 9/10
20000/20000 [==============================] - 117s - loss: 0.0281 - val_loss: 0.0143
Epoch 10/10
20000/20000 [==============================] - 118s - loss: 0.0278 - val_loss: 0.0172
``` 

## Future Steps:
1.	Add Lane Detection







