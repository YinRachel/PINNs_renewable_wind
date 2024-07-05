This is a model for predicitng the Bus voltage angles under the uncertainty of renewable wind energy. I employ the power system data from MATPOWER Case 9. I combine swing equation in the loss function of neural network. 

## renewable wind power scenario
I use matlab for simulating the wind speed and generate wind power data in a time period of 60 seconds, and add the gust amplification factor from the 20s to 30s to simulate the gust. There are 300 wind scenarios

## training data
Although there are 300 scenarios which I aim to use monte carlo simulation to present the uncertainty brought by wind. But there are only 20 scenarios for training the PINNs, becasue 1) I don't have so much for computation resources, 2) the results show the model show a good performance when there are ony 20 scenario. 

## test data
I employ an unseen wind speed scenario for testing the result, the model show a good performance. But it can have a lot of room for improvement.

## result smoothing 
The predicted value is smoothed 

## Outputs
The script outputs:

A plot comparing the predicted values to the actual data.
An L2 error metric printed to the console.
The MSE, MSE_data and MSE_physics are printed seprarately.
