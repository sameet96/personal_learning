#%%
#==========================================================
#Single nod neural net
#==========================================================
weight = 0.1  # weight for the single node of neural network

def neural_network(input, weight):
    prediction = input * weight
    return prediction

#%% 
number_of_toes = [8.5, 9.5, 10, 9]
input = number_of_toes[0]
pred = neural_network(input, weight)
print(pred)

#%%
#==========================================================
#Multiple input and weights neural network
#==========================================================

def w_sum(a,b):
    assert(len(a) == len(b))
    output = 0
    for i in range(len(a)):
        output += (a[i] * b[i])
    return output
# # # weights = [0.1, 0.2, 0]
def neural_network(input, weight):
    prediction = w_sum(input ,weight)
    return prediction

#%%

toes = [8.5, 9.5, 9.9, 9.0]
games_won = [0.65,0.8,0.8,0.9]
number_of_fans = [1.2, 1.3, 0.5, 1.0]

input = [toes[0], games_won[0], number_of_fans[0]] # dot products or vector
                # multiplication as it gives you the notion similarity between two vectors(similar to AND operation)
pred = neural_network(input, weights)
print(pred)
# %%

#Numpy version for multinode network
import numpy as np
weights = np.array([0.1,0.2,0])
def neural_network(input, weights):
    pred = input.dot(weights)
    return pred

input = np.array([toes[0], games_won[0], number_of_fans[0]])
pred = neural_network(input, weights)
print(pred)

#%%
###############################################################
## Predicting with multiple inputs and multiple outputs using Hidden layers
###############################################################
 
import numpy as np

hidden_weight = np.array([
    [0.1, 0.2, -0.1],  #hidden[0]
    [-0.1, 0.1, 0.9],  #hidden[1]
    [0.1, 0.4, 0.1]  #hidden[2]
]).T

hp_wgt = np.array([
    [0.3, 1.1, -0.3],  #hurt?
    [0.1, 0.2, 0.0],  #win?
    [0.0, 0.3, 0.1]  #sad?
]).T

weights = [hidden_weight, hp_wgt]

def neural_nework(input, weights):
    hid = input.dot(weights[0])
    pred = hid.dot(weights[1])
    return pred

toes = [8.5, 9.5, 9.9, 9.0]
games_won = [0.65,0.8,0.8,0.9]
number_of_fans = [1.2, 1.3, 0.5, 1.0]

input = np.array([toes[0], games_won[0], number_of_fans[0]])

pred = neural_network(input, weights)
print(pred)