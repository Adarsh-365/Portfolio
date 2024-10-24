import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivaive_sigmoid(x):
    return x*(1-x)

input_data = np.array([[0,0],
                       [0,1],
                       [1,0],
                       [1,1]])

output = np.array([[0],[1],[1],[0]])

np.random.seed(42)

input_node = 2
hidden_h1_node = 2
output_node = 1

weight_h1 = np.random.uniform(1,-1,size=(input_node,hidden_h1_node))
weight_output = np.random.uniform(1,-1,size=(hidden_h1_node,output_node))

bias_h1 =np.random.uniform(size=(1,hidden_h1_node))
bias_output =np.random.uniform(size=(1,output_node))

learning_rate = 0.1
epoch = 10000
for i in range(epoch):
    
    h1_input = np.dot(input_data,weight_h1) + bias_h1
    h1_output = sigmoid(h1_input)
    
    op_input = np.dot(h1_output,weight_output)+bias_output
    predicted_op = sigmoid(op_input)
    
    error = output - predicted_op
    epoch= epoch+1
    
  
    
    #backpropogatation
    d_output = error*derivaive_sigmoid(predicted_op)
    h1_error = np.dot(d_output,weight_output.T)
    
    d_h1 = h1_error*derivaive_sigmoid(h1_output)
    
    weight_output += np.dot(h1_output.T,d_output)*learning_rate
    bias_output += np.sum(d_output,axis=0,keepdims=1)*learning_rate
    
    weight_h1 += np.dot(input_data.T,d_h1)*learning_rate
    bias_h1 += np.sum(d_h1,axis=0,keepdims=1)*learning_rate
    
     
print(predicted_op) 
print(np.sum(error),epoch)
