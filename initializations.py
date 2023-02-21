import torch
import torch.nn.functional as F
import numpy as np

def weight_init(method, hidden_size, input_size, output_size, g):
    '''
    INPUTS
    hidden_size = number of nodes in hidden layer
    input_size = number of nodes in input layer
    output_size = number of nodes in output layer
    g = self.g value
    
    OUTPUTS
    weight_ih = input-to-hidden weights
    weight_oh = hidden-to-output weights
    W = hidden-to-hidden weights
    '''

    root_inv_hidden = 1 / np.sqrt(hidden_size)

    
    if method == 'log':
        #this was Dennis's idea! since apparently most biological distributions are log normal

        #input-to-hidden
        weight_ih = torch.FloatTensor(hidden_size, input_size).log_normal_(-root_inv_hidden, root_inv_hidden)

        #hidden-to-output
        weight_oh = torch.FloatTensor(output_size, hidden_size).log_normal_(-root_inv_hidden, root_inv_hidden)

        #hidden-to-hidden
        #W = torch.FloatTensor(hidden_size, hidden_size).log_normal_(-root_inv_hidden, root_inv_hidden)
        #copying Leo to shrink these down
        W = torch.FloatTensor(hidden_size, hidden_size).log_normal_(0, g * root_inv_hidden)
        W /= 10 * (torch.linalg.vector_norm(W, ord=2))


        
    elif method == 'uniform':
        #input-to-hidden
        weight_ih = torch.FloatTensor(hidden_size, input_size).uniform_(-root_inv_hidden, root_inv_hidden)

        #hidden-to-output
        weight_oh = torch.FloatTensor(output_size, hidden_size).uniform_(-root_inv_hidden, root_inv_hidden)

        #hidden-to-hidden: start as log normal, then divide by 10*norm to normalize values to 0-0.1
        W = torch.FloatTensor(hidden_size, hidden_size).log_normal_(0, g * root_inv_hidden)
        W /= 10 * (torch.linalg.vector_norm(W, ord=2))
        

    elif method == 'He':
        '''
        The He initialization method is calculated as a random number with a Gaussian probability distribution (G)
        with a mean of 0.0 and a standard deviation of sqrt(2/n), where n is the number of inputs to the node.
        weight = G (0.0, sqrt(2/n))
        '''

        # input-to-hidden 
        weight_ih = torch.normal(0, np.sqrt(2 / (input_size)), (hidden_size, input_size))

        # hidden-to-output 
        weight_oh = torch.normal(0, np.sqrt(2 / (hidden_size)), (output_size, hidden_size))

        # hidden-to-hidden
        W = torch.normal(0, g * np.sqrt(2 / (hidden_size-1)), (hidden_size, hidden_size)) #not sure if self.g placement here makes sense (i just guessed)



    elif method == 'normal':
        #input-to-hidden
        weight_ih = torch.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, input_size))

        #hidden-to-output
        weight_oh = torch.normal(0, 1 / np.sqrt(hidden_size), (output_size, hidden_size))

        #hidden-to-hidden
        W = torch.normal(0, g / np.sqrt(hidden_size), (hidden_size, hidden_size))

    
    
    W.fill_diagonal_(0) #remove autapses
        
    return weight_ih, weight_oh, W