import numpy as np
from layers import *

class MLP:
    def __init__(self):
        self.layers = []
        self.mode = 'TRAIN'
        
    def add(self, layer):
        '''
        add a new layer to the layers of model.
        '''
        self.layers.append(layer)
    
    def set_mode(self, mode):
        if mode == 'TRAIN' or mode == 'TEST':
            self.mode = mode
        else:
            raise ValueError('Invalid Mode')
    
    def forward(self, x, y):
        loss, scores = None, None
        ############################################################################
        # TODO: Implement the forward pass of MLP model.                           #
        # Note that you have the layers of the model in self.layers and each one   #
        # of them has a forward() method.                                          #
        # The last layer is always a LossLayer which in this assignment is only    #
        # SoftmaxCrossEntropy.                                                     #
        # You have to compute scores (output of model before applying              #
        # SoftmaxCrossEntropy) and loss which is the output of SoftmaxCrossEntropy #
        # forward pass.                                                            #
        # Do not forget to pass mode=self.mode to forward pass of the layers! It   #
        # will be used later in Dropout and Batch Normalization.                   #
        # Do not forget to add the L2-regularization loss term to loss. You can    #
        # find whether a layer has regularization_strength by using get_reg()      #
        # method. Note that L2-regularization is only used for weights of fully    #
        # connected layers in this assignment.                                     #
        ############################################################################
        out = x
        loss = 0
        for i, layer in enumerate(self.layers, start = 0):
            if i <= len(self.layers)-2:
                out = layer.forward(out, mode=self.mode)
                scores = out
            else:
                out = layer.forward(out, y)
            if isinstance(layer, FullyConnected):
                loss += layer.reg * (np.sum(layer.params['w'].data ** 2) + np.sum(layer.params['b'].data ** 2))
        
        loss += out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return scores, loss
        
        
    def backward(self):
        ############################################################################
        # TODO: Implement the backward pass of the model. Use the backpropagation  #
        # algorithm.                                                               #                         
        # Note that each one of the layers has a backward() method and the last    #
        # layer would always be a SoftmaxCrossEntropy layer.                       #
        ############################################################################
        dout = None
        for i, layer in enumerate(reversed(self.layers), start = 0):
            if i == 0:
                dout = layer.backward()
            else:
                dout = layer.backward(dout)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
            
    def __str__(self):
        '''
        returns a nice representation of model
        '''
        splitter = '===================================================='
        return splitter + '\n' + '\n'.join('layer_{}: '.format(i) + 
                                           layer.__str__() for i, layer in enumerate(self.layers)) + '\n' + splitter
