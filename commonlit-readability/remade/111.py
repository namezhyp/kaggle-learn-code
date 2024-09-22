import numpy as np

class LinearLayer:
    def __init__(self,input_size,output_size):
        self.weights=np.random.randn(input_size,output_size)
        self.bias=np.zeros(output_size)

    def forward(self,input_data):
        self.input=input_data
        self.output=np.dot(self.input,self.weights)+self.bias
        return self.output