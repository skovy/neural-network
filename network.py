from perceptron import Perceptron
from connection import Connection
from input_perceptron import InputPerceptron

class Network:
  def __init__(self):
    # a single, simple perceptron Network
    # TODO: what should the first value be here?
    self.input = InputPerceptron(5)

    # bias is always set to 1
    bias = InputPerceptron(1)

    # TODO: update initial weight
    first_connection = Connection(self.input, 0.5)
    bias_connection = Connection(bias, 0.5)

    self.perceptron = Perceptron([first_connection, bias_connection])

    print(self.perceptron.output())


