from perceptron import Perceptron
from connection import Connection
from input_perceptron import InputPerceptron

LEARNING_CONSTANT = 0.5

# a simple neural network
# params:
#   number_of_inputs: the number of inputs this network needs to accept
class Network:
  def __init__(self, number_of_inputs):
    self.number_of_inputs = number_of_inputs
    self.input_perceptrons = []

    # generate the required number of input perceptrons
    for i in range(0, number_of_inputs):
      self.input_perceptrons.append(InputPerceptron())

    # TODO: make dynamic, currently only a single perceptron with n inputs
    self.perceptron = self.create_hidden_perceptron(self.input_perceptrons)

  # create a bias perceptron that always outputs 1
  def create_bias_perceptron(self):
    return InputPerceptron(1)

  # create a hidden perceptron that is part of a layer in the network
  def create_hidden_perceptron(self, parents):
    # initialize connections to the new perceptron with a connection to a bias input perceptron
    connections = [Connection(self.create_bias_perceptron(), LEARNING_CONSTANT)]

    # add all parents as connections as inputs
    for parent in parents:
      connections.append(Connection(parent, LEARNING_CONSTANT))

    return Perceptron(connections)

  def run_single_input(self, inputs):
    if len(inputs) != self.number_of_inputs:
      raise Exception("The number of input values does not match the number of input nodes")

    # first set all of the inputs
    for i in range(0, self.number_of_inputs):
      self.input_perceptrons[i].update_input(inputs[i])

    print(self.perceptron.output())




