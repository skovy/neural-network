from perceptron import Perceptron
from connection import Connection
from input_perceptron import InputPerceptron

# a simple neural network
# params:
#   number_of_inputs: the number of inputs this network needs to accept
#   initial_weights: temporary initial weights for testing
class Network:
  LEARNING_CONSTANT = 0.5

  def __init__(self, number_of_inputs, initial_weights):
    # TODO: remove
    self.initial_weights = initial_weights

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
    connections = [Connection(self.create_bias_perceptron(), self.initial_weights[0])]

    # add all parents as connections as inputs
    for i, parent in enumerate(parents):
      connections.append(Connection(parent, self.initial_weights[i + 1]))

    return Perceptron(connections)

  def run_single_training_input(self, inputs, expected_output):
    if len(inputs) != self.number_of_inputs:
      raise Exception("The number of input values does not match the number of input nodes")

    # first set all of the inputs
    for i in range(0, self.number_of_inputs):
      self.input_perceptrons[i].update_input(inputs[i])

    actual_output = self.perceptron.output()

    if actual_output == expected_output:
      print("Network produced the correct output")
      return True # the correct output
    else:
      print("Network produced the wrong output")

      connections = self.perceptron.get_input_connections()

      for conn in connections:
        # we only update the connections that actually impacted the output of this perceptron
        if  conn.source.output() > 0:
          weight = conn.get_weight()
          adjustment = 0

          if actual_output > expected_output:
            # we need to decrease weights, because our actual output was too large
            adjustment = Network.LEARNING_CONSTANT * -1
          else:
            # we need to increase weights, because our actual output was too small
            adjustment = Network.LEARNING_CONSTANT

          conn.update_weight(weight + adjustment)

      return False # the wrong output





