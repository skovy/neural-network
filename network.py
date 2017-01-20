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

  # TODO: make dyanmic, assuming a single perceptron network
  def final_weights(self):
    for conn in self.perceptron.get_input_connections():
      print("%s with final weight: %f" % (conn, conn.get_weight()))

  # run a single set of inputs through a "trained" network
  # params:
  #   inputs: an array of inputs for each input perceptron, keep order consistent
  def run_single_input(self, inputs):
    if len(inputs) != self.number_of_inputs:
      raise Exception("The number of input values does not match the number of input nodes")

    # first set all of the inputs
    for i in range(0, self.number_of_inputs):
      self.input_perceptrons[i].update_input(inputs[i])

    return self.perceptron.output()

  # train the network by providing the expected output in addition to the outputs
  # params:
  #   inputs: an array of inputs for each input perceptron, keep order consistent
  #   expected_output: the value that the network should produce, to help with learning
  def run_single_training_input(self, inputs, expected_output):
    actual_output = self.run_single_input(inputs)

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

  # train the network provided an array of training data
  # params:
  #   training_set: an array of dictionaries that contain an array of
  #                 `inputs` and integer for the `expected_output`
  def train(self, training_set):
    all_correct = False

    while all_correct != True:
      all_correct = True # assume everything is correct!

      for example in training_set:
        correct_output = self.run_single_training_input(example['inputs'], example['expected_output'])
        if not correct_output:
          # one bad examples ruins it for all of us :(
          # but we don't break, we still have to go through the rest of the examples and get a turn :)
          all_correct = False



