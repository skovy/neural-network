from perceptron import Perceptron
from connection import Connection
from input_perceptron import InputPerceptron

# a simple neural network
# params:
#   number_of_inputs: the number of inputs this network needs to accept
#   configuration: the number of perceptrons and the number of layers (array, each
#                  element is a layer with number of perceptrons)
#   initial_weights: optional, initial weights for perceptrons (3D array), includes bias weights
class Network:
  def __init__(self, number_of_inputs, configuration, initial_weights = []):
    self.number_of_inputs = number_of_inputs # a single integer for the total number of input perceptron
    self.configuration = configuration # a 1D array, each element is a layer, each number is the number of perceptrons
    self.initial_weights = initial_weights # a 3D array of weights, including the bias (last element)

    self.input_perceptrons = [] # an array of the input perceptrons
    self.hidden_perceptrons = [] # a 2D array of the network, each element is a layer, each array contains layer's perceptrons
    self.output_perceptron = None # the last perceptron in the network that provides the final output

    # check validity of initial weights
    if len(self.initial_weights) != 0 and len(self.initial_weights) != len(self.configuration):
      raise Exception("Initial weights doesn't match the configuration")

    # create the entire network, input perceptrons, hidden perceptrons (layers), connections, biases, etc
    self.create_input_perceptrons()
    self.create_hidden_perceptrons()

  # create the desired number of input perceptrons and add them to the network
  def create_input_perceptrons(self):
    # generate the required number of input perceptrons
    for i in range(0, self.number_of_inputs):
      # create an empty input perceptron, we don't know the first input value so initialize blank
      self.input_perceptrons.append(InputPerceptron())

  # create the hidden perceptrons within the network
  def create_hidden_perceptrons(self):
    # generate each layer in the network
    for index in range(0, len(self.configuration)):
      parents = []
      if index == 0:
        # the first layer's parents are the input nodes
        parents = self.input_perceptrons
      else:
        # the second layer and beyond connects to the previous hidden layer
        parents = self.hidden_perceptrons[index - 1]

      layer = self.create_layer(self.configuration[index], parents, self.initial_weights[index])
      self.hidden_perceptrons.append(layer)

    # set the final out perceptron in the network
    # this assumes there is one and only one output perceptron at the end of the network
    self.output_perceptron = self.hidden_perceptrons[-1][0]

  # create a single layer of hidden perceptrons, with connections to their parents and a bias
  def create_layer(self, number_of_perceptrons, parents, layer_initial_weights):
    layer = []

    for index in range(0, number_of_perceptrons):
      perceptron = self.create_hidden_perceptron(parents, layer_initial_weights[index])
      layer.append(perceptron)

    return layer

  # create a hidden perceptron that is part of a layer in the network
  def create_hidden_perceptron(self, parents, perceptron_initial_weights):
    input_connections = []

    # add all parents as input_connections as inputs
    for index, parent in enumerate(parents):
      connection = Connection(parent, perceptron_initial_weights[index])
      input_connections.append(connection)
      # parents need a reference to their outputs later for error correction learning
      parent.add_output_connection(connection)

    # initialize input_connections to the new perceptron with a connection to a bias input perceptron
    input_connections.append(Connection(self.create_bias_perceptron(), perceptron_initial_weights[-1]))

    return Perceptron(input_connections)

  # create a bias perceptron that always outputs 1
  def create_bias_perceptron(self):
    return InputPerceptron(1)

  # print the entire networks final weights
  def final_weights(self):
    self.output_perceptron.final_weights()

  # run a single set of inputs through a "trained" network
  # params:
  #   inputs: an array of inputs for each input perceptron, keep order consistent
  def run_single_input(self, inputs, is_training = False):
    if len(inputs) != self.number_of_inputs:
      raise Exception("The number of input values does not match the number of input nodes")

    # first set all of the inputs
    for i in range(0, self.number_of_inputs):
      self.input_perceptrons[i].update_input(inputs[i])

    return self.output_perceptron.output(is_training)

  # train the network by providing the expected output in addition to the outputs
  # params:
  #   inputs: an array of inputs for each input perceptron, keep order consistent
  #   expected_output: the value that the network should produce, to help with learning
  def run_single_training_input(self, inputs, expected_output):
    actual_output = self.run_single_input(inputs, True)

    # reset the deltas from the previous round
    self.output_perceptron.reset_delta();

    # calculate and set the delta values for each network
    self.output_perceptron.calculate_output_delta(expected_output)
    self.output_perceptron.update_connections()

    output = 0
    if actual_output >= 0:
      output = 1

    if expected_output == output:
      print("Network produced correct value.")
      return True
    else:
      print("Network produced incorrect value.")
      return False

  # train the network provided an array of training data
  # params:
  #   training_set: an array of dictionaries that contain an array of
  #                 `inputs` and integer for the `expected_output`
  def train(self, training_set, training_iterations):
    all_correct = False

    iteration = 0
    while all_correct != True and iteration < training_iterations:
      all_correct = True # assume everything is correct!
      iteration += 1

      print("##### Starting Iteration %d #####" % (iteration))

      for example in training_set:
        correct_output = self.run_single_training_input(example['inputs'], example['expected_output'])
        if not correct_output:
          # one bad examples ruins it for all of us :(
          # but we don't break, we still have to go through the rest of the examples and get a turn :)
          all_correct = False

    print("Training stopped after %d out of max %d iterations over the example set." % (iteration, training_iterations))



