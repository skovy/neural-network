import math

# a single perceptron in the network
# params:
#   input_connection: an array of Connections that are connected as inputs
class Perceptron:
  counter = 0

  def __init__(self, input_connections):
    self.input_connections = input_connections
    self.last_output = None
    self.last_delta = None

    # create an unique identifier for easier logging
    self.identifier = 'Perceptron #{0}'.format(Perceptron.counter)
    Perceptron.counter += 1
    print("Created %s" % self.identifier)

  def __str__(self):
     return self.identifier

  def output(self, is_training = False):
    total_sum = 0
    for conn in self.input_connections:
      total_sum += conn.get_source().output(is_training) * conn.get_weight()

    # determine the final result, we only use the sigmoid function when performing training
    # otherwise we just use the weighted sums
    final_result = 0
    if is_training:
      final_result = self.sigmoid(total_sum)
    else:
      final_result = total_sum

    print("%s produced total sum: %f and sigmoid: %f and output: %d" % (self.identifier, total_sum, final_result, 0))

    self.last_output = final_result
    return final_result

  def get_input_connections(self):
    return self.input_connections

  # calculate the sigmoid value
  def sigmoid(self, x):
    return 1 / (1 + math.exp(-x))

  # calculate the delta of error for the final output node
  def calculate_output_delta(self, desired):
    self.last_delta = self.last_output * (1 - self.last_output) * (desired - self.last_output)
    print("%s produced delta value: %f" % (self.identifier, self.last_delta))

    self.calculate_previous_layer_delta()

  # call the next layers perceptrons and calculate their delta's using this delta value
  def calculate_previous_layer_delta(self):
    for conn in self.input_connections:
      if isinstance(conn.get_source(), Perceptron):
        conn.get_source().calculate_delta(self.last_delta * conn.get_weight())

  # calculate the delta of error for a hidden layer perceptron
  def calculate_delta(self, previous_weighted_delta):
    self.last_delta = self.last_output * (1 - self.last_output) * previous_weighted_delta
    print("%s produced delta value: %f" % (self.identifier, self.last_delta))

    self.calculate_previous_layer_delta()

  def final_weights(self):
    for conn in self.get_input_connections():
      # only print other perceptrons in the network and ignore biases/inputs/etc
      if isinstance(conn.get_source(), Perceptron):
        conn.get_source().final_weights()

      print("%s with final weight: %f" % (conn, conn.get_weight()))
