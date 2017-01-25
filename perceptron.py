import math

# a single perceptron in the network
# params:
#   input_connection: an array of Connections that are connected as inputs
class Perceptron:
  counter = 0

  def __init__(self, input_connections):
    self.input_connections = input_connections

    # create an unique identifier for easier logging
    self.identifier = 'Perceptron #{0}'.format(Perceptron.counter)
    Perceptron.counter += 1
    print("Created %s" % self.identifier)

  def __str__(self):
     return self.identifier

  def output(self):
    total_sum = 0
    for conn in self.input_connections:
      total_sum += conn.get_source().output() * conn.get_weight()

    final_result = self.sigmoid(total_sum)

    print("%s produced total sum: %f and sigmoid: %f and output: %d" % (self.identifier, total_sum, final_result, 0))

    return final_result

  def get_input_connections(self):
    return self.input_connections

  # calculate the sigmoid value
  def sigmoid(self, x):
    return 1 / (1 + math.exp(-x))

  def final_weights(self):
    for conn in self.get_input_connections():
      # only print other perceptrons in the network and ignore biases/inputs/etc
      if isinstance(conn.get_source(), Perceptron):
        conn.get_source().final_weights()

      print("%s with final weight: %f" % (conn, conn.get_weight()))
