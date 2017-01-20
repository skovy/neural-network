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

    # convert the total sum to a 1 or 0
    output = 0
    if total_sum >= 0:
      output = 1

    print("%s produced sum: %f and output: %d" % (self.identifier, total_sum, output))

    return output

  def get_input_connections(self):
    return self.input_connections
