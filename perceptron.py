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

  def output(self):
    output = 0
    for conn in self.input_connections:
      output += conn.get_source().output() * conn.get_weight()

    print("%s output value: %f" % (self.identifier, output))

    if output >= 0:
      return 1
    else:
      return 0
