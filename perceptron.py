# a single perceptron in the network
# params:
#   input_connection: an array of Connections that are connected as inputs
class Perceptron:
  def __init__(self, input_connections):
    self.input_connections = input_connections

  def output(self):
    output = 0
    for conn in self.input_connections:
      output += conn.get_source().output() * conn.get_weight()
    return output
