# a connection is a link in the neural network between two perceptrons
class Connection:
  # params:
  #   source: the perceptron that will feed it's output through this connection
  #   initial_weight: the initial weight of this connection
  def __init__(self, source, initial_weight):
    self.source = source
    self.weight = initial_weight

  def get_source(self):
    return self.source

  def get_weight(self):
    return self.weight
