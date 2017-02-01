# a connection is a link in the neural network between two perceptrons
class Connection:
  counter = 0

  # params:
  #   source: the perceptron that will feed it's output through this connection
  #   initial_weight: the initial weight of this connection
  def __init__(self, source, initial_weight):
    self.source = source
    self.sink = None
    self.weight = initial_weight

    # create an unique identifier for easier logging
    self.identifier = 'Connection #{0} [Source: {1}]'.format(Connection.counter, self.source)
    Connection.counter += 1
    print("Created %s with weight %f" % (self.identifier, self.weight))

  def __str__(self):
     return self.identifier

  def get_source(self):
    return self.source

  def get_sink(self):
    return self.sink

  def get_weight(self):
    return self.weight

  def update_weight(self, new_weight):
    self.weight = new_weight
    print("Updated %s with weight %f" % (self.identifier, self.weight))

  def set_sink(self, sink):
    self.sink = sink

