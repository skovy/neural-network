# a simple perceptron to handle only the inputs to the network as well as biases
# params:
#   input: the value that is input into the network from this "perceptron"
class InputPerceptron:
  counter = 0

  def __init__(self, input = None):
    self.input = input
    self.output_connections = []

    # create an unique identifier for easier logging
    self.identifier = 'Input Perceptron #{0}'.format(InputPerceptron.counter)
    InputPerceptron.counter += 1
    print("Created %s" % self.identifier)

  def __str__(self):
     return self.identifier

  def add_output_connection(self, output_connection):
    self.output_connections.append(output_connection)

  def output(self, is_training = False):
    return self.input

  def get_last_output(self):
    return self.input

  def update_input(self, new_input):
    print("Updating %s input to %f" % (self.identifier, new_input))
    self.input = new_input
