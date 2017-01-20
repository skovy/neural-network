# a simple perceptron to handle only the inputs to the network as well as biases
# params:
#   input: the value that is input into the network from this "perceptron"
class InputPerceptron:
  counter = 0

  def __init__(self, input = 0):
    self.input = input

    # create an unique identifier for easier logging
    self.identifier = 'Input Perceptron #{0}'.format(InputPerceptron.counter)
    InputPerceptron.counter += 1
    print("Created %s" % self.identifier)

  def output(self):
    return self.input

  def update_input(self, new_input):
    print("Updating %s input to %d" % (self.identifier, new_input))
    self.input = new_input
