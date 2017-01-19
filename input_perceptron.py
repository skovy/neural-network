# a simple perceptron to handle only the inputs to the network as well as biases
# params:
#   input: the value that is input into the network from this "perceptron"
class InputPerceptron:
  def __init__(self, input):
    self.input = input

  def output(self):
    return self.input

  def update_output(self, new_input):
    self.input = new_input
