## A Simple Neural Network

A simple feed-forward neural network that uses back-propgation for training.

#### Usage

`$ python main.py <configuration_file> <training_data_file> <show_visualization>`

Example: `$ python main.py configs/single-layer.json training-data/xor.json true`

#### Configurations

The `/configs` directory contains network configurations. Each object contains
data to dynamically generate a network. The `number_of_inputs` correponds to
the number of inputs the network should accept. The `config` array corresponds
to the entire network, each element is a layer and each element's value is the
number of perceptrons for that layer. The last element should always be `1` as
the network assumes there is a single output perceptron. The `initial_weights`
correponds to the initial weights for all of the connections. It's a 3-dimensional
array. The first dimension is the entire network and each element is a signle layer,
similar to the `config`. The second dimension is a layer and each element is
a single perceptron in that layer. The third dimension is an individual perceptron
and each element correponds to a connection. The number of weights should be
the `length(previous_layer) + 1`. This is because it needs a connection to every
previous perceptron _and_ a bias perceptron. The last weight is for the bias perceptron.
Order does matter, the first element will correspond to the first connection,
or the first perceptron in the pervious layer, etc. The first hidden layer's previous
layer will be the inputs, or `number_of_inputs + 1`.

#### Training Data

The `/training-data` directory contains examples of training data. Each object
contains a `data` key that corresponds to an array of examples. Each example
contains the `inputs` for each input perceptron and the `expected_output` from
the output perceptron. The network assumes there is only one output perceptron.

Training Data:

- `and.json`: the AND function (`&&`)
- `or.json`: the OR function (`||`)
- `xor.json`: the XOR function
- `nxor.json`: the NXOR function
