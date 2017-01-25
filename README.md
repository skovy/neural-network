## A Simple Neural Network

A simple feed-forward neural network that uses back-propgation for training.

#### Install Packages

- `$ pip install --target=. plotly`

#### Usage

`$ python main.py <configuration_file:string> <training_data_file:string> <training_iterations:integer> <show_visualization:boolean> <start_pos:integer> <end_pos:integer> <steps:integer>`

- `configuration_file`: the configuration file to represent the network
- `training_data_file`: the training data file to train the network
- `training_iterations`: a number representing the maximum number of training iterations
- `show_visualization`: a boolean whether to generate a visualization of the trained network
- `start_pos`: _(required if show visualization is true)_ the start position of the `x`, `y` coordinates
- `end_pos`: _(required if show visualization is true)_ the end position of the `x`, `y` coordinates
- `steps`: _(required if show visualization is true)_ the number of steps between each point

Examples:

- `$ python main.py configs/single-layer.json training-data/xor.json 100 false`
- `$ python main.py configs/single-layer.json training-data/xor.json 4000 true -1 2 40`

#### Configurations

The `/configs` directory contains network configurations. Each object contains
data to dynamically generate a network. The `number_of_inputs` corresponds to
the number of inputs the network should accept. The `config` array corresponds
to the entire network, each element is a layer and each element's value is the
number of perceptrons for that layer. The last element should always be `1` as
the network assumes there is a single output perceptron. The `initial_weights`
corresponds to the initial weights for all of the connections. It's a 3-dimensional
array. The first dimension is the entire network and each element is a single layer,
similar to the `config`. The second dimension is a layer and each element is
a single perceptron in that layer. The third dimension is an individual perceptron
and each element corresponds to a connection. The number of weights should be
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

#### Examples

Executing `$ python main.py configs/single-layer.json training-data/xor.json 4000
true -1 2 40` produces something like the following. It creates a network with
2 input nodes, a hidden layer with 4 perceptrons and single output perceptron.

![single-hidden-layer](/assets/single-hidden-layer.png)

Provided with the 4 training examples for the XOR function, with a maximum iteration
of `4000` for training and graphing from `(-1, -1)` to `(2, 2)` with `20` steps per
unit produces the following visualization.

![XOR](/assets/xor.png)
