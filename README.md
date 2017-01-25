### A Simple Neural Network

A simple feed-forward neural network that uses back-propgation for training.

#### Usage

`$ python main.py <training_data_file> <show_visualization>`

Example: `$ python main.py training-data/xor.json true`

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
