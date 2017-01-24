# data manipulation and helpers
import numpy as np

# graphing helpers
import plotly as py
import plotly.graph_objs as go

# the neural network
from network import Network

# create a network with 2 inputs and the initial weights for the connections
# from the input and the bias input
n = Network(2, [0.1, 0.2, 0.3])

# a simple training set to learn the OR function
or_function_training_set = [
  { 'inputs': [0, 0], 'expected_output': 0 },
  { 'inputs': [0, 1], 'expected_output': 1 },
  { 'inputs': [1, 0], 'expected_output': 1 },
  { 'inputs': [1, 1], 'expected_output': 1 }
]

# a simple training set to learn the AND function
and_function_training_set = [
  { 'inputs': [0, 0], 'expected_output': 0 },
  { 'inputs': [0, 1], 'expected_output': 0 },
  { 'inputs': [1, 0], 'expected_output': 0 },
  { 'inputs': [1, 1], 'expected_output': 1 }
]

# perform training with the provided training set
n.train(or_function_training_set)
n.final_weights()

# all (x, y) pairs that produce a positive output
positive_x = []
positive_y = []

# all (x, y) pairs that produce a negative output
negative_x = []
negative_y = []

steps = 5 # number of steps on each axis
start_pos = -1 * steps # the x and y mins
end_pos = 2 * steps # the x and y max

# generate a "matrix" with much finer steps to "brute-force chart" the resulting function
for i in range(start_pos, end_pos):
  for j in range(start_pos, end_pos):
    x = i / float(steps)
    y = j / float(steps)
    if n.run_single_input([x, y]):
      positive_x.append(x)
      positive_y.append(y)
    else:
      negative_x.append(x)
      negative_y.append(y)

# create traces of the positive and negative outputs
# and generate a scatter plot of the resulting data
positive_trace = go.Scatter(
    name = 'Positive Output',
    x = positive_x,
    y = positive_y,
    mode = 'markers',
    marker = dict(
        size = 4,
        color = '#06D6A0'
    )
)
negative_trace = go.Scatter(
    name = 'Negative Output',
    x = negative_x,
    y = negative_y,
    mode = 'markers',
    marker = dict(
        size = 4,
        color = '#EF476F'
    )
)
data = [positive_trace, negative_trace]
py.offline.plot(data, filename='scatter.html')
