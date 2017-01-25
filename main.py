import sys
import json

# data manipulation and helpers
import numpy as np

# graphing helpers from Plotyly
import plotly as py
import plotly.graph_objs as go

# the custom neural network
from network import Network

if len(sys.argv) != 3:
  raise Exception("Incorrect parameters. Correct usage: python main.py <training_data_file:string> <show_visualization:boolean>")

training_data_file = sys.argv[1]
show_visualization = (sys.argv[2] == "true")

configuration = [4, 1]

# configuration = [2, 1]
# initial_weights = [[[2, -2, 0], [1, 3, -1]], [[3, -2, -1]]]

initial_weights = [[[-0.1, -0.1, 0.2], [-0.1, 0.2, 0.3], [0.4, -0.5, 0.1], [0.2, 0.1, -0.3]], [[-0.8, 0.3, 0.2, 0.1, 0.2]]]

# create a network with 2 inputs and the initial weights for the connections
# from the input and the bias input
n = Network(2, configuration, initial_weights)

# a simple training set to learn the OR function
or_function_training_set =

# a simple training set to learn the AND function
and_function_training_set =

# a simple training set to learn the NXOR function
nxor_function_training_set =

with open(training_data_file) as data_file:
    data = json.load(data_file)

# perform training with the provided training set
n.train(data['data'])
n.final_weights()

if show_visualization:
  # all (x, y) pairs that produce a positive output
  positive_x = []
  positive_y = []

  # all (x, y) pairs that produce a negative output
  negative_x = []
  negative_y = []

  steps = 1 # number of steps on each axis
  start_pos = 0 * steps # the x and y mins
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
          size = 15,
          color = '#06D6A0'
      )
  )
  negative_trace = go.Scatter(
      name = 'Negative Output',
      x = negative_x,
      y = negative_y,
      mode = 'markers',
      marker = dict(
          size = 15,
          color = '#EF476F'
      )
  )
  data = [positive_trace, negative_trace]
  py.offline.plot(data, filename='scatter.html')
