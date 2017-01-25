import sys
import json

# data manipulation and helpers
import numpy as np

# graphing helpers from Plotyly
import plotly as py
import plotly.graph_objs as go

# the custom neural network
from network import Network

# input validation and conversions
if len(sys.argv) < 5:
  raise Exception("Incorrect parameters. Correct usage: python main.py <configuration_file:string> <training_data_file:string> <training_iterations:integer> <show_visualization:boolean>")

configuration_file = sys.argv[1]
training_data_file = sys.argv[2]
training_iterations = int(sys.argv[3])
show_visualization = (sys.argv[4] == "true")

if show_visualization and len(sys.argv) != 8:
  raise Exception("When creating a visualization, <start_pos:integer> <end_pos:integer> <steps:integer> are required.")

# create a network with inputs, initial weights for the connections
# from the input and the bias input from the desired configuration
with open(configuration_file) as data_file:
  config = json.load(data_file)

n = Network(config['number_of_inputs'], config['config'], config['initial_weights'])

# perform training with the provided training set
with open(training_data_file) as data_file:
  data = json.load(data_file)

n.train(data['data'], training_iterations)
n.final_weights()

if show_visualization:
  steps = int(sys.argv[7]) # number of steps on each axis
  start_pos = int(sys.argv[5]) * steps # the x and y mins
  end_pos = int(sys.argv[6]) * steps # the x and y max

  # all (x, y) pairs that produce a positive output
  positive_x = []
  positive_y = []

  # all (x, y) pairs that produce a negative output
  negative_x = []
  negative_y = []

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
