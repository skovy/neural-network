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
if len(sys.argv) < 2:
  raise Exception("Incorrect parameters. Correct usage: python graph_training_data.py <training_data_file:string>")

training_data_file = sys.argv[1]

# perform training with the provided training set
with open(training_data_file) as data_file:
  data = json.load(data_file)

# all (x, y) pairs that produce a positive output
positive_x = []
positive_y = []

# all (x, y) pairs that produce a negative output
negative_x = []
negative_y = []

for sample in data['data']:
  if sample['expected_output'] == 0:
    negative_x.append(sample['inputs'][0])
    negative_y.append(sample['inputs'][1])
  else:
    positive_x.append(sample['inputs'][0])
    positive_y.append(sample['inputs'][1])


# create traces of the positive and negative outputs
# and generate a scatter plot of the resulting data
positive_trace = go.Scatter(
  name = 'Positive Output',
  x = positive_x,
  y = positive_y,
  mode = 'markers',
  marker = dict(
      size = 10,
      color = '#06D6A0'
  )
)
negative_trace = go.Scatter(
  name = 'Negative Output',
  x = negative_x,
  y = negative_y,
  mode = 'markers',
  marker = dict(
      size = 10,
      color = '#EF476F'
  )
)
data = [positive_trace, negative_trace]
py.offline.plot(data, filename='training.html')
