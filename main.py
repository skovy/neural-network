from network import Network

n = Network(2, [0.1, 0.2, 0.3])

training_set = [
  { 'inputs': [0, 0], 'expected_output': 0 },
  { 'inputs': [0, 1], 'expected_output': 1 },
  { 'inputs': [1, 0], 'expected_output': 1 },
  { 'inputs': [1, 1], 'expected_output': 1 }
]

all_correct = False

while all_correct != True:
  all_correct = True # assume everything is correct!

  for example in training_set:
    correct_output = n.run_single_training_input(example['inputs'], example['expected_output'])
    if not correct_output:
      # one bad examples ruins it for all of us :(
      # but we don't break, we still have to go through the rest of the examples and get a turn :)
      all_correct = False
