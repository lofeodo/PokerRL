import sys
import os
import subprocess
base_dir = os.getcwd()
sys.path.append(os.path.join(base_dir, 'src'))

from Settings.arguments import arguments
from Game.card_to_string_conversion import card_to_string
from arguments_parser import parse_arguments

args = sys.argv[1:]
street, starting_idx, approximate = parse_arguments(args)

# Generate data for this street
street_name = card_to_string.street_to_name(street)
data_dir = os.path.join(arguments.data_path, street_name, f'{approximate}_npy')

print(f"Processing {street_name}...")

# Use subprocess with environment variables
env = os.environ.copy()
env['PYTHONPATH'] = os.path.join(base_dir, 'src')

# Generate data
print(f"=== Generating data ===")
subprocess.run([
    'python',
    os.path.join(base_dir, 'scripts', 'generate_data.py'),
    '--street', str(street),
    '--start-idx', str(starting_idx),
    '--approximate', str(approximate),
], env=env)

# Convert sample to TFRecords
print(f"=== Converting data to TFRecords ===")
subprocess.run([
    'python',
    os.path.join(base_dir, 'scripts', 'convert_npy_to_tfrecords.py'),
    '--street', str(street),
    '--start-idx', str(starting_idx),
    '--approximate', str(approximate)
], env=env)

# Train with minimal epochs just to test
print(f"=== Training ===")
subprocess.run([
    'python', 
    os.path.join(base_dir, 'scripts', 'train_nn.py'),
    '--street', str(street),
    '--start-idx', str(starting_idx),
    '--approximate', str(approximate),
    '--epochs', f'{arguments.num_epochs}'
], env=env)

print(f"Completed {street_name}")
