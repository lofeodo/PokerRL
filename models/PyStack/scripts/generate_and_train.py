import sys
import os
import subprocess
os.chdir('..')
base_dir = os.getcwd()
sys.path.append(os.path.join(base_dir, 'src'))

from Settings.arguments import arguments
from Game.card_to_string_conversion import card_to_string
from arguments_parser import parse_arguments

# Just test with river for initial test
streets = [4] # River only

args = sys.argv[1:]
_, starting_idx, approximate = parse_arguments(args)

for street in streets:
    # Generate data for this street
    street_name = card_to_string.street_to_name(street)
    data_dir = os.path.join(arguments.data_path, street_name, f'{approximate}_npy')
    
    print(f"Processing {street_name}...")
    
    # Use subprocess with environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(base_dir, 'src')
    
    # Generate data
    subprocess.run([
        'python',
        os.path.join(base_dir, 'scripts', 'generate_data.py'),
        '--street', str(street),
        '--starting_idx', str(starting_idx),
        '--approximate', approximate,
        '--sample_size', '100'
    ], env=env)
    
    # Convert sample to TFRecords
    subprocess.run([
        'python',
        os.path.join(base_dir, 'scripts', 'convert_to_tfrecords.py'),
        '--street', str(street),
        '--starting_idx', str(starting_idx),
        '--approximate', approximate
    ], env=env)
    
    # Train with minimal epochs just to test
    subprocess.run([
        'python', 
        os.path.join(base_dir, 'scripts', 'train_nn.py'),
        '--street', str(street),
        '--starting_idx', str(starting_idx),
        '--approximate', approximate,
        '--epochs', '1'
    ], env=env)
    
    print(f"Completed {street_name}")
