import sys
import os
os.chdir('..')
sys.path.append(os.path.join(os.getcwd(),'src'))

from Game.card_to_string_conversion import card_to_string
from Settings.arguments import arguments
from arguments_parser import parse_arguments
from train_nn import main as train_nn

# Just test with river for initial test
streets = [4] # River only

args = sys.argv[1:]
_, starting_idx, approximate = parse_arguments(args)

for street in streets:
    # Generate data for this street
    street_name = card_to_string.street_to_name(street)
    data_dir = os.path.join(arguments.data_path, street_name, f'{approximate}_npy')
    
    print(f"Processing {street_name}...")
    
    # Generate tiny dataset for testing
    os.system(f'python scripts/generate_data.py {street} {starting_idx} {approximate} --sample_size=100')
    
    # Convert sample to TFRecords
    os.system(f'python scripts/convert_to_tfrecords.py {street} {starting_idx} {approximate}')
    
    # Train with minimal epochs just to test
    os.system(f'python scripts/train_nn.py {street} {starting_idx} {approximate} --epochs=2')
    
    print(f"Completed {street_name}")
