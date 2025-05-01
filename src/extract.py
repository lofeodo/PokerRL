import tarfile
import json
import os
from tqdm import tqdm



def extract_holdem_hands(input_file, output_file, max_hands=10000):
    """Extract Texas Hold'em hands from the IRC dataset"""
    hands = []
    count = 0
    
    print(f"Opening {input_file}...")
    with tarfile.open(input_file, "r:gz") as tar:
        # Find all holdem files
        holdem_files = [name for name in tar.getnames() 
                        if "holdem" in name.lower() and name.endswith('.tgz')]
        
        print(f"Found {len(holdem_files)} Texas Hold'em files")
        
        for holdem_file in tqdm(holdem_files):
            try:
                # Extract the inner tar file
                inner_file = tar.extractfile(holdem_file)
                if not inner_file:
                    continue
                    
                # Get time period from filename
                period = holdem_file.split('.')[-2]
                
                with tarfile.open(fileobj=inner_file, mode="r:gz") as inner_tar:
                    # Find the hdb file
                    hdb_path = None
                    for name in inner_tar.getnames():
                        if name.endswith('/hdb'):
                            hdb_path = name
                            break
                    
                    if not hdb_path:
                        print(f"No hdb file found in {holdem_file}")
                        continue
                        
                    # Process hdb file
                    hdb_file = inner_tar.extractfile(hdb_path)
                    if not hdb_file:
                        continue
                        
                    for line in hdb_file:
                        parts = line.decode().strip().split()
                        if len(parts) < 8:
                            continue
                            
                        timestamp = parts[0]
                        dealer = parts[1]
                        hand_num = parts[2]
                        num_players = parts[3]
                        
                        # Create a simple hand record
                        hand = {
                            "_id": f"holdem_{period}_{timestamp}",
                            "game": "holdem",
                            "timestamp": timestamp,
                            "dealer": int(dealer),
                            "hand_num": int(hand_num),
                            "num_players": int(num_players)
                        }
                        
                        # Add pot sizes for each stage
                        stages = ["flop", "turn", "river", "showdown"]
                        for i, stage in enumerate(stages):
                            if 4 + i < len(parts):
                                pot_info = parts[4 + i].split('/')
                                if len(pot_info) == 2:
                                    hand[stage] = {
                                        "num_players": int(pot_info[0]),
                                        "pot_size": int(pot_info[1])
                                    }
                        
                        # Add board cards if available
                        if len(parts) > 8:
                            hand["board"] = parts[8:]
                        
                        hands.append(hand)
                        count += 1
                        
                        if count >= max_hands:
                            break
                    
                    if count >= max_hands:
                        break
            except Exception as e:
                print(f"Error processing {holdem_file}: {e}")
                
            if count >= max_hands:
                break
    
    print(f"Extracted {len(hands)} hands")
    
    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(hands, f)
    
    print(f"Saved to {output_file}")
    return hands

if __name__ == "__main__":
    extract_holdem_hands("IRCdata.tgz", "holdem_hands.json", max_hands=10000)