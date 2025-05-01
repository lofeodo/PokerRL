import tarfile

with tarfile.open("IRCdata.tgz", "r:gz") as tar:
    holdem_files = [name for name in tar.getnames() if "holdem" in name.lower() or "nolimit" in name.lower()]
    
    print(f"Found {len(holdem_files)} potential Texas Hold'em files:")
    for file in holdem_files[:10]:  # Print first 10
        print(f" - {file}")