# Poker Hands Dataset

Simple scripts to extract and browse the IRC Poker Database. Note here only hold'em hands are included. Most (over 95%) hands are dropped either because they're not hold'em or because of data corruption.

## Usage

Setting up the environment:

```bash
python -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
```

Download the compressed file from University of Alberta website, and extract all hold'em hands data into `hands.json`:

```bash
curl -O http://poker.cs.ualberta.ca/IRC/IRCdata.tgz  # or wget, whichever works on your OS
python3 src/extract.py
```

There will be 9,478,019 hands extracted in total (after roughly 9 hours of running). Majority of hand data will be skipped due to either invalid game type or corrupted records.

Browse the extracted hands in console:

```bash
python3 src/browse.py
```

## References

- [IRC Poker Database](http://poker.cs.ualberta.ca/irc_poker_database.html), Computer Poker Research Group, University of Alberta, 2017.
- [Miami Data Science Meetup](https://github.com/dksmith01/MSDM/blob/987836595c73423b89f83b29747956129bec16c2/.ipynb_checkpoints/MDSM%20Project%201%20Poker%20Python%20Wrangling%20Code-checkpoint.ipynb), 2015.
