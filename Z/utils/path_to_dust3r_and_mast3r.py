import sys

# Insert paths at the very top of the script
MASt3R_PATH = "/home/dario/_MINE/mast3r"
DUST3R_PATH = "/home/dario/_MINE/mast3r/dust3r/dust3r"

if MASt3R_PATH not in sys.path:
    sys.path.insert(0, MASt3R_PATH)
if DUST3R_PATH not in sys.path:
    sys.path.insert(0, DUST3R_PATH)