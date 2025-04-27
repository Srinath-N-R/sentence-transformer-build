import regex as re

pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\s+""")
PAD_ID = 998
CLS_ID = 999
IGNORE_INDEX = 111
MAX_LEN = 512
VOCAB_SIZE = 1000
D_MODEL = 64
NHEAD = 8
NUM_LAYERS = 4
PROJ_DIM = 32
DROPOUT = 0.1
MARGIN = 0.2