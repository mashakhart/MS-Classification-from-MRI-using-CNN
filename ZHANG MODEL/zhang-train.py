import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab, vocab
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import re
from collections import Counter
from typing import List, Tuple, Dict, Optional, Any

!wget --quiet https://princeton-nlp.github.io/cos484/assignments/a2/eng.train #change this, obv
!wget --quiet https://princeton-nlp.github.io/cos484/assignments/a2/eng.val #change this, obv
!cat eng.train | head -n 50

