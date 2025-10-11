import os
from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")