
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

import nltk
from nltk.corpus import words
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import os, sys, random, string, warnings, librosa, joblib
import librosa.display
from matplotlib.patches import Ellipse
from hmmlearn import hmm

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


