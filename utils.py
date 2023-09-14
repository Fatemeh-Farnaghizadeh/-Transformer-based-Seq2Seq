import torch 
from torch import nn
import pandas as pd
import spacy
import json

#Dataset Settings
EN_TRAIN_PATH = r'.\data\train_en.txt'
DE_TRAIN_PATH = r'.\data\train_de.txt'

EN_VAL_PATH = r'.\data\val_en.txt'
DE_VAL_PATH = r'.\data\val_de.txt'

EN_TEST_PATH = r'.\data\test_en.txt'
DE_TEST_PATH = r'.\data\test_de.txt'

BATCH_SIZE = 10
NUM_WORKERS = 1
SHUFFLE = True
PIN_MEMORY = True
BATCh_FIRST = False