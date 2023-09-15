import torch 
from torch import nn
import json


#Dataset Settings
EN_TRAIN_PATH = r'.\data\train_en.txt'
DE_TRAIN_PATH = r'.\data\train_de.txt'

EN_VAL_PATH = r'.\data\val_en.txt'
DE_VAL_PATH = r'.\data\val_de.txt'

EN_TEST_PATH = r'.\data\test_en.txt'
DE_TEST_PATH = r'.\data\test_de.txt'

BATCH_SIZE = 32
NUM_WORKERS = 1
SHUFFLE = True
PIN_MEMORY = True
BATCh_FIRST = False

#Train_Settings
EPOCHS = 5
EMBED_SIZE = 512 
NUM_HEADS = 8
NUM_LAYERS_ENCODER = 3
NUM_LAYERS_DECODER = 3
MAX_LEN = 100
FORWARD_EXP = 4
DROP_P = 0.1
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LR = 3e-4
CRITERION = nn.CrossEntropyLoss()


STOI_GER_PATH = "stoi_ger.json"
ITOS_GER_PATH = "itos_ger.json"

STOI_ENG_PATH = "stoi_eng.json"
ITOS_ENG_PATH = "itos_eng.json"


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def save_dic(file_path, dict):

    with open(file_path, 'w') as file:
        json.dump(dict, file)



def load_json(file_path):

    with open(file_path, 'r') as file:
        dict = json.load(file)
        
    return dict
