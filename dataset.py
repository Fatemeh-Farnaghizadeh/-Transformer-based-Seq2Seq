#!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#!pip install torchtext==0.6.0
#!pip install spacy
#!python -m spacy download en
#!python -m spacy download de

import pandas as pd  
import spacy 
import torch
from torch.nn.utils.rnn import pad_sequence  
from torch.utils.data import DataLoader, Dataset

import utils


df_train_eng = pd.read_csv(utils.EN_TRAIN_PATH, delimiter='\t', header=None, names=['english'])

df_train_ger = pd.read_csv(utils.DE_TRAIN_PATH, delimiter='\t', header=None, names=['german'])

train_df = pd.merge(df_train_eng, df_train_ger, left_index=True, right_index=True)
train_df.columns = ['english', 'german']

spacy_eng = spacy.load("en_core_web_sm")
spacy_ger = spacy.load('de_core_news_sm')

class Vocabulary:

    def __init__(self, freq_threshold):
        self.itos_de = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi_de = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

        self.itos_en = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi_en = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos_en)

    @staticmethod
    def tokenizer_En(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    @staticmethod
    def tokenizer_De(text):
        return [tok.text.lower() for tok in spacy_ger.tokenizer(text)]

    def build_ger_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:

            for word in self.tokenizer_De(sentence):

                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi_de[word] = idx
                    self.itos_de[idx] = word
                    idx += 1

    def build_eng_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:

            for word in self.tokenizer_En(sentence):

                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi_en[word] = idx
                    self.itos_en[idx] = word
                    idx += 1

    def numericalize_De(self, text):
        tokenized_text = self.tokenizer_De(text)

        return [
            self.stoi_de[token] if token in self.stoi_de else self.stoi_de["<UNK>"]
            for token in tokenized_text
        ]
    
    def numericalize_En(self, text):
        tokenized_text = self.tokenizer_En(text)

        return [
            self.stoi_en[token] if token in self.stoi_en else self.stoi_en["<UNK>"]
            for token in tokenized_text
        ]
      
        
class CustomDataset(Dataset):
    
    def __init__(self, train_df, freq_threshold=3):
        self.df = train_df

        self.eng_texts = self.df["english"]
        self.ger_texts = self.df["german"]

        self.vocab = Vocabulary(freq_threshold)
        
        self.vocab.build_ger_vocabulary(self.ger_texts.tolist())
        self.vocab.build_eng_vocabulary(self.eng_texts.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        eng_text = self.df.loc[index, 'english']
        ger_text = self.df.loc[index, 'german']

        numericalized_eng = [self.vocab.stoi_en["<SOS>"]]
        numericalized_eng += self.vocab.numericalize_En(eng_text)
        numericalized_eng.append(self.vocab.stoi_en["<EOS>"])

        numericalized_ger = [self.vocab.stoi_de["<SOS>"]]
        numericalized_ger += self.vocab.numericalize_De(ger_text)
        numericalized_ger.append(self.vocab.stoi_de["<EOS>"])

        return torch.tensor(numericalized_ger), torch.tensor(numericalized_eng)


class MyCollate:
    def __init__(self, pad_idx_eng, pad_idx_ger):
        self.pad_idx_ger = pad_idx_ger
        self.pad_idx_eng = pad_idx_eng

    def __call__(self, batch):
        targets_eng = [item[1] for item in batch]
        targets_eng = pad_sequence(targets_eng, batch_first=utils.BATCh_FIRST, padding_value=self.pad_idx_eng) 

        targets_ger = [item[0] for item in batch]
        targets_ger = pad_sequence(targets_ger, batch_first=utils.BATCh_FIRST, padding_value=self.pad_idx_ger)

        return targets_ger, targets_eng


def get_loader(train_df, batch_size=utils.BATCH_SIZE, num_workers=utils.NUM_WORKERS, shuffle=utils.SHUFFLE, pin_memory=utils.PIN_MEMORY):
    dataset = CustomDataset(train_df)

    pad_idx_eng = dataset.vocab.stoi_en["<PAD>"]
    pad_idx_ger = dataset.vocab.stoi_de["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx_eng=pad_idx_eng, pad_idx_ger=pad_idx_ger)
    )

    return loader, dataset

if __name__ == "__main__":

    loader, dataset = get_loader(train_df)

    for idx, (english_vect, german_vect) in enumerate(loader):
        print('english_vect', english_vect)
        print('german_vect', german_vect)
