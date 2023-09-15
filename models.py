# import utils
# import dataset

# import torch
# from torch import nn

       
# class Seq2SeqTransformer(nn.Module):

#     def __init__(
#         self,
#         embedding_size,
#         src_vocab_size,
#         trg_vocab_size,
#         src_pad_idx,
#         num_heads,
#         num_encoder_layers,
#         num_decoder_layers,
#         forward_expansion,
#         dropout,
#         max_len,
#         device,
#     ):
#         super(Seq2SeqTransformer, self).__init__()
        
#         self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
#         self.src_position_embedding = nn.Embedding(max_len, embedding_size)
#         self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
#         self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

#         self.device = device
#         self.transformer = nn.Transformer(
#             embedding_size,
#             num_heads,
#             num_encoder_layers,
#             num_decoder_layers,
#             forward_expansion,
#             dropout,
#         )
#         self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
#         self.dropout = nn.Dropout(dropout)
#         self.src_pad_idx = src_pad_idx

#     def make_src_mask(self, src):
#         src_mask = src.transpose(0, 1) == self.src_pad_idx

#         # (N, src_len)
#         return src_mask.to(self.device)

#     def forward(self, src, trg):
#         src_seq_length, N = src.shape
#         trg_seq_length, N = trg.shape

#         src_positions = (
#             torch.arange(0, src_seq_length)
#             .unsqueeze(1)
#             .expand(src_seq_length, N)
#             .to(self.device)
#         )

#         trg_positions = (
#             torch.arange(0, trg_seq_length)
#             .unsqueeze(1)
#             .expand(trg_seq_length, N)
#             .to(self.device)
#         )

#         embed_src = self.dropout(
#             (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
#         )
#         embed_trg = self.dropout(
#             (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
#         )

#         src_padding_mask = self.make_src_mask(src)
#         trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
#             self.device
#         )

#         out = self.transformer(
#             embed_src,
#             embed_trg,
#             src_key_padding_mask=src_padding_mask,
#             tgt_mask=trg_mask,
#         )

#         out = self.fc_out(out)

#         return out


# if __name__ == '__main__':

#     seq_model = Seq2SeqTransformer(embedding_size=10,
#         src_vocab_size=30,
#         trg_vocab_size=40,
#         src_pad_idx=0,
#         num_heads=2,
#         num_encoder_layers=5,
#         num_decoder_layers=5,
#         forward_expansion=2,
#         dropout=0.2,
#         max_len=10,
#         device=utils.DEVICE).to(utils.DEVICE)

#     #batch = 2
#     #seq_length = 8
#     #vocab_size = 30
#     source = torch.randint(0, 30, size=(8, 2)).to(utils.DEVICE)

#     #batch = 2
#     #seq_length = 6
#     #vocab_size = 40
#     target =  torch.randint(0, 40, size=(6, 2)).to(utils.DEVICE)

#     outputs = seq_model(source, target)

#     print(outputs.shape)   
import utils
import dataset
import torch
from torch import nn
import math

class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Seq2SeqTransformer, self).__init__()

        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

        # Positional encoding parameters
        self.max_len = max_len
        self.positional_encoding = self.create_positional_encoding(max_len, embedding_size)

    def create_positional_encoding(self, max_len, embedding_size):
        positional_encoding = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding.to(self.device)

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        # Calculate positional encodings for both source and target sequences
        src_positions = self.positional_encoding[:src_seq_length, :]
        trg_positions = self.positional_encoding[:trg_seq_length, :]

        embed_src = self.dropout(
            (self.src_word_embedding(src) + src_positions.unsqueeze(1))  # Add an extra dimension for batch
        )
        
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + trg_positions.unsqueeze(1))  # Add an extra dimension for batch
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )

        out = self.fc_out(out)

        return out

if __name__ == '__main__':
    seq_model = Seq2SeqTransformer(
        embedding_size=10,
        src_vocab_size=30,
        trg_vocab_size=40,
        src_pad_idx=0,
        num_heads=2,
        num_encoder_layers=5,
        num_decoder_layers=5,
        forward_expansion=2,
        dropout=0.2,
        max_len=10,
        device=utils.DEVICE
    ).to(utils.DEVICE)

    source = torch.randint(0, 30, size=(8, 2)).to(utils.DEVICE)
    target = torch.randint(0, 40, size=(6, 2)).to(utils.DEVICE)

    outputs = seq_model(source, target)

    print(outputs.shape)