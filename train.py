import dataset
import utils
import models

import torch
from torch import optim


if __name__ == '__main__':
    df = dataset.train_df
    train_loader, data = dataset.get_loader(df, num_workers=1) 
    
    utils.save_dic(utils.STOI_GER_PATH, data.vocab.stoi_de)
    utils.save_dic(utils.ITOS_GER_PATH, data.vocab.itos_de)

    utils.save_dic(utils.STOI_ENG_PATH, data.vocab.stoi_en)
    utils.save_dic(utils.ITOS_ENG_PATH, data.vocab.itos_en)

    n_vocab_ger = len(data.vocab.itos_de)
    n_vocab_eng = len(data.vocab.itos_en)

    seq_model = models.Seq2SeqTransformer(embedding_size=utils.EMBED_SIZE,
        src_vocab_size=n_vocab_ger,
        trg_vocab_size=n_vocab_eng,
        src_pad_idx=data.vocab.stoi_de['<SOS>'],
        num_heads=utils.NUM_HEADS,
        num_encoder_layers=utils.NUM_LAYERS_ENCODER,
        num_decoder_layers=utils.NUM_LAYERS_DECODER,
        forward_expansion=utils.FORWARD_EXP,
        dropout=utils.DROP_P,
        max_len=utils.MAX_LEN,
        device=utils.DEVICE).to(utils.DEVICE)
    
    optimizer = optim.Adam(seq_model.parameters(), lr=utils.LR)
    criterion = utils.CRITERION

    for epoch in range(utils.EPOCHS):

        checkpoint = {"state_dict": seq_model.state_dict()}
        utils.save_checkpoint(checkpoint)

        for source_batch, target_batch in train_loader:
            source_batch = source_batch.to(utils.DEVICE)
            target_batch = target_batch.to(utils.DEVICE)

            outputs = seq_model(source_batch, target_batch[:-1, :]).to(utils.DEVICE)
           
            outputs = outputs.reshape(-1, outputs.shape[2])
            target_batch = target_batch[1:].reshape(-1)

            optimizer.zero_grad()

            loss = criterion(outputs, target_batch)

            # Back prop
            loss.backward()
            
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(seq_model.parameters(), max_norm=1)

            # Gradient descent step
            optimizer.step()
        
        print(f"epoch {epoch}: {loss}")

