import torch
import utils
import dataset
import models
import translation
from torchtext.data.metrics import bleu_score

def main():
    df = dataset.train_df
    train_loader, data = dataset.get_loader(df, num_workers=1) 

    n_vocab_ger = len(data.vocab.itos_de)
    n_vocab_eng = len(data.vocab.itos_en)

    
    model = models.Seq2SeqTransformer(embedding_size=utils.EMBED_SIZE,
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

    utils.load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    model.eval()

    translator_obj = translation.Translator()

    data_iterator = iter(train_loader)
    batch = next(data_iterator)

    str_sources = []
    str_targets = []

    for i in range(utils.BATCH_SIZE):
        x_int = batch[0][:, i]
        y_int = batch[1][:, i]

        x_str = [data.vocab.itos_de[i] for i in x_int.tolist()]
        y_str = [data.vocab.itos_en[j] for j in y_int.tolist()]

        str_sources.append(x_str)
        str_targets.append(y_str)

    translated_all = []

    for j in range(len(str_sources)):
        translated = translator_obj.translate_sentence(model, str_sources[j], utils.DEVICE, max_length=50, eval=True)
        translated_all.append(translated)
    
    str_targets = [seq[1:] for seq in str_targets]

    return(bleu_score(translated_all, str_targets))

if __name__ == '__main__':
    score = main()
    print('score', score)