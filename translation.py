import torch
import utils
import dataset
import models


class Translator():

    def __init__(self):
        self.stoi_ger = utils.load_json(utils.STOI_GER_PATH)
        self.itos_ger = utils.load_json(utils.ITOS_GER_PATH)

        self.stoi_eng = utils.load_json(utils.STOI_ENG_PATH)
        self.itos_eng = utils.load_json(utils.ITOS_ENG_PATH)

        self.n_vocab_ger = len(self.stoi_ger)
        self.n_vocab_eng = len(self.stoi_eng)

        self.vocab_object = dataset.Vocabulary(0.5)


    def translate_sentence(self, model, sentence, device, max_length=50, eval=False):

        if eval == False:
            tokenized_sentence = self.vocab_object.tokenizer_De(sentence)

            tokenized_sentence.insert(0, self.itos_ger['0'])
            tokenized_sentence.append(self.itos_ger['2'])
        
        else:
            tokenized_sentence = sentence

        text_to_indices = [self.stoi_ger[token] for token in tokenized_sentence]

        sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)


        
        outputs = [self.stoi_eng["<SOS>"]]

        for _ in range(max_length):
            trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

            with torch.no_grad():
                output = model(sentence_tensor, trg_tensor)

            best_guess = output.argmax(2)[-1, :].item()
            outputs.append(best_guess)

            if best_guess == self.stoi_eng["<EOS>"]:
                break

        translated_sentence = [self.itos_eng[str(idx)] for idx in outputs]


        return translated_sentence[1:]
        

if __name__=='__main__':
    translator = Translator()

    sentence = "Ein Hund sitzt auf einem Felsen"

    
    model = models.Seq2SeqTransformer(embedding_size=utils.EMBED_SIZE,
        src_vocab_size=translator.n_vocab_ger,
        trg_vocab_size=translator.n_vocab_eng,
        src_pad_idx=translator.stoi_ger['<SOS>'],
        num_heads=utils.NUM_HEADS,
        num_encoder_layers=utils.NUM_LAYERS_ENCODER,
        num_decoder_layers=utils.NUM_LAYERS_DECODER,
        forward_expansion=utils.FORWARD_EXP,
        dropout=utils.DROP_P,
        max_len=utils.MAX_LEN,
        device=utils.DEVICE).to(utils.DEVICE)
    
    utils.load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    model.eval()

    translated = translator.translate_sentence(model, sentence, utils.DEVICE, max_length=50)

    print(translated)


    



