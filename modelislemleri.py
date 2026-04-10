from verihazirlik import Verihazirlik
from model import Mousemodel ,TransformerConfig, Train , BasicQuestion
import torch



class Modelislemleri():
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Kullanılan device:", self.device)
        print("CUDA mevcut mu:", torch.cuda.is_available())
        print("GPU adı:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Yok")

        self.pad_id = 0

        #==================VERİ HAZIRLIK ======================
        self.verihazirlik = Verihazirlik()
        self.sentences = self.verihazirlik.dosyaokuma()
        #print(self.sentences[0:16])
        self.input_tensor,self.word2id,self.id2word = self.verihazirlik.tokenize_and_pad(self.sentences)
        self.vocab_size = self.verihazirlik.getvocabsize()
        #print("vocab size : ", self.vocab_size)
        self.max_len = self.input_tensor.shape[1]
        self.x,self.y = self.verihazirlik.makex_y(self.input_tensor)

        #==================DATALOADER İŞLEMLERİ ======================
        self.query = BasicQuestion()
        self.train_load = self.query.settrainloader(self.x,self.y)


        

        #==================MODEL İŞLEMLERİ ======================
        self.config = TransformerConfig(pad_id=0, max_len=self.max_len, vocab_size=self.vocab_size)
        #print(self.config.vocab_size,self.config.max_len,self.config.pad_id,self.vocab_size)
        self.mousemodel = Mousemodel(self.config)

        #print("vocab_size:", self.config.vocab_size)
        #print("d_model:", self.config.d_model)
        #print("lm_head:", self.mousemodel.lm_head)

        self.trainer = Train()
        
    def modeltrain(self):
        self.trainer.train(model = self.mousemodel,device = self.device,train_loader=self.train_load,epochs=50)
        
    def model_save(self):
        import os
        import pickle

        os.makedirs("saved_models", exist_ok=True)
        torch.save(self.mousemodel.state_dict(),"saved_models/model2.pt")

        vocab = {
            "word2id": self.word2id,
            "id2word": self.id2word,
            "max_len": self.max_len,
            "pad_id": 0,
            "vocab_size": self.vocab_size
            }

        with open("saved_models/vocab2.pkl","wb") as f:       #wb write binary
            pickle.dump(vocab,f)





if __name__ == "__main__":
    modelislemleri = Modelislemleri()
    modelislemleri.modeltrain()
    modelislemleri.model_save()