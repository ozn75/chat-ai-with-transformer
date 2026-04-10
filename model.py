from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import TensorDataset, DataLoader

@dataclass
class TransformerConfig:
    pad_id: int
    max_len: int
    vocab_size: int

    d_model: int= 512
    dropout: float = 0.1
    d_ff: int = d_model * 4
    num_head: int  = 8
    head_dim = d_model // num_head
    num_layers: int = 8




class MHSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig): 
        super().__init__()
        
        self.config = config

        self.qkv_proj = nn.Linear(self.config.d_model,3 * self.config.d_model)
        
        self.out_line = nn.Linear(self.config.d_model, self.config.d_model)

        self.drop1 = nn.Dropout(self.config.dropout)
        self.residdrop = nn.Dropout(self.config.dropout)

        casual = torch.tril(torch.ones(self.config.max_len,self.config.max_len))  # (T,T)

        self.register_buffer(
            "casualmask",
            casual.unsqueeze(0).unsqueeze(0),  # 1,1,T,T,
            persistent=True                 # modelle birlikte kaydet
        )

    def forward(self,x,padmask: bool = None):
        B,T,D = x.shape

        Q,K,V = torch.split(self.qkv_proj(x),self.config.d_model,dim=-1)   # multihead 3 böldüm

        # heads için reshape

        Q = Q.view(B,T,self.config.num_head,self.config.head_dim).transpose(1,2)  #(B,num_head,T,head_dim
        K = K.view(B,T,self.config.num_head,self.config.head_dim).transpose(1,2)
        V = V.view(B,T,self.config.num_head,self.config.head_dim).transpose(1,2)

        #print("Q shape:", Q.shape)

        attn_weights = torch.matmul(Q,K.transpose(-2,-1)) / (math.sqrt(self.config.d_model))    #QK^T / sqrt(d_model)

        #====================CASUAL MASK ============================

        casual_mask = self.casualmask[:,:,:T,:T]  #MASKENİN T DEN T YE KADAR OLAN KISMI 
        #print("casual mask shape:", casual_mask.shape)
        attn_weights = attn_weights.masked_fill(casual_mask == 0 ,float("-inf"))

        if padmask is not None:
            attn_weights = attn_weights.masked_fill(padmask == self.config.pad_id ,float("-inf"))


        attn_weights = F.softmax(attn_weights,dim=-1)   # hepsini 1 ile 0 arasına sıkıştır

        attn_weights = self.drop1(attn_weights)

        out = torch.matmul(attn_weights,V)  # (B,num_head,T,head_dim)

        #print("out shape:", out.shape)

        out = out.transpose(1,2).contiguous().view(B,T,self.config.d_model)  # (B,T,d_model))

        out = self.out_line(out)

        out = self.residdrop(out)

        return out


class FFN(nn.Module):
    def __init__(self,config: TransformerConfig):
        super().__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model,config.d_ff),
            nn.ReLU(),
            nn.Linear(config.d_ff,config.d_model),
            nn.Dropout(config.dropout)
            )

    def forward(self,x):
        return self.ffn(x)



class EncoderBlock(nn.Module):
    def __init__(self,config: TransformerConfig):
        super().__init__()

        self.config = config

        self.att = MHSelfAttention(self.config)

        self.ffn = FFN(self.config)

        self.norm1 = nn.LayerNorm(self.config.d_model)
        self.norm2 = nn.LayerNorm(self.config.d_model)

        self.drop = nn.Dropout(self.config.dropout)

    def forward(self,x,padmask: bool = None):
        attn_weights = self.att(self.norm1(x),padmask)

        x = x + self.drop(attn_weights)

        ffn_out = self.ffn(self.norm2(x))

        x = x + self.drop(ffn_out)

        return x


class Encoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([
            EncoderBlock(self.config) for _ in range(self.config.num_layers)
            ])

        self.final_ln = nn.Linear(self.config.d_model,self.config.d_model)

    def forward(self,x,padmask: bool = None):

        for layer in self.layers:
            x = layer(x,padmask)

        x = self.final_ln(x)

        return x



"""
====================================
POSENC HİÇ BAKMADIM
====================================
"""
class PositionalEncoding(nn.Module):
    def __init__(self,config : TransformerConfig):
        super().__init__()

        self.d_model = config.d_model
        self.max_len = config.max_len

        pe = torch.zeros(self.max_len, self.d_model)  # (max_len, d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, d_model)
        T = x.size(1)
        #print(self.pe[:, :T].shape)
        #print("posenc öncesi x shape:", x.shape)
        x = x + self.pe[:, :T]
        #print("posenc sonrası x shape:", x.shape)
        return x



class Mousemodel(nn.Module):
    def __init__(self,config: TransformerConfig):
        super().__init__()
        self.config = config

        self.embeder = nn.Embedding(config.vocab_size, config.d_model)  # kelime sayısı kadar embedding, d_model boyutunda)

        self.posenc = PositionalEncoding(config)

        self.encoder = Encoder(config)

        self.drop = nn.Dropout(config.dropout)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size)  # dil modelleme kafası, d_model boyutunu tekrar kelime sayısına çevirir

    
    def make_pad_mask(self, input_ids):
        # input_ids: (B, T)
        mask = (input_ids != self.config.pad_id)          # (B, T)
        mask = mask.unsqueeze(1).unsqueeze(2)      # (B, 1, 1, T)
        return mask


    def forward(self,x,padmask: bool = None):
        
        mask = self.make_pad_mask(x)  # (B, 1, 1, T)

        #print("düz x: ",x.shape)

        x_emb = self.embeder(x)  # (B,T,d_model)

        x = self.posenc(x_emb)

        #print("posenc sonrası x shape:", x.shape)

        x = self.drop(x)

        x = self.encoder(x,mask)

        #print("encoder sonrası x shape:", x.shape)

        x = self.lm_head(x)  # (B,T,vocab_size)

        return x



class Train():
    def train(self,
        model,
        device,
        train_loader,
        epochs=100,
        ):

        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
        """
        CosineAnnealingLR (Scheduler): Öğrenme oranı (lr - Learning Rate) modelin her
        adımda ne kadar büyük bir adım atacağını belirler. Eğitim başlarken büyük
        adımlar atmak (hızlı öğrenmek), sona yaklaştıkça ise ince ayar yapmak için
        çok minik adımlar atmak isteriz. Bu "Scheduler", öğrenme oranını bir kosinüs dalgası gibi yavaşça düşürür
        """
        
        for epoch in range(epochs):

            model.train()

            total_loss = 0

            for batch in train_loader:
                x,y = batch


                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()    # ağırlıkları temizle

                logits = model(x)

                B,T,V = logits.shape
                #print("train içinde logits shape:", logits.shape)
                #print("train içinde y shape:", y.shape)

                loss = F.cross_entropy(
                    logits.view(B*T,V),
                    y.view(B*T),
                    ignore_index=0
                    )

                loss.backward()          # geri yayılım   hatayı hesapla

                # gradient clipping (çok kritik) 1.0 dan fazla oynatma

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()         # ağırlıkları güncelle

                total_loss += loss.item()

            scheduler.step()           # öğrenme oranını güncelle 

            avg_loss = total_loss / len(train_loader)

            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")



class BasicQuestion():
    def topk(logits,k):
        values,indices = torch.topk(logits,k,dim=-1)                    #listedeki en büyük $k$ tane değeri (values) ve bu değerlerin hangi kelimelere ait olduğunu (indices) bulur
        
        probs = torch.zeros_like(logits)
        
        probs.scatter_(1,indices,values)  # probs matrisine indices konumlarına values değerlerini yerleştirir, diğer konumlar sıfır kalır)
        
        probs = F.softmax(probs,dim=-1)  # en büyük $k$ değerler arasında olasılık dağılımı oluşturur
       
        return probs

    def generate(model,
        input_ids,
        device,
        eos_id,
        top_k = 10,
        temperature = 1.5,
        max_new_token=40):

        model.eval()
        """
            Eğitimi bitirdik, model artık üretim (inference) yapacak. Bu kod, içerideki
            tüm Dropout (sabotajcı) katmanlarını kapatır. Modelin kafasını karıştırmayı
            bırakıp tüm odaklanma gücüyle (tam kapasite) çalışmasını sağlarız.
        """
        input_ids = input_ids.to(device)

        with torch.no_grad():
            for _ in range(max_new_token):

                #print("buraya kadar çalıştı1")

                logits = model(input_ids)

                #print("buraya kadar çalıştı2")

                last_logits = logits[:,-1,:]

                last_logits = last_logits / temperature

                #print("buraya kadar çalıştı3")

                """  temperature
                    Modelin ürettiği ham puanları (logits) bir ısı ayarına böleriz.Sıcaklık
                    Düşükse ($T \approx 0.1$): Puanlar arasındaki fark açılır. Model çok
                    "garantici" olur, hep en emin olduğu, en standart kelimeleri seçer (Sıkıcı
                    ama doğru cevaplar).Sıcaklık Yüksekse ($T \approx 1.5$): Puanlar birbirine
                    yaklaşır. Model "çılgınlaşır", normalde hiç kurmayacağı yaratıcı veya
                    tuhaf cümleler kurmaya başlar.Normal ($T = 1.0$): Modelin orijinal hali bozulmaz.
                """

                probs = BasicQuestion.topk(last_logits, top_k)

                """
                Eğer sözlüğünde 30.000 kelime varsa ve sen sıradaki kelimeyi rastgele
                seçmeye kalkarsan, model "bugün hava çok... pencere" gibi saçma sapan kelimeler seçebilir.
                top_k=10 dediğimizde modele şunu diyoruz: "Bana olasılığı en yüksek olan sadece
                en iyi 10 kelimeyi bırak, geri kalan 29.990 kelimenin ihtimalini direkt SIFIR yap, çöpe at!"
                Böylece model sadece mantıklı seçenekler arasında gidip gelir.
                """

                next_token = torch.multinomial(probs,1)
                """
                Bu komut, PyTorch'un "ağırlıklı zar atma" fonksiyonudur.
                En iyi 10 kelimeyi (Top-K) aldık diyelim. Eğer argmax kullansaydık model her
                seferinde %100 oranla birinci olan kelimeyi seçerdi ve hep aynı robotik cümleleri kurardı.
                multinomial ise olasılıklara göre rastgelelik katar. "Güzel" kelimesinin %70,
                "Güneşli" kelimesinin %30 ihtimali varsa, zarı atar ve bazen %30'luk kelimeyi
                seçerek cümleye doğallık/farklılık kazandırır. 1 rakamı ise "sadece 1 kelime seç" demektir.
                """
                if next_token.item() == eos_id:
                    break

                input_ids = torch.cat([input_ids,next_token],dim=1)

            return input_ids

    def translate2id(prompt,word2id,bos_id,unk_id):
        input_ids = ([bos_id])
        words = prompt.lower().strip().split()
        #print(words)
        for word in words:
            id = (word2id.get(word,unk_id))
            input_ids.append(id)

        input_ids = torch.tensor(input_ids)
        #print("input shape ",input_ids.shape)
        return input_ids

    def translate2word(ids,id2word,eos_id):
        words = []
        ids = ids.squeeze().tolist()  # (T,) tensörünü listeye çevirir
        #print("ids listesi:", ids)
        for i in ids:
            #print("id:", i)
            if i == eos_id:
              break
            kelime = id2word.get(i,"-")
            words.append(kelime)

        return words

    def settrainloader(self,x,y):
        #from torch.utils.data import TensorDataset, DataLoader

        #print("dataloaderx shape:", x.shape)
        dataset = TensorDataset(x, y)
        #print("dataset oluşturuldu, örnek veri: ", len(x))
        batch_size = 16
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return train_loader
