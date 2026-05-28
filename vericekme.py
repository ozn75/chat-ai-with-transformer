import wikipediaapi
import asyncio
import os
import time

class searchwiki():
    def dosyaya_yazdir(self,bolum, dosya_adi, anakonu=""):
        # Eğer bölümün içinde hiç metin yoksa (sadece alt başlık barındıran boş bir kapsayıcıysa) dosyaya yazma
        if bolum.text.strip():
            with open(dosya_adi, 'a', encoding='utf-8') as dosya:
                dosya.write(f"[p]\n")
                dosya.write(f"Ana Konu : {anakonu}\n\n")
                dosya.write(f"Başlık : {bolum.title}\n\n")
                # [0:] yazmana gerek yok, bolum.text zaten metnin tamamını temsil eder
                dosya.write(f"İçerik : {bolum.text}\n") 
                dosya.write(f"[/p]\n\n")
            with open("datalar/finished_topics.txt","r",encoding="utf-8") as f:
                mevcut_basliklar = [line.lower().strip() for line in f.readlines()]
            if anakonu.lower().strip() not in mevcut_basliklar:
                with open("datalar/finished_topics.txt","a",encoding="utf-8") as f:
                    f.write(anakonu + "\n")
                    print(f"'{anakonu}' başlığı finished_topics.txt dosyasına eklendi.")

    # ÖNEMLİ: Tüm alt başlıkları da tarayan recursive yazdırma döngüsü
    def butun_basliklari_yazdir(self,sections, dosya_adi, anakonu):
        for bolum in sections:
            # 1. Önce bulunduğumuz ana bölümü dosyaya yazdırıyoruz
            self.dosyaya_yazdir(bolum, dosya_adi, anakonu)
        
            # 2. SİHİRLİ NOKTA: Eğer bu bölümün alt başlıkları varsa, fonksiyonu onlar için tekrar çalıştır!
            if bolum.sections:
                self.butun_basliklari_yazdir(bolum.sections, dosya_adi, anakonu)

    def baslik_varmi(self,baslik: str):
        """
        baslik varmı kontrol eder yoksa ekler topics
        """
        baslik = baslik.lower().strip()
        path = "datalar/topics.txt"
        with open(path,"r",encoding="utf-8") as f:
            mevcut_basliklar = [line.strip().lower() for line in f.readlines()]

        if baslik not in mevcut_basliklar:
            with open(path,"a",encoding="utf-8") as f:
                f.write(baslik + "\n")
                #print(f"'{baslik}' başlığı topics.txt dosyasına eklendi.")
        else:
            pass
            #print(f"'{baslik}' başlığı zaten topics.txt dosyasında mevcut.")



    async def main(self,konu):
        # Türkçe Wikipedia ve senin e-postan ayarlandı
        wiki = wikipediaapi.AsyncWikipedia(user_agent='TransformerEgitimVerisi (ozankiraz-123@hotmail.com)', language='tr')
    
        # Doğru sayfa adı belirtildi
        page = wiki.page(konu)

        #print("Sayfa aranıyor...")
        #print(page._fetch.__getstate__())  # Sayfa durumunu kontrol etmek için

        if await page.exists():
            print(f"'{page.title}' sayfası bulundu! Veriler çekiliyor ve kaydediliyor...")
        
            sections = await page.sections
            
            # Sadece ana başlıkları değil, tüm alt başlıkları tarayıp dosyaya yazacak fonksiyonu başlat
            self.butun_basliklari_yazdir(sections, "datalar/aiveri2.txt", anakonu=page.title)
        
            #print("İşlem tamam! 'datalar/aiveri.txt' dosyasını kontrol edebilirsin.")

            for section in sections:
                if section.title.strip().lower() == "Ayrıca bakınız".lower():
                    icerik = section.text.strip().lower()
                    for topic in icerik.split("\n"):
                        if not topic.strip():
                            continue
                        if topic == "== kaynakça ==":
                            continue
                        self.baslik_varmi(topic)
                        #print("basliklar:============== ",topic)
        else:
            print("HATA: Sayfa bulunamadı!")

    def run(self,konu):
        with open("datalar/finished_topics.txt","r",encoding="utf-8") as f:
            mevcut_basliklar = [line.lower().strip() for line in f.readlines()]

        if konu.lower().strip() not in mevcut_basliklar:
            print(f"'{konu}' başlığı finished_topics.txt dosyasında mevcut değil işleniyor")
            asyncio.run(self.main(konu=konu))
        else:
            # Sistemi çalıştır
            print(f"'{konu}' başlığı zaten finished_topics.txt dosyasında mevcut. İşlem atlandı.")

                


    def yeni_konu_varmi(self)->bool:
        with open("datalar/topics.txt","r",encoding="utf-8") as f:
            mevcut_basliklar = [line.lower().strip() for line in f.readlines()]
            
        with open("datalar/finished_topics.txt","r",encoding="utf-8") as f1:
            biten_basliklar = [line.lower().strip() for line in f1.readlines()]

        for baslik in mevcut_basliklar:
            if baslik not in biten_basliklar:
                return True  # Yeni bir konu var
            else:
                continue

        return False



wiki = searchwiki()

#dur = True
sayac = 1

while sayac <= 100:
    
    with open("datalar/topics.txt","r",encoding="utf-8") as f:
        lines = [line.strip().lower() for line in f.readlines()]
        for line in lines:
            print(f"'{line}' başlığı işleniyor...")
            time.sleep(2)  # Her döngü arasında 025 saniye bekle
            if line:
                wiki.run(konu=line)
            else:
                continue
            print("SAYAÇ ==========",sayac)
            sayac += 1
            if sayac > 100:
                print("100 başlık işlendi, döngü sonlandırılıyor.")
                break
                

    #dur = wiki.yeni_konu_varmi()
    
    print("Tüm konular işlendi!")