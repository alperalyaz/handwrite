# handwrite

🇬🇧 [English](README.md) · 🇹🇷 Türkçe

El yazısından OpenType font üretir. Form yok, kutu yok, şablon yok:
**elinizdeki herhangi bir el yazısı sayfasının fotoğrafını verin, yeter.**

```bash
handwrite serve
```

Tarayıcı kendiliğinden açılır. Gerisi ekranda: fotoğrafı sürükleyin ya da
kamerayla çekin, fontu indirin. Elinizde yazılı bir kağıt yoksa arayüz
kopyalamanız için bütün karakterleri kapsayan bir metin verir.

Ölçülen: tek bir defter sayfası (14 satır, ~500 karakter) ~30 saniyede 87
karakterlik tam bir fonta dönüşüyor.

Komut satırını tercih edenler için aynı işi yapan komutlar:

```bash
export GOOGLE_AI_API_KEY=...
handwrite read defterim.jpg -o Benim.ttf --preview   # şablonsuz, model okur

handwrite sheets -o calisma                          # basılı çalışma sayfası
handwrite build calisma/sheets.json foto*.jpg -o Benim.ttf
```

---

## Yapay zekâ tam olarak nerede

İş bölümü bilinçli ve tek cümleyle özetlenebilir:

> **Model *ne* yazdığını söyler, dinamik programlama *nerede* olduğunu bulur.**

| iş | kim yapar | neden |
|---|---|---|
| Sayfada ne yazdığını okumak | Gemini | Anlamsal iş; bağlamdan tamamlıyor, Türkçeyi biliyor |
| Harfin pikselini bulmak | DP | Görsel modellerin sınır kutuları güvenilmez; font em biriminde çalışır ve "yaklaşık" bir kutu glifin ayağını keser |
| Bozuk glifi ayıklamak | kümeleme | Ölçülebilir ve ucuz |
| Olmayan harfi üretmek | 3 kademeli sentez | Aşağıda |

Modele piksel koordinatı sordurmak cazip ama yanlış olurdu. Ölçüm: Gemini 2.5
Flash sentetik Türkçe el yazısını **CER 0,0038** ile okuyor (6 satırın 4'ü
birebir doğru). Kalan hata tek karakterlik ekleme/düşürme.

### Modelin hatasına dayanıklılık

Okuma mükemmel değil, olması da gerekmiyor. Segmentasyon iki kademeli olduğu
için — önce kelimeler, sonra kelime içi karakterler — okunan metindeki tek
harflik bir hata yalnız **kendi kelimesini** etkiler; satırın geri kalanı kendi
aralıklarına oturmayı sürdürür. Tek kademeli bir hizalamada aynı hata satırdaki
bütün glifleri kaydırırdı.

Bozulan o birkaç glif de fonta giremez: kendi harflerinin örnek kümesinde aykırı
kalıp eleme aşamasında düşerler.

## Eksik karakterler: tek sayfa yetmez, biz tamamlarız

Doğal bir metin küçük harfleri bol verir ama düzyazıda "Q" da geçmez, "%" de.
Ölçtüğümüz gerçek dağılım (500 karakterlik bir sayfa):

| grup | durum |
|---|---|
| küçük harfler | bol bol var (`a` ~155, `e` ~115 örnek); tek gerçek eksik `j` |
| büyük harfler | çoğu yok — düzyazıda sadece cümle başları |
| rakam, noktalama | metinde geçtiği kadar |
| `q w x` | yok |

**86 karakterin ~44'ü tek sayfadan çıkmıyor.** Bu boşluk üç kademede, güvenilirlik
sırasıyla doldurulur:

1. **Bileşim** — eksik karakter kullanıcının *gerçek* kalem izlerinden monte
   edilir: `Ç` = `C` + `ç`'nin çengeli, `Ğ` = `G` + `ğ`'nin şapkası,
   `İ` = `I` + `i`'nin noktası. Her parça o elden çıktığı için sonuç kusursuza
   yakın.
2. **Büyük/küçük harf aktarımı** — bazı harflerin büyüğü küçüğünün büyütülmüşüdür
   (`c/C`, `o/O`, `s/S`, `v/V`...). Kullanıcının kendi harfi cap-height'a
   ölçeklenir, kalem kalınlığı geri çekilir. Biçimi gerçekten değişenler
   (`a/A`, `e/E`, `g/G`) bu listede yoktur.
3. **Referans biçim + kullanıcının kalemi** — geri kalan için bir referans yazı
   tipinin harf biçimi alınır; ölçülen kalem kalınlığı, oranlar, köşe
   yuvarlaklığı ve el titremesi uygulanır. Genişlikler kullanıcının kendi
   harfleriyle kalibre edilir (ölçülen düzeltme ~0,89: matbaa harfi el
   yazısından geniştir).

Üretilen her karakter teşhis ekranında açıkça bildirilir. Kullanıcıya
üretilmiş bir harfi kendi yazısıymış gibi göstermek doğru olmaz.

Doğrudan görsel üretim (bir resim modeline "bu elle Q çiz" demek) bilerek
kullanılmadı: bugünkü modeller "el yazısı gibi" şeyler üretiyor ama harfin
iskeleti yanlış çıkıyor, kalem kalınlığı tutmuyor ve belirli bir eli taklit
edemiyorlar. Font glifi, resimden çok daha az hata affeder.

---

## Neden kutu yok

Calligraphr'ın kutuları keyfî bir tasarım tercihi değil, gerçek bir problemin
kaçamağı: **bir karalamanın hangi harf olduğunu bilmek zor.** Kutu, o soruyu
kullanıcıya yaptırır — kutunun yeri etiketin ta kendisidir.

handwrite o soruyu hiç sormaz. Kullanıcı, sistemin *önceden bildiği* bir metni
yazar. Böylece problem "tanıma"dan "hizalama"ya iner:

> Elimizde N karakterlik bir metin ve o metnin yazılmış hali var. Mürekkebi
> soldan sağa tam olarak N parçaya nereden bölmeliyiz?

Bu, çözülmüş bir problem türüdür: kısıtlı en iyi bölme. Yapay zekâ modeli,
eğitim verisi ya da GPU gerekmez — dinamik programlama yeter.

## Boru hattı

```
fotoğraf → sayfa kaydı → mürekkep ayırma → satır → eğim düzeltme
        → karakter segmentasyonu → glif normalizasyonu → varyant seçimi
        → vektörleştirme → TTF
```

Her adım kendi modülünde ve tek başına test edilebilir.

### Sayfa kaydı (`template.py`, `preprocess.py`)

Çalışma sayfasının dört köşesinde ArUco işaretleri vardır. Bunlar üç şeyi
birden çözer:

- **Perspektif.** Telefonla eğik çekilmiş fotoğraf, homografi ile kanonik sayfa
  koordinatlarına oturtulur. Eğiklik tahmin edilmez, çözülür.
- **Satır–metin eşleşmesi.** Hangi bandın hangi metne karşılık geldiği kesin
  bilinir; "bu satır hangi cümleydi?" diye tahmin yürütmek gerekmez.
- **Sayfa kimliği.** Her sayfa farklı işaret kimlikleri taşır, bu yüzden
  fotoğrafları herhangi bir sırada, adlandırmadan yükleyebilirsiniz.

### Mürekkep ayırma

Basılı kılavuz çizgilerini ve örnek metni elemek için tek bir kural kullanılır:

> Bir piksel, şablonun orada olmasını söylediği tondan belirgin şekilde koyuysa
> el yazısıdır.

Şablonu biz ürettiğimiz için her pikselin nominal gri değerini biliriz. Basım ve
tarama zincirinin bu değerleri nasıl kaydırdığı (`gözlenen ≈ a·nominal + b`)
sayfanın kendisinden ölçülür — farklı yazıcı, kağıt ve ışıkta kendiliğinden
kalibre olur.

Bu kuralın önemi: kullanıcı taban çizgisinin üstünden geçtiğinde kalem izi
basılı çizgiden koyu olduğu için harf korunur. Kaba dikdörtgen maskeleme bunu
yapamaz — ya çizgiyi bırakır ya harfi keser.

### Karakter segmentasyonu (`segment.py`) — çekirdek

İki maliyet dengelenir:

**Kesim maliyeti.** Kesim düz bir dikey çizgi değil, aşağı doğru yana kayabilen
bir *dikiş*tir (seam). Taban çizgisi civarındaki mürekkebi kesmek indirimlidir:
bitişik yazıda harfleri bağlayan çizgi tam oradadır ve kesilmesi *gereken* yer
orasıdır.

**Genişlik maliyeti.** "m" geniş, "i" dardır. Mürekkebin ipucu vermediği yerde
(tamamen bitişik yazı) kesimi doğru yere bu önsel oturtur. Önseller bir
geçişten sonra kullanıcının *kendi* yazısından ölçülüp güncellenir — kimi
insanın "m"si dar, kiminin "a"sı geniştir.

Karışık yazı bu bileşimden tek mekanizmayla geçer: harfler ayrıksa kesim
maliyeti boşlukta zaten sıfırdır ve kesim oraya oturur; bitişikse genişlik
önseli devreye girip en az mürekkep kesen yeri seçer.

### Tutarlılık (`glyph.py`)

Ham kesitler doğrudan fonta konsa ortaya fidye mektubu çıkar. Yapılanlar:

- **Ortak eğim.** Sayfa başına tek bir eğim ölçülür ve giderilir. Satır başına
  ayrı düzeltmek tutarsızlık üretirdi.
- **Ölçülmüş x-yüksekliği.** Segmentasyondan *sonra*, gerçek x-yüksekliği
  harflerinden ölçülür. Satırın mürekkep profilinden kestirmek, salt büyük
  harfli satırlarda (alfabe satırı) cap-height'ı x-yüksekliği sanır ve o
  satırın bütün glifleri küçük çıkar.
- **Akla yatkınlık filtresi.** x-yüksekliğinde olması gereken bir "a" iki katı
  boydaysa, kesim komşusundan parça kopartmıştır. Kümelemeye girmeden elenir.
- **Dikey düzenleme.** Kağıtta dikey titreme her harfte rastgeledir ve gözü
  rahatsız etmez. Fontta ise aynı glif defalarca kullanılır, yani titreme
  *sistematik* olur — bir kez yukarıda kalmış "a" metnin her yerinde yukarıda
  kalır. Bu yüzden dikey konum büyük ölçüde hizalanır.

### Doğal görünüm: varyant döngüsü (`fontbuild.py`)

El yazısı fontlarının sahte görünmesinin bir numaralı sebebi her "a"nın piksel
piksel aynı olmasıdır. Her karakterin birden çok gerçek örneği fonta konur ve
OpenType `calt` ile sırayla kullanılır:

```fea
sub @BASE @BASE' by @ALT1;    # temel harften sonraki -> 1. varyant
sub @ALT1 @BASE' by @ALT2;    # 1. varyanttan sonraki -> 2. varyant
```

Üçüncü kurala gerek yok: ALT2'den sonraki harf hiçbir kurala uymaz ve temel
biçimde kalır, döngü kendiliğinden `BASE→ALT1→ALT2→BASE` olur.

Varyantlar *çekirdek* havuzdan seçilir. Çeşitliliği doğrudan "merkeze en uzak
örnek" diye aramak, tam da hasarlı örnekleri seçmek demektir — yanına "r"
yapışmış bir "o", tanımı gereği en uzaktaki örnektir.

### Ölçülmüş metrikler

İlerleme genişlikleri, yan boşluklar ve boşluk karakterinin genişliği tahmin
edilmez; kağıt üzerindeki gerçek harf aralıklarından ölçülür. Kişinin yazı
ritmi fonta böyle geçer.

## Kalite ölçümü

`handwrite.synth` sentetik el yazısı üretirken **her mürekkep pikselinin hangi
karakter tarafından çizildiğini** kaydeder. Bu sayede segmentasyon göz kararı
değil piksel sayarak ölçülür (`handwrite.bench`):

- **saflık** — bir glife atanan mürekkebin yüzde kaçı gerçekten o harfe ait?
- **kapsama** — o harfin mürekkebinin yüzde kaçı glife girdi?

İkisi birden gerekir: yalnız saflığa bakılsa çok dar kesmek, yalnız kapsamaya
bakılsa çok geniş kesmek "iyi" görünürdü.

Mevcut durum — 4 sayfa, 1138 değerlendirilen karakter, 9° eğik ve %45 oranında
bitişik sentetik yazı, kamera bozulması (perspektif + düzensiz ışık + bulanıklık
+ gürültü) uygulanmış:

| ölçüt | medyan | ortalama |
|---|---|---|
| saflık | 1.000 | 0.904 |
| kapsama | 0.985 | 0.895 |

Karakterlerin %82,3'ü her iki ölçütte de 0.80'in üstünde; %7,3'ü 0.40'ın
altında.

Kalan hatalı kuyruğu aykırı değer elemesi temizler; fonta giren glifler bu
yüzden ölçülen ortalamadan daha temizdir.

Parametreler tek bir sayfaya göre değil, farklı yazı stillerinde (eğik bitişik /
dik ayrık / geriye eğik) ölçülerek seçildi — tek sayfada en iyi puanı veren
ayar diğer stillerde en kötüsü çıkıyor.

## Bilinen sınırlar

- **Üretilen harfler kullanıcının eli değil.** 3. kademeyle üretilen ~32
  karakter (çoğu büyük harf, rakam, noktalama) aynı fontun parçası gibi durur
  ama dikkatli bir göz farkı görür. Kapsamayı tam ve gerçek istiyorsanız
  şablonlu mod ya da ikinci bir "eksikleri tamamlama" sayfası gerekir.
- **Okuma hatasının tespiti güvenilir değil.** Hizalama maliyeti şüpheli
  kelimeleri yakalamak için kullanılıyor ama iki kademeli bölme hatayı kelime
  sınırını kaydırarak kısmen soğurduğu için maliyet beklendiği kadar
  yükselmiyor. Asıl koruma, bozuk gliflerin aykırı elemesinde düşmesi.
- **Tamamen bitişik yazı** (harfler baştan sona bağlı) en zor durum. Çalışıyor
  ama hata oranı ayrık yazıdan yüksek.
- **Kerning çifti üretilmiyor.** Ölçülen yan boşluklar doğal aralığı büyük
  ölçüde veriyor ama gerçek çift bazlı kerning için örnek sayısı yetersiz.
- **Bağlantı (`liga`) modellenmiyor.** Bitişik yazıda harfleri birleştiren
  giriş/çıkış çizgileri fontta üretilmiyor; harfler ayrı duruyor.
- Sayfa sayısı ArUco sözlüğü nedeniyle 12 ile sınırlı.

## Kurulum

**Linux / macOS**

```bash
python3 -m venv .venv
.venv/bin/pip install -e .          # çekirdek
.venv/bin/pip install -e '.[web]'   # web arayüzü de
.venv/bin/pip install -e '.[web,test]'
.venv/bin/pytest                    # 58 test, ~4 dk
```

**Windows (PowerShell)**

Yol ayıracı ve sanal ortam düzeni farklıdır; `python3` yerine `py` kullanılır:

```powershell
py -m venv .venv
.venv\Scripts\pip install -e .
.venv\Scripts\pip install -e ".[web]"
.venv\Scripts\pytest

$env:GOOGLE_AI_API_KEY = "..."
.venv\Scripts\handwrite read yazim.jpg -o Benim.ttf --preview
```

`py` komutu yoksa Python kurulu değildir: python.org/downloads adresinden
kurun ve kurulumda **"Add python.exe to PATH"** kutusunu işaretleyin.
Windows'un "Python bulunamadı, Microsoft Store'dan yükleyin" mesajı, kurulu
olmayan Python için gösterilen bir yer tutucudur.

Birden çok fotoğraf verirken joker karakter iki platformda da çalışır
(`foto*.jpg`): PowerShell kalıpları yerleşik olmayan komutlar için
genişletmediğinden, genişletmeyi araç kendisi yapar.

## Modüller

| dosya | iş |
|---|---|
| `config.py` | bütün ayarlanabilir sabitler ve tipografik önseller |
| `template.py` | çalışma sayfası üretimi, sayfa geometrisi |
| `preprocess.py` | perspektif düzeltme, aydınlatma, mürekkep ayırma |
| `lines.py` | satır çıkarma, dikey metrikler, eğim |
| `segment.py` | kısıtlı karakter segmentasyonu (çekirdek algoritma) |
| `glyph.py` | normalizasyon, aykırı eleme, varyant seçimi |
| `vectorize.py` | bitmap → Bézier kontur |
| `fontbuild.py` | TTF üretimi, `calt` varyant döngüsü |
| `pipeline.py` | uçtan uca akış ve teşhis |
| `ai/gemini.py` | satır satır el yazısı okuma (Gemini) |
| `synthesize.py` | eksik gliflerin üretilmesi (3 kademe) |
| `synth.py` | sentetik el yazısı üreteci (test için) |
| `bench.py` | segmentasyon doğruluğu ölçümü |
| `specimen.py` | örnek sayfa çizimi |
| `web.py` | web arayüzü — son kullanıcının gördüğü tek yüz |
| `cli.py` | komut satırı (geliştirme ve toplu iş) |

## Lisans

[MIT](LICENSE) © 2026 Alper Alyaz
