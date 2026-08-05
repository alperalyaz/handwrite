# handwrite

El yazısıyla doldurulmuş sayfaların fotoğrafından tutarlı bir OpenType font üretir.

Calligraphr'ın kutu doldurtmasına gerek yok: normal bir metin gibi, çizgili
kağıda doğal şekilde yazarsınız.

```
handwrite sheets -o calisma          # 1. sayfaları üret, A4'e %100 ölçekte yazdır
                                     # 2. örnek metni kendi elinle yaz, fotoğrafla
handwrite build calisma/sheets.json foto*.jpg -o Benim.ttf --preview
```

Web arayüzü için `pip install 'handwrite[web]'` sonrası `handwrite serve`.

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

- **Tamamen bitişik yazı** (harfler baştan sona bağlı) en zor durum. Çalışıyor
  ama hata oranı ayrık yazıdan yüksek.
- **Kerning çifti üretilmiyor.** Ölçülen yan boşluklar doğal aralığı büyük
  ölçüde veriyor ama gerçek çift bazlı kerning için örnek sayısı yetersiz.
- **Bağlantı (`liga`) modellenmiyor.** Bitişik yazıda harfleri birleştiren
  giriş/çıkış çizgileri fontta üretilmiyor; harfler ayrı duruyor.
- **Eksik karakter sentezi yok.** Metinde geçmeyen bir karakter fonta girmez.
  Gözlenen gliflerden eksikleri üretmek (few-shot font generation) doğal bir
  sonraki adım.
- Sayfa sayısı ArUco sözlüğü nedeniyle 12 ile sınırlı.

## Kurulum

```bash
pip install -e .            # çekirdek
pip install -e '.[web]'     # web arayüzü de
pytest                      # 33 test, ~50 sn
```

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
| `synth.py` | sentetik el yazısı üreteci (test için) |
| `bench.py` | segmentasyon doğruluğu ölçümü |
| `specimen.py` | örnek sayfa çizimi |
| `cli.py` / `web.py` | komut satırı ve web arayüzü |
