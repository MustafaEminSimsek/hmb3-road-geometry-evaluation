# HMB3 sürüş görüntüsü analizi

## Proje yapısı

| Klasör | İçerik |
| --- | --- |
| [src/](src/) | Görünüm farkı ve yol geometrisi sınıflandırma kodu |
| [data/](data/) | Orijinal video ve düzenlenmiş karelerin ZIP arşivi |
| [models/](models/) | SegFormer ONNX ağırlığı |
| [docs/](docs/) | Çalışmada kullanılan bilimsel kaynakların PDF dosyaları |
| [results/suitability/](results/suitability/) | Nihai uygunluk tablosu, özet ve [işaretli örnekler](results/suitability/examples/) |
| [results/lucyrestyle/](results/lucyrestyle/) | LucyRestyle uygunluk tablosu ve işaretli örnekleri |
| [results/comparison/](results/comparison/) | InstructPix2Pix ve LucyRestyle için kare bazında karşılaştırma |
| [results/appearance/](results/appearance/) | İlk görünüm farkı deneyi ve önizlemeler |
| [results/experiments/](results/experiments/) | Pilot çalışma, JPEG kontrolleri ve 0,60 kenar eşiği sonuçları |
| `.local/` | Bu bilgisayarda kullanılan çalışma zamanı paketleri ve önbellek |

Nihai uygunluk sonuçları [uygunluk.csv](results/suitability/uygunluk.csv) dosyasındadır. Aşağıdaki komutları proje kökünden çalıştırın. Mevcut sonuçları saklayarak yeni bir çalışma yapmak için `--output` ile farklı bir konum seçin.

## Bilimsel kaynaklar

GitHub kopyasından yeni bir tarama çalıştırmak için orijinal HMB3 videosunu ve düzenlenmiş kare arşivlerini `data/` klasörüne yerleştirin. [SegFormer ONNX ağırlığını](https://huggingface.co/Xenova/segformer-b0-finetuned-cityscapes-512-1024/resolve/498e4add74041c22a9908388f600d836d57f06ea/onnx/model.onnx) `models/segformer_cityscapes.onnx` adıyla kaydedin. Kayıtlı skorlardan karşılaştırmayı yeniden üretmek için `python src/hmb3_compare.py --mask-only` yeterli.

| Makale | Projedeki yeri |
| --- | --- |
| [Efficient Domain Augmentation for Autonomous Driving Testing Using Diffusion Models](https://arxiv.org/abs/2409.13661) | Yol geometrisi doğrulama akışı ve 0,90 OC-TSS eşiği |
| [Assessing Quality Metrics for Neural Reality Gap Input Mitigation](docs/OC-TSS_Assessing_Quality_Metrics.pdf) | OC-TSS yönteminin özgün çalışması |
| [SegFormer](docs/SegFormer.pdf) | HMB3 yol maskelerini çıkaran SegFormer-B0 modelinin mimarisi ve benchmark sonuçları |
| [Cityscapes](docs/Cityscapes.pdf) | Kullandığımız SegFormer ağırlığının eğitildiği veri seti |
| [Boundary IoU](docs/Boundary_IoU.pdf) | HMB3 deneyine eklenen yol kenarı örtüşme ölçümü |
| [U-Net](docs/U-Net.pdf) | Referans çalışmada kullanılan segmentasyon mimarisi |
| [InstructPix2Pix](docs/InstructPix2Pix.pdf) | Düzenlenmiş karelerin üretim yöntemi |

## İlk görünüm farkı deneyi

## Çalıştırma

Python ortamına gereken paketleri kurun.

```powershell
python -m pip install -r requirements.txt
python src/hmb3_detect.py --evaluate results/appearance/hmb3_evaluation.json --previews results/appearance/previews
```

Varsayılan girdiler `data/` içindeki HMB3 MP4 ve ZIP dosyalarıdır. Program ZIP'teki sayısal zaman damgalarını sıralar ve bunları videonun kare sırasıyla eşleştirir. Farklı girdiler için `--video`, `--archive` ve `--output` seçenekleri vardır. Toplu giriş, kesintisiz ve aynı sıra ile hazırlanmış bir video/ZIP çifti gerektirir. Referans kare numarası bilinen tek bir görsel için aşağıdaki komut JSON sonucu verir. İlk kare 0 numaralıdır.

```powershell
python src/hmb3_detect.py --candidate aday.jpg --frame 0
```

## Karar yöntemi

Orijinal kare adayla aynı boyuta getirilir. Görünüm skoru, küçültülmüş ve Gauss filtresiyle yumuşatılmış Lab görüntülerindeki ortalama mutlak farktır. Gradyan kosinüs benzerliği 0,35'in altındaki eşleşmeler `belirsiz` olarak işaretlenir. Bu kontrol özellikle yanlış sahneleri yakalamaya yöneliktir. Birbirine çok benzeyen komşu kareler zaman damgasıyla doğrulanmalıdır. İlk %60'lık bloktan oluşturulan JPEG kontrollerin %99,5'lik görünüm skoru `normal` için üst sınırdır. Bu değerin 1,5 katı `edited` için alt sınırdır. Aradaki ölçümler `belirsiz` olur. Son %35 test için kullanılır, arada %5'lik tampon vardır.

## Makaledeki yol doğrulaması

Makalenin III-B bölümündeki yol sınıfı benzerliği iki ikili yol maskesinin Jaccard benzerliğiyle hesaplanır. U-Net çıktıları mevcutsa CSV'deki `oc_tss` bu skoru, `road_validation` ise varsayılan 0,90 eşiğiyle yol geometrisi kararını verir. Görünüm etiketi bu skordan bağımsız hesaplanır. Yol maskeleriyle birlikte referans yol ve arka plan alanlarının görünüm farkları da kaydedilir.

```powershell
python src/hmb3_detect.py --road-model road_unet.onnx --model-width 512 --model-height 384 --previews results/appearance/previews
```

Bu komuttaki ONNX modeli, RGB, [0,1] ölçeğinde NCHW giriş ve [1,1,H,W] biçiminde yol olasılığı çıkışı vermelidir. Model farklı normalizasyon gerektiriyorsa bu dönüşüm ONNX dışa aktarımına eklenmelidir. Yazarların model arşivi dört sürüş modeli içeriyor. İlk görünüm değerlendirmesinin CSV dosyasında U-Net yol skoru boş ve yol doğrulaması `unavailable` olarak kalır. HMB3 üzerinde SegFormer ile üretilen yol skorları aşağıdaki uygunluk komutunun çıktısındadır.

- [Makalenin yöntem ve deney bölümleri](https://arxiv.org/abs/2409.13661)
- [Yazarların Jaccard hesabı](https://github.com/deib-polimi/online-testing-augmented-simulator/blob/master/scripts/semantic/compute_segmentation_metric.py)
- [Yazarların U-Net değerlendirme kodu](https://github.com/DragonBanana/udacity-gym/blob/HEAD/udacity_gym/extras/model/segmentation/unet/evaluate.py)

## HMB3 test sonucu

Test bloğunda arşivdeki 1.964 kare `edited` olarak etiketlendi. Orijinal karelerden JPEG kalite 60 ve 90 ile oluşturulan 3.928 normal kontrolün 3.870'i `normal`, 58'i `belirsiz` çıktı. Bu testte edited precision, recall ve F1 1,00, tüm test girdilerindeki belirsiz oranı %0,98 oldu. Bu ölçüm tek HMB3 kaydına, arşivdeki düzenlemelere ve belirtilen normal kontrol dönüşümlerine aittir.

## Yol geometrisine göre uygunluk

`hmb3_suitability.py`, eşlenen video ve ZIP karelerine Cityscapes ile eğitilmiş aynı SegFormer-B0 modelini uygular. Bu deneyde yol sınıfının sayısal etiketi 0'dır. Model 320×256 boyutunda RGB girdi alır. Her sınıfın çıktısı 512×384 boyutuna taşınır ve piksel başına en yüksek skorlu sınıf seçilir.

```powershell
python -m pip install -r requirements.txt
python src/hmb3_suitability.py
```

Model dosyası `models/segformer_cityscapes.onnx` konumundadır. `--model` ile konumu değiştirilebilir. Varsayılan komut tüm kareleri tarar. `--stride 56` yaklaşık 2,8 saniyede bir kare seçer.

Windows'ta GPU ile çalıştırmak için ayrı bir Python ortamında `onnxruntime-directml` paketini kurup komuta `--directml` ekleyin. CSV ve örnek görseller `--output` ile seçilen klasöre yazılır.

- Road IoU, iki yol maskesinin kesişimini birleşimine böler. Başlangıç eşiği makaledeki 0,90'dır.
- Boundary IoU, aynı oranı maskelerin iç kenarından 8 piksel genişliğindeki yol bantlarında hesaplar. Görüntünün dış çerçevesi sınır sayılmaz. İlk pilotta 0,60 kullanıldı. JPEG kalite 60 ile yeniden sıkıştırılmış 24 değişmemiş kontrolde bu eşikle 14, 0,40 ile 22 çift iki eşiği birden geçti. Bu nedenle varsayılan kenar eşiği 0,40 olarak ayarlandı.
- İki eşiği geçen çift `uygun`, eşiklerden en az birinin altında kalan çift `uygunsuz` olarak etiketlenir. Çok küçük veya büyük maske, boş kenar ya da zayıf eşleşme durumunda karar `inceleme` olur.
- Renkli örneklerde yeşil ortak bölgeyi, turuncu yalnızca orijinal maskeyi, mavi yalnızca düzenlenmiş maskeyi gösterir.

Otomatik kararlar modelin çıkardığı maskelere bağlıdır. Düşük skor, gerçek geometri farkı veya segmentasyon hatası kaynaklı olabilir. Kenar bantları araç gibi nesnelerin yol maskesinde bıraktığı sınırları da içerebilir. Şerit çizgisi konumu ayrıca ölçülmemiştir. Eşikler elle etiketlenmiş HMB3 verisiyle doğrulanmamıştır. Mevcut CSV'ye `--scores DOSYA.csv` vererek model çalıştırmadan yeni eşik uygulayabilirsiniz.

[SegFormer-B0 Cityscapes ONNX](https://huggingface.co/Xenova/segformer-b0-finetuned-cityscapes-512-1024/tree/498e4add74041c22a9908388f600d836d57f06ea/onnx) · [Boundary IoU yöntemi](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html)

### Görsel inceleme örnekleri

| Kare | Road IoU | Boundary IoU | Otomatik karar | Görsel gözlem |
| --- | ---: | ---: | --- | --- |
| 1904 | 0,989 | 0,674 | Uygun | Islak yol görünümü eklenmiş. Sağ viraj ve görünen çizgiler iki karede benzer konumda. |
| 3472 | 0,981 | 0,605 | Uygun | Sol viraj ve yolun sağ sınırı iki karede benzer konumda. |
| 1792 | 0,942 | 0,154 | Uygunsuz | Parlama altında orijinal sağ beyaz çizginin bir kısmı silikleşmiş. Yol alanı eşiği geçerken kenar eşiği geçilmiyor. |
| 2856 | 0,853 | 0,182 | Uygunsuz | Sağ banket ve gölgeli sol bölgelerde iki maske farklı yorum yapıyor. Maskedeki farkın geometri değişikliğinden kaynaklandığı kesin değil. |
| 448 | 0,954 | 0,214 | Uygunsuz | Şerit çizgileri görsel olarak benzer konumda. Özellikle sağ banket, iki yol maskesinde farklı yerde bitiyor. Otomatik eleme görsel inceleme gerektiriyor. |
| 4872 | 0,872 | 0,141 | Uygunsuz | Düzenlenmiş karede orijinal araçların bulunduğu bölgede şemsiyeli bir yaya beliriyor. Sarı kutular bu farkı elle işaretliyor. |

İşaretli görseller `results/suitability/examples` klasöründedir. Pilot, 56 karede bir alınan 101 çiftten oluşur. Road IoU 0,90 ve Boundary IoU 0,40 eşiklerinde pilotta 40 çift geçer, 61 çift elenir. [JPEG kontrol tablosu](results/experiments/boundary_060_pilot/jpeg60_kontrol.csv)

### Tam arşiv sonuçları

5.614 eşleşmenin 2.279'u uygun, 3.327'si uygunsuz ve 8'i inceleme olarak işaretlendi. Yalnızca kenar eşiğinde elenen çift sayısı 2.387. Boundary IoU eşiği 0,50 olsaydı 1.068, 0,60 olsaydı 425 çift iki eşiği de geçecekti.

- [Kare bazında uygunluk sonuçları](results/suitability/uygunluk.csv)
- [Sınıf sayıları, eşik karşılaştırması ve model özeti](results/suitability/ozet.json)

### Güncel karşılaştırma · Road IoU ve Boundary IoU

`python src/hmb3_compare.py --mask-only` komutu, Road IoU ≥ 0,90 ve Boundary IoU ≥ 0,40 koşullarını birlikte uygular. Görüntü kenarı benzerliği ve maske alanı ön filtreleri karar dışında kalır. 5.614 çiftte InstructPix2Pix için 2.279, LucyRestyle için 60 çift iki eşiği de geçti. Road IoU eşiğini geçip Boundary IoU'da elenen çift sayısı sırasıyla 2.387 ve 1.333.

- [Kare bazında kararlar](results/comparison_mask_only/kare_bazinda_karsilastirma.csv)
- [Karşılaştırma özeti](results/comparison_mask_only/ozet.json)

### Tek eşikli karşılaştırma · Yalnızca Road IoU

Her iki arşivde aynı 5.614 kaynak kare, mevcut SegFormer-B0 maskelerinden hesaplanan Road IoU ile yeniden değerlendirildi. Karar eşiği 0,90. Boundary IoU ve görüntü kenarı benzerliği tanı amaçlı sütunlar olarak saklanır. Road IoU hesaplanamazsa kare incelemeye ayrılır.

```powershell
python src/hmb3_compare.py --road-only
```

| Sonuç | InstructPix2Pix | LucyRestyle |
| --- | ---: | ---: |
| Uygun | 4.666 (%83,1) | 1.393 (%24,8) |
| Uygunsuz | 948 | 4.221 |
| İnceleme | 0 | 0 |

Her iki yöntemde 1.320 kaynak kare eşiği geçti. Önceki kuralla incelemeye ayrılan 2.551 LucyRestyle çiftinden 518’i Road IoU eşiğini geçti, 2.033’ü eşik altında kaldı. Sonuçlar SegFormer-B0 maskeleri üzerinden elde edilmiştir.

- [Yeni ve önceki kararların kare bazında karşılaştırması](results/comparison_road_only/kare_bazinda_karsilastirma.csv)
- [Ana karşılaştırmanın özeti](results/comparison_road_only/ozet.json)

### Ek karşılaştırma · Yol alanı, yol kenarı ve ön kontroller

LucyRestyle ZIP dosyasındaki `hmb3_lucyedit_0000.jpg` ile başlayan kare dizisi orijinal videonun sıfır tabanlı kare numarasıyla eşleştirilir. Zaman damgaları mevcut HMB3 InstructPix2Pix arşivinden alınır. Deney iki düzenleme seti için aynı SegFormer-B0 ağırlığını, 512×384 karşılaştırma boyutunu ve 0,90 / 0,40 eşiklerini kullanır. LucyRestyle karelerinin en-boy oranı kaynak kareden farklıdır. Ayrıca yoğun sis, parlama ve karanlık hem maskeyi hem eşleşme güveni skorunu etkileyebilir. Sonuçlar bu iki arşiv için geçerlidir.

```powershell
python src/hmb3_suitability.py --archive data/hmb3_lucyrestyle_frames.zip --frame-prefix hmb3_lucyedit_ --output results/lucyrestyle
python src/hmb3_compare.py
```

- [LucyRestyle kare sonuçları ve örnekler](results/lucyrestyle/)
- [Aynı kaynak karede iki düzenlemenin skorlarını karşılaştıran tablo](results/comparison/kare_bazinda_karsilastirma.csv)
- [Sınıf sayıları, skor medyanları ve ortak karelerin karşılaştırması](results/comparison/ozet.json)

| 5.614 kaynak karenin sonucu | InstructPix2Pix | LucyRestyle |
| --- | ---: | ---: |
| Uygun | 2.279 | 47 |
| Uygunsuz | 3.327 | 3.016 |
| İnceleme | 8 | 2.551 |
| Medyan Road IoU | 0,953 | 0,840 |
| Medyan Boundary IoU | 0,364 | 0,124 |

İki yöntemde de değerlendirilebilen 3.057 karede InstructPix2Pix 1.217, LucyRestyle 47 uygun sonuç verdi. Bunların 37’si ortaktı. LucyRestyle’da 2.538 kare, 0,35 gradyan benzerliği eşiğinin altında kaldı. [Aynı karede iki yöntemin işaretli karşılaştırması](results/comparison/yalniz_ip2p_uygun_04799.jpg)
