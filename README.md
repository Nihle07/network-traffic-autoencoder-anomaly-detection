# network-traffic-autoencoder-anomaly-detection
Autoencoder-based network traffic anomaly detection project

# Network Traffic Autoencoder Anomaly Detection

Bu projede normal ağ trafiği öğrenilerek anormal ağ davranışlarının tespit edilmesi amaçlanmıştır. Model, PyTorch ile geliştirilen Autoencoder mimarisidir.

## Çalışma Akışı

1. Normal ağ trafiği PCAP olarak kaydedildi.
2. PCAP kayıtları 10 saniyelik pencerelere ayrıldı.
3. Her pencere için 10 ağ özelliği çıkarıldı.
4. Normal veriler ile Autoencoder modeli eğitildi.
5. Doğrulama reconstruction error değerlerinin %95 persentili threshold olarak belirlendi.
6. Yoğun bağlantı, reset/başarısız bağlantı, DNS yoğunluğu ve trafik hacmi anomalileri test edildi.

## Kullanılan Özellikler

- packet_count
- bytes_per_second
- average_packet_size
- tcp_packet_count
- udp_packet_count
- syn_count
- rst_count
- dns_query_count
- unique_domains
- unique_destination_ports

## Kod Dosyaları

- `extract_features.py`: Normal trafik kayıtlarından eğitim özelliklerini çıkarır.
- `prepare_trainandvalidation_data.py`: Veriyi eğitim ve doğrulama kümelerine ayırır ve ölçekler.
- `train_autoencoder.py`: Autoencoder modelini eğitir ve threshold değerini hesaplar.
- `extract_test_features.py`: Test ve anomali kayıtlarından özellik çıkarır.
- `test_model.py`: Eğitilmiş modeli test eder, sonuçları ve grafikleri üretir.

## Nihai Sonuç

Nihai threshold değeri **2.1749** olarak belirlenmiştir. Model, 11 anomali penceresinin 5’ini doğru tespit etmiş; normal test verisindeki 31 pencerenin 18’ini doğru normal olarak sınıflandırmıştır. Sonuçlar, yaklaşımın çalıştığını ancak yanlış alarm oranını azaltmak için daha çeşitli normal trafik verisine ihtiyaç olduğunu göstermektedir.

> Ham PCAP ve CSV veri dosyaları depoya eklenmemiştir.
> 
