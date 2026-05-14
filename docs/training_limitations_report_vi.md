# Báo Cáo Tiến Độ Và Phân Tích Kết Quả Huấn Luyện KWS Few-Shot

## 1. Mục Tiêu Của Đề Tài

Mục tiêu của đề tài là xây dựng hệ thống **Keyword Spotting few-shot open-set** có khả năng:

- cho phép người dùng thêm một từ khóa mới chỉ bằng khoảng **3-5 mẫu âm thanh**;
- nhận diện đúng từ khóa đã đăng ký;
- từ chối các âm thanh/từ không thuộc tập từ khóa;
- tiến tới chạy được trong bối cảnh **streaming microphone**, không chỉ nhận diện file audio tĩnh.

Kỳ vọng ban đầu là hệ thống có thể đạt độ chính xác cao, lý tưởng khoảng **90-95%** trong điều kiện demo tốt. Tuy nhiên sau nhiều lần thử nghiệm huấn luyện, kết quả thực tế cho thấy bài toán khó hơn dự kiến, đặc biệt ở phần **open-set rejection** và **cross-dataset generalization**.

## 2. Pipeline Đã Xây Dựng Và Cải Tiến

Trong giai đoạn vừa qua, em đã thực hiện các cải tiến chính sau:

### 2.1. Chuẩn hóa dữ liệu MSWC

Ban đầu quá trình tải và xử lý MSWC khá tốn thời gian vì dữ liệu gốc ở dạng OPUS. Em đã sửa notebook huấn luyện để:

- dùng MSWC English Top500 thay vì tải full không kiểm soát;
- tải MSWC Top500 legacy;
- extract OPUS;
- convert OPUS sang WAV;
- lưu WAV cache lên Google Drive;
- nếu Drive đã có WAV cache hợp lệ thì dùng lại trực tiếp, không convert lại.

Cache hiện tại có khoảng:

```text
MSWC: 95,516 WAV
word dirs: 478
train words: 430
val words: 48
```

Điều này giúp pipeline dữ liệu ổn định hơn và tránh mất thời gian convert lại từ đầu.

### 2.2. Sửa lỗi cache và Colab

Trong quá trình chạy Colab có một số lỗi thực tế:

- DEMAND noise dataset bị tải dở do lỗi `504 timeout`, chỉ còn khoảng 32 file thay vì 272 file.
- Local MSWC cache ở `/content/mswc_local` có lúc copy dở, làm dataset train chỉ còn 2 class.
- Copy từng file WAV nhỏ từ Google Drive sang `/content` rất chậm vì có hơn 95 nghìn file nhỏ.

Em đã xử lý bằng cách:

- kiểm tra DEMAND đủ tối thiểu 250 WAV, nếu thiếu thì xóa cache dở và tải lại;
- kiểm tra local MSWC cache có đủ phần lớn train words, nếu thiếu thì xóa và copy lại;
- đổi hướng localize MSWC bằng **tar cache một file lớn** thay vì copy từng WAV nhỏ.

Sau khi sửa, dataset train đúng trở lại:

```text
Dataset: 85,916 samples
Usable classes: 430
```

### 2.3. Model và training

Model chính đang dùng là:

- DSCNN-L encoder;
- input feature: MFCC;
- loss: Triplet Loss;
- episodic training;
- semi-hard mining;
- early stopping theo validation AUC trên MSWC held-out set.

Ngoài ra project đã có nhánh thử nghiệm:

- EdgeSpot-lite;
- mel 40x101;
- trainable PCEN;
- SCAF loss.

Tuy nhiên trong giai đoạn này em ưu tiên đánh giá DSCNN baseline trước vì đây là hướng ổn định hơn để làm báo cáo và demo.

## 3. Thiết Lập Thí Nghiệm Mới Nhất

Run mới nhất:

```text
Run tag: dscnn_top500_mpw200_clean
Model: DSCNN
Feature: MFCC
Loss: Triplet
Dataset: MSWC Top500 cache
Samples: 85,916
Usable classes: 430
Episodes: 200 per epoch
Epochs: 35
N-way: 30
N-samples per class: 20
DataLoader workers: 2
Early stopping: patience 8, min_delta 0.001
```

Quá trình train nhìn có vẻ tốt nếu chỉ xét MSWC validation:

```text
Best MSWC val_auc: 0.9585 tại epoch 25
Early stop tại epoch 33
```

Một số mốc validation:

| Epoch | val_acc | val_auc |
|---:|---:|---:|
| 1 | 0.3037 | 0.8492 |
| 5 | 0.4247 | 0.9195 |
| 10 | 0.4973 | 0.9319 |
| 15 | 0.5533 | 0.9427 |
| 20 | 0.5820 | 0.9510 |
| 25 | 0.6153 | 0.9585 |

Nếu chỉ nhìn bảng này thì model có vẻ đang học tốt. Tuy nhiên khi đánh giá trên GSC, kết quả lại không đạt mục tiêu.

## 4. Kết Quả Đánh Giá Trên GSC

Sau khi train xong, em đánh giá model trên Google Speech Commands v2 với các protocol fixed/random và k-shot 5/10.

| Experiment | AUC | EER | FRR@5% | ACC@5% | KW-ACC | F1 |
|---|---:|---:|---:|---:|---:|---:|
| gsc_fixed_k5 | 0.8494 | 0.2383 | 0.6052 | 0.7421 | 0.6980 | 0.6808 |
| gsc_fixed_k10 | 0.8701 | 0.2190 | 0.5592 | 0.7604 | 0.7500 | 0.7039 |
| gsc_random_k5 | 0.8235 | 0.2582 | 0.6444 | 0.7417 | 0.7836 | 0.6575 |
| gsc_random_k10 | 0.8329 | 0.2511 | 0.5944 | 0.7608 | 0.8184 | 0.6660 |

Nhận xét:

- `KW-ACC` ở `gsc_fixed_k5` chỉ đạt **69.8%**, thấp hơn nhiều so với mục tiêu 80-85%.
- Khi tăng lên `k=10`, `KW-ACC` tăng lên **75.0%**, nhưng vẫn chưa đủ tốt.
- `gsc_random_k10` đạt `KW-ACC=81.84%`, nhưng chỉ số open-set vẫn yếu.
- `FRR@5%` rất cao, dao động khoảng **55-64%**. Nghĩa là nếu hệ thống bị ép giữ false accept rate ở 5%, nó bỏ sót hơn một nửa keyword.

Điểm yếu nghiêm trọng nhất không chỉ là closed-set keyword accuracy, mà là **khả năng open-set ở vùng FAR thấp**.

## 5. Phân Tích Nguyên Nhân

### 5.1. MSWC validation không phản ánh tốt GSC/deployment

Trong run mới nhất, MSWC validation AUC đạt **0.9585**, nhưng khi sang GSC thì AUC chỉ khoảng **0.8494-0.8701** ở fixed protocol.

Điều này cho thấy:

- model học được đặc trưng trên MSWC;
- nhưng đặc trưng đó không transfer tốt sang GSC;
- validation trên MSWC chưa đủ đại diện cho môi trường demo/deployment.

Nói cách khác, checkpoint tốt nhất theo MSWC val không nhất thiết là checkpoint tốt nhất cho GSC hoặc microphone thật.

### 5.2. Triplet loss chưa tạo embedding separation đủ tốt

Trong log train, loss gần như quanh:

```text
loss ≈ 0.999
d_pos ≈ d_neg
```

Ví dụ:

```text
d_pos=0.815, d_neg=0.816
```

Điều này cho thấy khoảng cách giữa positive pair và negative pair chưa tách rõ. Với few-shot KWS, embedding space cần đảm bảo:

- cùng từ phải gần nhau;
- từ khác phải cách xa nhau;
- từ chưa biết phải không rơi gần prototype của từ đã đăng ký.

Khi `d_pos` và `d_neg` gần bằng nhau, prototype classifier dễ nhầm các từ có âm gần nhau.

### 5.3. Bài toán open-set khó hơn closed-set

Closed-set chỉ cần chọn từ gần nhất. Open-set phải quyết định thêm câu hỏi:

> Audio này có thuộc bất kỳ từ khóa đã đăng ký nào không?

Đây là phần khó hơn nhiều. Kết quả `FRR@5%` cao cho thấy khi giảm false alarm, hệ thống phải tăng threshold nghiêm ngặt và bỏ sót nhiều keyword thật.

### 5.4. Các cặp từ gần âm gây nhầm lẫn

Các cặp hay nhầm thuộc nhóm gần âm vị:

- `no/go/down/dog`;
- `four/forward/follow`;
- `three/tree`;
- `off/up`;
- các số và từ ngắn một âm tiết.

Với clip 1 giây tĩnh, model đã khó phân biệt. Khi đưa sang streaming microphone, vấn đề còn khó hơn vì từ có thể lệch cửa sổ, bị cắt đầu/cuối, hoặc dính tiếng nền.

### 5.5. Augmentation có thể đang quá mạnh

Run mới dùng augmentation để tăng robustness:

- DEMAND noise;
- SNR thấp;
- speed perturb;
- SpecAugment.

Augmentation giúp chống nhiễu, nhưng nếu quá mạnh có thể làm mất chi tiết âm vị quan trọng, đặc biệt với các từ ngắn và giống âm. Điều này có thể làm MSWC val tăng nhưng GSC keyword accuracy không tăng tương ứng.

## 6. Kết Luận Sau Nhiều Lần Train

Sau nhiều lần train và sửa pipeline, em kết luận tạm thời:

1. Pipeline dữ liệu đã ổn định hơn trước.
2. Model có học trên MSWC, thể hiện qua MSWC `val_auc` tăng đến **0.9585**.
3. Tuy nhiên model không generalize đủ tốt sang GSC/open-set.
4. Kết quả hiện tại chưa đạt mục tiêu 90-95%.
5. Mục tiêu thực tế hơn trong giai đoạn này là:
   - static few-shot `KW-ACC >= 80-85%`;
   - giảm `FRR@5%`;
   - cải thiện streaming bằng calibration và state machine, không chỉ train thêm.

Điều quan trọng là thất bại hiện tại không phải do lỗi code đơn giản nữa, mà là do giới hạn của pipeline hiện tại:

- metric chọn checkpoint chưa đúng với deployment;
- embedding loss chưa tách đủ tốt;
- open-set rejection chưa được calibration đủ mạnh;
- streaming distribution khác audio clip tĩnh.

## 7. Hướng Khắc Phục Tiếp Theo

### 7.1. Không chọn checkpoint chỉ bằng MSWC val_auc

Cần thêm GSC validation proxy hoặc streaming validation proxy trong quá trình chọn checkpoint. Nếu không, model có thể tốt trên MSWC nhưng kém trên GSC/demo.

### 7.2. Thử cấu hình train nhẹ hơn

Run tiếp theo nên giảm augmentation và margin:

```text
TRIPLET_MARGIN = 0.5
noise_prob = 0.25
SNR = 5-20 dB
speed perturb = 0.95-1.05
hard-pair mining = 0.0
```

Mục tiêu là giữ chi tiết âm vị tốt hơn, thay vì over-regularize.

### 7.3. Cải thiện enrollment và threshold

Với 3-5 mẫu/từ, không nên chỉ lấy mean prototype đơn giản. Cần:

- crop đúng vùng có keyword;
- loại mẫu quá ngắn/quá dài/quá nhiễu;
- tạo augmented embeddings từ mỗi mẫu;
- dùng multi-prototype hoặc robust prototype;
- calibrate threshold riêng từng từ bằng impostor/negative bank.

### 7.4. Cải thiện streaming engine

Để demo microphone tốt hơn, cần chuyển trọng tâm từ static accuracy sang streaming system:

- VAD/energy endpointing;
- multi-window scoring 600/800/1000/1200 ms;
- margin giữa top-1 và top-2;
- voting nhiều frame liên tiếp;
- cooldown chống duplicate;
- báo event gồm keyword, start/end, confidence, distance, margin.

### 7.5. EdgeSpot-lite và phase KD

EdgeSpot trong paper không chỉ là đổi architecture. Điểm mạnh nằm ở:

- mel frontend;
- trainable PCEN;
- temporal attention;
- SCAF;
- knowledge distillation từ teacher lớn như Wav2Vec2;
- training trên dữ liệu rất lớn.

Project hiện mới có EdgeSpot-lite. Nếu còn thời gian, hướng tiếp theo là thử EdgeSpot-lite trước, sau đó mới đến KD teacher.

## 8. Đánh Giá Trung Thực Về Mục Tiêu 90-95%

Với yêu cầu chỉ có 3-5 mẫu cho mỗi từ, lại còn cần open-set và streaming, mục tiêu 90-95% là rất khó nếu chỉ dùng DSCNN + triplet + prototype mean.

Mức có thể đặt thực tế hơn:

- static controlled audio: 80-85%;
- open-set ở FAR thấp: cần calibration tốt mới ổn;
- streaming microphone: đánh giá bằng false alarm/hour, miss rate và latency thay vì chỉ accuracy clip tĩnh.

Do đó, hướng phát triển hợp lý là không chỉ tiếp tục train thêm, mà phải cải thiện cả hệ thống:

```text
data distribution + checkpoint selection + enrollment + threshold calibration + streaming state machine
```

## 9. Tóm Tắt Gửi Giảng Viên

Trong giai đoạn này em đã hoàn thiện lại pipeline dữ liệu, cache MSWC WAV trên Drive, sửa các lỗi Colab/cache, và train lại DSCNN trên MSWC Top500 với 85,916 mẫu thuộc 430 class. Kết quả validation trên MSWC đạt cao nhất `val_auc=0.9585`, cho thấy model có học được embedding trên tập train/val. Tuy nhiên khi đánh giá cross-dataset trên Google Speech Commands, kết quả chưa đạt kỳ vọng: `gsc_fixed_k5` chỉ đạt `KW-ACC=69.8%`, `AUC=0.8494`, và `FRR@5%=60.52%`. Điều này cho thấy model chưa generalize tốt và open-set rejection còn yếu.

Nguyên nhân chính được xác định là MSWC validation chưa phản ánh tốt deployment/GSC, triplet embedding chưa tách đủ rõ giữa positive và negative, các từ ngắn/gần âm gây nhầm lẫn nhiều, và bài toán open-set/streaming khó hơn closed-set classification. Vì vậy em chưa thể báo cáo rằng model đã đạt mức 90-95%. Hướng tiếp theo là chọn checkpoint bằng GSC/streaming proxy, giảm augmentation quá mạnh, cải thiện enrollment/threshold calibration, và hoàn thiện streaming state machine.

