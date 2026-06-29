# Audit công thức so với reference M1 - 2026-06-14

## Kết luận nhanh

Dự án đã có phần lớn cơ chế toán học trong code: prototype inference, L2 scoring, threshold tại FAR, DET/AUC/EER, MFCC, PCEN, Triplet, GE2E và SCAF. Tuy nhiên bản thesis mới nhất `docs/thesis/Do_An_KWS_thesis_day_du_vi_2026_06_13.md` chưa trình bày đầy đủ công thức tương tự reference `docs/references/M1-Phan_Thanh_Binh-KWS_Master.pdf`.

Nói ngắn gọn: hệ thống có đủ nền để viết công thức, nhưng tài liệu hiện tại mới đủ mức giải thích kỹ thuật, chưa đủ mức công thức học thuật như reference.

## 1. Ví dụ cụ thể về một run GSC v2 trong dự án

Giả sử chạy `gsc_edgespot_exact`, `run_idx = 0`, `k_shot = 10`, `query_split = test`.

Tập positive target luôn có 11 lớp:

```text
yes, no, up, down, left, right, on, off, stop, go, _silence_
```

Trong đó `_silence_` không phải spoken word bình thường. Nó được sinh từ các đoạn crop 1 giây của thư mục `_background_noise_`.

Tập negative có 25 spoken words còn lại ngoài 10 command target:

```text
zero, one, two, three, four, five, six, seven, eight, nine,
bed, bird, cat, dog, happy, house, marvin, sheila, tree, wow,
backward, forward, follow, learn, visual
```

Pha enrollment/support:

- Với `yes`, code lấy 10 file WAV từ `validation_list.txt`.
- 10 file này được đưa qua feature extractor và encoder để tạo 10 embedding.
- Prototype của `yes` là trung bình 10 embedding đó:

```text
p_yes = normalize((z_yes,1 + z_yes,2 + ... + z_yes,10) / 10)
```

- Làm tương tự cho `no`, `up`, ..., `go`.
- Với `_silence_`, code lấy 10 crop từ background noise rồi tính prototype silence.

Pha query/test:

- Với mỗi target như `yes`, code lấy tối đa 50 query samples từ split test. Các mẫu này phải được accept và phân loại đúng là `yes`.
- Với mỗi negative word như `cat`, code cũng lấy tối đa 50 query samples. Vì `cat` không được enroll, kết quả đúng là reject thành `unknown`.
- Nếu `cat` bị accept thành `yes`, `no`, `stop`, ... thì đó là false accept.
- Nếu `yes` bị reject thành `unknown`, đó là false reject.
- Nếu `yes` bị accept nhưng gán thành `no`, đó là lỗi keyword confusion.

Ngưỡng tại `FAR = 1%` được chọn từ toàn bộ score sao cho tỷ lệ negative bị accept không vượt 1%. Sau đó open-set accuracy được tính trên cả positive và negative query.

Dev/test trong dự án:

- `dev30`: chạy 30 repeated runs, thường dùng để chọn checkpoint hoặc phân tích dev.
- `test100`: chạy 100 repeated runs, dùng cho báo cáo cuối.

## 2. Các nhóm công thức có trong reference M1

Reference M1 có các nhóm công thức chính sau:

| Nhóm | Công thức/mô tả trong reference | Trạng thái trong thesis hiện tại |
| --- | --- | --- |
| ProtoNet encoder | `f_theta: R^D -> R^M` | Có nhắc, chưa trình bày thành hệ công thức đầy đủ |
| Prototype | `c_k = 1 / |S_k| sum f_theta(x_i)` | Có trong bảng inference, nhưng chưa đánh số/giải thích như reference |
| Softmax trên khoảng cách | `p_theta(y=k|x_q) = exp(-d(f_theta(x_q), c_k)) / sum exp(...)` | Thiếu, trong khi cần để so baseline probability với direct L2 |
| Episodic support/query | `S_e`, `Q_e`, N-way K-shot | Có mô tả bằng lời, thiếu công thức |
| Episodic NLL loss | `L_e = -1/(NQ) sum log p_theta(...)` | Thiếu |
| Open-set decision | `argmax_i p_i nếu max_i p_i >= gamma, unknown nếu không` | Có mô tả threshold, thiếu công thức baseline probability |
| MFCC | pre-emphasis, framing, windowing, DFT, power, mel filters, log, DCT | Thesis chỉ mô tả bằng lời, thiếu chuỗi công thức |
| DSCNN | depthwise conv và pointwise conv | Có mô tả kiến trúc, thiếu công thức |
| Triplet | `max(0, d(a,p)-d(a,n)+m)` | Có mô tả bằng lời, thiếu công thức trong thesis chính |
| AP/ArcFace/OpenMAX | cosine margin, unknown score | Dự án có SCAF, nhưng thesis thiếu công thức angular/sub-center |
| Metrics | TP, FP, TN, FN, TPR, FPR, FAR, FRR, DET | Có mô tả metric, thiếu công thức đầy đủ |
| Direct distance | loại bỏ probability normalization, dùng L2 radius | Có trong code và mô tả, cần viết công thức rõ hơn |
| Mean DET | trung bình DET theo class/run | Có code, thesis chưa viết formal |

## 3. Bằng chứng từ code dự án

Các file đã kiểm tra:

- `src/evaluation/protocols.py`: định nghĩa `GSC_POSITIVE_WORDS`, `EDGESPOT_TARGET_WORDS`, `GSC_ALL_35_WORDS`, partition positive/negative và vòng evaluate nhiều runs.
- `src/evaluation/gsc.py`: support lấy từ `validation_list.txt`, query lấy từ `testing_list.txt` hoặc train/dev split; `_silence_` sinh từ `_background_noise_`.
- `src/evaluation/metrics.py`: có `compute_det_curve`, `compute_mean_det`, `compute_auc`, `compute_eer`, `get_threshold_at_far`, `compute_open_set_acc_at_far`, `compute_frr_at_far`.
- `src/features/mfcc.py`: có STFT, power spectrum, mel filterbank, log-mel và DCT để tạo MFCC.
- `src/features/pcen.py`: có PCEN trainable với smoother IIR và adaptive gain compression.
- `src/models/prototypical.py`: có Triplet loss và mining.
- `src/models/ge2e.py`: có GE2E centroid classification loss.
- `src/models/arcface.py`: có ArcFace và Sub-center ArcFace loss.

## 4. Công thức nên bổ sung vào thesis

### 4.1. Embedding và prototype

```text
z = f_theta(x)
z_hat = z / ||z||_2
p_c = normalize(1/K * sum_{i=1..K} f_theta(x_i^c))
d_c(x) = ||z_hat - p_c||_2
c* = argmin_c d_c(x)
```

### 4.2. Direct L2 scoring và open-set decision của dự án

```text
s(x) = - min_c d_c(x)
tau_alpha = max{tau : FAR(tau) <= alpha}

predict(x) =
  c*,       nếu s(x) >= tau_alpha
  unknown,  nếu s(x) < tau_alpha
```

Với demo có close-word guard:

```text
margin(x) = d_second(x) - d_best(x)
accept nếu d_best <= threshold và margin >= accept_margin
```

### 4.3. Episodic training

```text
C_e subset C_train, |C_e| = N
S_e = union_{c in C_e} {(x_i^c, c)}_{i=1..K}
Q_e = union_{c in C_e} {(x_j^c, c)}_{j=K+1..K+Q}
```

Với ProtoNet-style NLL:

```text
L_e = -1/(NQ) * sum_{(x_q,y_q) in Q_e} log p_theta(y=y_q | x_q)
```

### 4.4. Probability baseline để giải thích vì sao direct L2 tốt hơn

```text
p_theta(y=k|x_q) =
  exp(-d(f_theta(x_q), c_k)) /
  sum_{k'} exp(-d(f_theta(x_q), c_{k'}))
```

Phần này quan trọng vì reference M1 so sánh probability-based scoring với direct L2 scoring. Thesis của dự án đang dùng direct L2 là chính, nhưng vẫn nên viết probability baseline để người đọc hiểu điểm khác.

### 4.5. MFCC pipeline

```text
x[n] = y[n] - alpha * y[n-1]
x_m[n] = x[n + mH]
x_m[n] = x_m[n] * w[n]
X_m[k] = sum_{n=0}^{N-1} x_m[n] exp(-j 2 pi k n / N)
P_m[k] = |X_m[k]|^2
S_m[r] = sum_k P_m[k] H_r[k]
M(f) = 2595 log10(1 + f/700)
C[q] = sum_r log(S_m[r]) cos(pi q (2r+1) / (2R))
```

### 4.6. PCEN

Code dự án dùng PCEN trainable:

```text
M(t,f) = (1 - s) M(t-1,f) + s E(t,f)
PCEN(t,f) = ( E(t,f) / (eps + M(t,f))^alpha + delta )^r - delta^r
```

Trong thesis cần giải thích `E(t,f)` là mel energy, `M(t,f)` là smoothed energy, còn `alpha`, `delta`, `r`, `s` có thể learnable.

### 4.7. Triplet loss

```text
L_triplet = mean max(0, d(a,p) - d(a,n) + m)
```

Trong dự án:

- `a`: anchor embedding.
- `p`: positive cùng word.
- `n`: negative khác word.
- `m`: margin.
- Mining có thể là random, hard hoặc semi-hard.

### 4.8. GE2E

```text
c_k = normalize(mean_{x_i in support(k)} z_i)
l_{q,k} = w * cos(z_q, c_k) + b
L_GE2E = CE(softmax(l_q), y_q)
```

Trong code, `w = exp(log_scale)` bị clamp trong khoảng hợp lý, `b` là bias learnable.

### 4.9. SCAF / Sub-center ArcFace

```text
cos(theta_j) = max_{r=1..K} <normalize(z), normalize(W_{j,r})>
phi_y = cos(theta_y + m)
logit_y = s * phi_y
logit_j = s * cos(theta_j), j != y
L_SCAF = CE(logits, y)
```

Điểm cần nhấn mạnh: mỗi class có nhiều sub-center, phù hợp khi cùng một word có nhiều speaker/accent. Nhưng nếu scale/margin/weight chưa tune, SCAF có thể collapse trên cap620.

### 4.10. Metric open-set

```text
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
FAR = FPR
FRR = 1 - TPR = FN / (TP + FN)
```

Tại target FAR `alpha`:

```text
tau_alpha = max{tau : FAR(tau) <= alpha}
ACC@alphaFAR =
  (# positive accepted đúng class + # negative rejected) / # all queries
```

DET curve:

```text
DET = {(FAR(tau), FRR(tau)) : tau chạy qua các ngưỡng}
```

Mean DET:

```text
mean_FRR(u) = 1/C * sum_{c=1..C} FRR_c(u)
```

trong đó các DET curve theo class được nội suy về cùng trục FAR `u`.

## 5. Đề xuất sửa thesis

Nên thêm một mục riêng trong Chương 3, ví dụ:

```text
3.11. Ký hiệu và công thức nền tảng
3.11.1. Embedding, prototype và direct L2 decision
3.11.2. Episodic training
3.11.3. MFCC và PCEN
3.11.4. Triplet, GE2E và SCAF
3.11.5. Metric open-set: FAR, FRR, ACC@FAR, AUC, EER, DET
```

Sau đó trong Chương 4 chỉ cần dùng lại ký hiệu này khi giải thích `dev30`, `test100`, `FAR=1%`, `FAR=5%` và so sánh EdgeSpot-4.

## 6. Trạng thái đủ/chưa đủ

| Mục | Code có chưa | Thesis viết đủ chưa | Kết luận |
| --- | --- | --- | --- |
| GSC target/negative protocol | Có | Gần đủ | Cần thêm ví dụ cụ thể và công thức tập |
| Prototype inference | Có | Một phần | Cần formal hóa |
| Direct L2 threshold | Có | Một phần | Cần công thức score và decision |
| Probability baseline | Có option trong evaluation | Thiếu | Cần thêm để giống reference |
| MFCC | Có | Thiếu công thức | Cần bổ sung |
| PCEN | Có | Thiếu công thức | Cần bổ sung |
| Triplet | Có | Thiếu công thức | Cần bổ sung |
| GE2E | Có | Thiếu công thức | Cần bổ sung |
| SCAF | Có | Thiếu công thức | Cần bổ sung |
| FAR/FRR/AUC/EER/DET | Có | Một phần | Cần viết formal |
| Mean DET | Có code | Thiếu | Cần bổ sung nếu muốn giống reference |

Kết luận cuối: thesis hiện tại chưa đầy đủ công thức tương tự reference. Cần bổ sung một section công thức formal trước khi coi bản Word là bản gần chuẩn nộp/bảo vệ.
