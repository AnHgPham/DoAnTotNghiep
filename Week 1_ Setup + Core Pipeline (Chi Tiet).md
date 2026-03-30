# Week 1: Setup + Core Pipeline
Muc tieu: Tu zero den pipeline `WAV -> MFCC -> DSCNN-L -> embedding (1,276)` chay duoc cho 1 file.
## Trang thai hien tai
* Co: 9 `.cursor/rules/*.mdc`, 7 `src/*/AGENTS.md`, master plan (`Untitled-1.md`), thesis PDF
* Chua co: bat ky file Python nao, `requirements.txt`, `configs/`, `README.md`, `__init__.py`
* Thu muc rong: `configs/`, `data/`, `notebooks/`, `scripts/`, `tests/`
## Day 1: Project Setup + Bat dau download data
### 1.1 `requirements.txt`
```warp-runnable-command
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
matplotlib>=3.7.0
tensorboard>=2.13.0
pytest>=7.3.0
gradio>=4.0.0
noisereduce>=3.0.0
speechbrain>=1.0.0
soundfile>=0.12.0
requests>=2.31.0
tqdm>=4.65.0
```
Cai dat: `pip install -r requirements.txt`
### 1.2 `configs/default.yaml`
Tat ca hyperparameters tap trung:
```yaml
seed: 42
audio:
  sample_rate: 16000
  duration_sec: 1.0
  target_length: 16000  # sample_rate * duration_sec
mfcc:
  n_mfcc: 40
  num_features: 10
  n_fft: 1024
  win_length: 640      # 40ms * 16000
  hop_length: 320      # 20ms * 16000
  n_mels: 40
model:
  architecture: "DSCNN-L"
  channels: 276
  num_ds_blocks: 5
  embedding_dim: 276
  feature_mode: "NORM"  # CONV | RELU | NORM
training:
  epochs: 40
  episodes_per_epoch: 400
  n_classes: 80
  n_samples: 20
  triplet_margin: 0.5
  optimizer:
    type: "Adam"
    lr: 0.001
  scheduler:
    type: "StepLR"
    step_size: 20
    gamma: 0.5
noise:
  demand_dir: "data/demand"
  prob: 0.95
  snr_db: 5.0
data:
  gsc_dir: "data/gsc_v2"
  mswc_dir: "data/mswc_en"
  demand_dir: "data/demand"
  mswc_train_words: 450
  mswc_val_words: 50
  mswc_eval_words: 263
checkpoint:
  dir: "checkpoints"
  save_every: 5
```
### 1.3 `src/*/__init__.py`
Tao `__init__.py` rong trong tat ca 7 thu muc:
`src/features/`, `src/models/`, `src/classifiers/`, `src/evaluation/`, `src/streaming/`, `src/enhancements/`, `src/demo/`
Them `src/__init__.py` (root src package).
### 1.4 `README.md`
Mo ta ngan: ten de tai, setup instructions (pip install, download data), project structure, cach chay.
### 1.5 Bat dau download data (chay song song voi code)
Tao 3 scripts trong `data/`:
* `data/download_gsc.py`: Download GSC v2 tu `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz` (~2.3GB), extract vao `data/gsc_v2/`
* `data/download_mswc.py`: Download MSWC English subset (763 words). Logic: fetch word list, filter top 500 + 263 eval words, download chi nhung words can.
* `data/convert_opus.py`: Convert MSWC OPUS -> WAV dung ffmpeg/torchaudio. Can thiet tren Windows.
**Bat dau chay download GSC + DEMAND ngay ngay 1** vi mat nhieu thoi gian.
## Day 2: MFCCExtractor (`src/features/mfcc.py`)
### Implementation
File: `src/features/mfcc.py`
```python
class MFCCExtractor:
    def __init__(self, n_mfcc=40, num_features=10, sample_rate=16000,
                 win_length_ms=40, hop_length_ms=20):
        # Tao torchaudio.transforms.MFCC
        # melkwargs: n_fft=1024, win_length=640, hop_length=320, n_mels=40
        # center=False (theo Rusci)
    def _pad_or_trim(self, waveform, target_length=16000):
        # Pad zeros right neu < target_length
        # Truncate right neu > target_length
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        # Input: (1, T) -> pad/trim -> MFCC(40) -> narrow(10) -> .mT -> (1, 49, 10)
    def extract_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        # Input: (B, 1, T) -> extract tung cai -> stack -> (B, 1, 49, 10)
```
Preprocessing flow chinh xac:
1. `self.mfcc_transform(waveform)` -> `(1, 40, 49)` (voi center=False)
2. `.narrow(dim=-2, start=0, length=self.num_features)` -> `(1, 10, 49)`
3. `.mT` -> `(1, 49, 10)`
Chi tiet quan trong:
* `center=False` trong melkwargs (Rusci dung `center=False`)
* mel_scale va norm: dung default cua torchaudio (torchaudio mac dinh `mel_scale='htk'`, Rusci dung `'slaney'` voi `norm='slaney'` — can kiem tra lai khi implement xem `torchaudio.transforms.MFCC` co ho tro `mel_scale` va `norm` khong, co the can dung `MelSpectrogram` rieng)
* Output phai deterministic (khong random)
* `extract_batch` co the dung vong for hoac vectorize
### Unit tests cho MFCC
File: `tests/test_mfcc.py`
* `test_mfcc_shape`: input `(1, 16000)` -> output `(1, 49, 10)`
* `test_mfcc_short_audio`: input `(1, 8000)` -> output `(1, 49, 10)` (auto pad)
* `test_mfcc_long_audio`: input `(1, 24000)` -> output `(1, 49, 10)` (auto truncate)
* `test_mfcc_batch`: input `(4, 1, 16000)` -> output `(4, 1, 49, 10)`
* `test_mfcc_num_features`: output last dim = 10
* `test_mfcc_deterministic`: 2 lan extract cung input -> cung output
## Day 3: NoiseAugmenter (`src/features/augmentation.py`)
### Implementation
File: `src/features/augmentation.py`
```python
class NoiseAugmenter:
    def __init__(self, noise_dir: Path, prob: float = 0.95, snr_db: float = 5.0):
        # Load tat ca WAV files tu noise_dir (DEMAND dataset)
        # Store list of noise file paths
    def _load_random_noise(self, target_length: int) -> torch.Tensor:
        # Chon random 1 noise clip
        # Loop neu noise < target_length, crop neu noise > target_length
    def _mix_snr(self, clean, noise, snr_db) -> torch.Tensor:
        # RMS-normalized mixing:
        # scale = rms(clean) / (rms(noise) * 10^(snr_db/20))
        # return clean + scale * noise
    def augment(self, waveform: torch.Tensor) -> torch.Tensor:
        # Random check prob -> neu pass, mix noise
        # Return original neu khong augment
```
Chi tiet:
* Fixed SNR = 5dB (khong phai range)
* RMS formula: `torch.sqrt(torch.mean(x ** 2))` + epsilon de tranh chia 0
* Noise duoc chon random moi lan augment
* CHI dung khi training, KHONG dung khi eval/inference
### Unit tests cho Augmentation
File: `tests/test_augmentation.py`
* `test_augment_prob_zero`: prob=0 -> output == input
* `test_augment_prob_one`: prob=1 -> output != input (khi co noise files)
* `test_augment_shape`: output shape == input shape
* `test_augment_snr`: Tinh SNR cua output, kiem tra xap xi 5dB
* Can co DEMAND data hoac tao mock noise files cho tests
## Day 4-5: DSCNN (`src/models/dscnn.py`)
### Implementation
File: `src/models/dscnn.py`
Day la file phuc tap nhat. Theo chinh xac Rusci `model_size_info_DSCNNL`:
```python
model_size_info_DSCNNL = [
    6, 276,
    10, 4, 2, 1,     # Initial conv: kernel=(10,4), stride=(2,1)
    276, 3, 3, 2, 2, # DS Block 1: 276ch, kernel=3x3, stride=(2,2)
    276, 3, 3, 1, 1, # DS Block 2: 276ch, kernel=3x3, stride=(1,1)
    276, 3, 3, 1, 1, # DS Block 3
    276, 3, 3, 1, 1, # DS Block 4
    276, 3, 3, 1, 1, # DS Block 5
]
```
Cau truc class:
```python
class DSBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride, use_layernorm=False):
        # DW: Conv2d(channels, channels, kernel, stride, groups=channels, padding=...)
        # BN after DW: BatchNorm2d(channels)
        # PW: Conv2d(channels, channels, 1)
        # After PW: BatchNorm+ReLU (blocks 1-4) OR LayerNorm (block 5)
    def forward(self, x):
        # DW -> BN -> ReLU -> PW -> (BN+ReLU or LayerNorm)
class DSCNN(nn.Module):
    def __init__(self, model_size='L', feature_mode='NORM'):
        # Parse model_size_info
        # Initial conv + BN + ReLU
        # 5 DS blocks (DSCNN-L) or 4 (DSCNN-S)
        # AvgPool global
        # self.embedding_dim = channels (276 or 64)
    def forward(self, x):
        # x: (B, 1, 49, 10)
        # Initial conv -> DS blocks -> AvgPool -> Flatten -> (B, 276)
        # L2-norm NOT applied here
```
DSCNN-L architecture chi tiet:
* Initial Conv2d(1, 276, kernel_size=(10,4), stride=(2,1), padding=SAME equiv) + BN + ReLU
    * Padding can tinh: `padding=((10-1)//2, (4-1)//2)` = `(4, 1)` hoac theo Rusci `(kernel-1)//2`
    * Kiem tra output shape sau initial conv: tu (B,1,49,10) -> cac dim giam theo stride
* DS Block 1: stride=(2,2) — spatial downsampling
* DS Blocks 2-5: stride=(1,1) — giu nguyen spatial size
* Block 5 khac biet: sau PW dung `LayerNorm(elementwise_affine=False)` thay vi BN+ReLU
    * Chu y: DW cua Block 5 VAN dung BN+ReLU, chi thay doi o sau PW
* AvgPool2d global: `nn.AdaptiveAvgPool2d(1)` -> (B, 276, 1, 1) -> flatten -> (B, 276)
* NO Linear projection layer
Padding calculation (quan trong):
* Cach Rusci tinh padding: `p_h = (kernel_h - 1) // 2`, `p_w = (kernel_w - 1) // 2`
* Initial conv (10,4) stride (2,1): padding (4,1)
    * H: (49 + 2*4 - 10) / 2 + 1 = (49+8-10)/2 + 1 = 47/2 + 1 = 24.5 -> can xem Rusci dung padding nao (co the padding=(5,2) hoac "same")
    * **Can doc lai source code Rusci de lay chinh xac cach tinh padding**
* DS blocks padding (3,3) stride khac nhau: padding=(1,1)
DSCNN-S (optional, cho ablation):
* 64 channels, 4 DS blocks, embedding_dim=64
* `model_size_info_DSCNNS = [5, 64, 10, 4, 2, 1, 64, 3, 3, 2, 2, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1]`
Feature mode:
* `CONV`: return raw output sau AvgPool+Flatten
* `RELU`: return output sau ReLU (last block)
* `NORM`: caller se apply F.normalize() externally — model van return raw output
* Ca 3 mode deu return raw output tu model, su khac biet nam o cach training/inference wrapper su dung
### Unit tests cho DSCNN
File: `tests/test_dscnn.py`
* `test_dscnn_l_output_shape`: input (4,1,49,10) -> output (4,276)
* `test_dscnn_s_output_shape`: input (4,1,49,10) -> output (4,64)
* `test_dscnn_l_l2_norm_external`: F.normalize(output) co norm=1
* `test_dscnn_l_param_count`: in ra param count de verify (khong assert cu the, chi log)
* `test_dscnn_forward_backward`: chay forward + backward, dam bao gradients khong None
* `test_dscnn_feature_modes`: test ca 3 modes "CONV", "RELU", "NORM" deu chay duoc
## Day 6: Integration Test + Milestone
### End-to-end pipeline test
File: `tests/test_pipeline.py`
```python
def test_full_pipeline():
    # 1. Tao random waveform (hoac load 1 file WAV thuc)
    wav = torch.randn(1, 16000)
    # 2. Extract MFCC
    extractor = MFCCExtractor()
    mfcc = extractor.extract(wav)
    assert mfcc.shape == (1, 49, 10)
    # 3. Add batch dim cho DSCNN
    mfcc_batch = mfcc.unsqueeze(0)  # (1, 1, 49, 10)
    # 4. Chay DSCNN
    model = DSCNN(model_size='L')
    embedding = model(mfcc_batch)
    assert embedding.shape == (1, 276)
    # 5. L2 normalize externally
    embedding_norm = F.normalize(embedding, p=2, dim=-1)
    assert torch.allclose(embedding_norm.norm(dim=-1), torch.ones(1), atol=1e-5)
```
### Milestone criteria
* `pytest tests/` passes tat ca tests
* Pipeline `WAV (1,16000) -> MFCC (1,49,10) -> DSCNN-L (1,276) -> L2-norm (1,276)` chay thanh cong
* Data download scripts ton tai va chay duoc (it nhat GSC script)
* `configs/default.yaml` co day du hyperparameters
* Tat ca `__init__.py` da tao
## Day 7: Buffer / Fix bugs / Bat dau download MSWC
* Fix bat ky test failures nao
* Hoan thien data download scripts neu chua xong
* Bat dau download MSWC (co the mat 1-2 ngay)
* Review code quality: type hints, docstrings, error handling
* Chuan bi cho Week 2 (doc truoc ve Triplet Loss, EpisodicBatchSampler)
## Rui ro Week 1
1. **Padding DSCNN**: Cach Rusci tinh padding co the khac voi standard PyTorch. Can doc lai source code chinh xac truoc khi implement.
2. **MFCC center=False**: torchaudio MFCC default `center=True`. Phai truyen `center=False` trong melkwargs.
3. **MSWC download cham**: Bat dau som, chay background. Week 1 chi can GSC + DEMAND de test.
4. **torchaudio MelSpectrogram params**: Rusci dung `mel_scale='slaney'`, `norm='slaney'`. Kiem tra torchaudio co support khong (co tu torchaudio 0.12+).
5. **OPUS tren Windows**: Cai ffmpeg truoc khi chay `convert_opus.py`. Test `torchaudio.load('test.opus')` som.
## Files se tao trong Week 1
```warp-runnable-command
requirements.txt
configs/default.yaml
README.md
src/__init__.py
src/features/__init__.py
src/models/__init__.py
src/classifiers/__init__.py
src/evaluation/__init__.py
src/streaming/__init__.py
src/enhancements/__init__.py
src/demo/__init__.py
src/features/mfcc.py
src/features/augmentation.py
src/models/dscnn.py
data/download_gsc.py
data/download_mswc.py
data/convert_opus.py
tests/test_mfcc.py
tests/test_augmentation.py
tests/test_dscnn.py
tests/test_pipeline.py
tests/__init__.py
```
