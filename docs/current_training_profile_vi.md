# Current Training Profile

Ngay 2026-05-15, workflow train khuyen nghi tam thoi cua project la:

```text
CURRENT DATA PROFILE = MSWC MICROSET ENGLISH
TEMPORARY RUN FOR UNIT/DISK SAVING
NOT TOP500 FULL
NOT FULL MSWC
NOT EDGESPOT PAPER REPRODUCTION
```

## Trang Thai Hien Tai

- Dataset train hien tai: `data/mswc_microset_en`.
- Nguon dataset: official MLCommons MSWC Microset English.
- Split hien tai: doc CSV chinh thuc cua Microset theo sample-level split:
  `en_train.csv`, `en_dev.csv`, `en_test.csv`. Ca 3 split deu co 31 keywords,
  nhung file audio khac nhau. Train dung `train_files.json`, khong quet toan bo
  folder `clips/<word>`.
- Muc dich: tiet kiem Colab units va dung luong Drive/disk trong khi chua co may train on dinh.
- Workflow chay: Colab notebook trong + copy tung cell lenh tu runbook.
- Runbook chinh: `docs/colab_microset_runbook_vi.md`.
- Bao cao ket qua Colab hien tai: `docs/colab_microset_experiment_report_vi.md`.
- Pipeline tiep theo cho Top500 full: `docs/colab_top500_full_runbook_vi.md`.
- Runtime tam thoi: A100 khuyen nghi, G4 chap nhan neu tiet kiem units; khong dung H100.

## Khong Duoc Claim

- Khong claim day la Top500 full clips/word.
- Khong claim day la full MSWC English.
- Khong claim reproduce EdgeSpot paper bang ket qua Microset.
- Khong dung Microset result de so truc tiep voi paper neu paper dung data lon hon.

## Huong Sau Nay

Khi co may truong, disk, hoac Drive du on dinh:

- Tao runbook rieng cho Top500 full.
- Dung MSWC English Top500 full clips/word.
- Dat run tag co hau to `_top500_full_v1`.
- Chay lai EdgeSpotFull va benchmark GSC exact.

## Ghi Nho Cho Lan Sau

- Khong sua tiep `notebooks/02_train_enhanced.ipynb` de train chinh.
- `02_train_enhanced.ipynb` duoc xem la legacy/experimental.
- Colab chi la moi truong GPU; logic that nam trong script repo:
  - `data/download_mswc_microset.py`
  - `scripts/train.py`
  - `scripts/evaluate_edgespot_protocol.py`
  - `scripts/mswc_data_report.py`
