from __future__ import annotations

from pathlib import Path

import make_complete_thesis_vi_2026_06_13 as base


ROOT = base.ROOT
OUT_DIR = ROOT / "docs" / "thesis"
OUT_MD = OUT_DIR / "Do_An_KWS_thesis_reference_style_vi_2026_06_14.md"
OUT_DOCX = OUT_DIR / "Do_An_KWS_thesis_reference_style_vi_2026_06_14.docx"


def h(level: int, text: str) -> base.Block:
    return base.h(level, text)


def p(text: str) -> base.Block:
    return base.p(text)


def bullets(items: list[str]) -> base.Block:
    return base.bullets(items)


def tbl(caption: str, headers: list[str], rows: list[list[str]]) -> base.Block:
    return base.tbl(base.simple_table(caption, headers, rows))


def table_block(table: base.TableBlock) -> base.Block:
    return base.tbl(table)


def fig(path: Path, caption: str) -> base.Block:
    return base.fig(path, caption)


def formula(text: str) -> base.Block:
    return p("$$\n" + text.strip() + "\n$$")


def title_blocks() -> list[base.Block]:
    return [
        h(0, "Few-Shot Open-Set Keyword Spotting dựa trên Embedding và Prototype Inference"),
        p(
            "Bản thesis này được tái cấu trúc ngày 2026-06-14 theo hướng của reference master: "
            "Giới thiệu, Nền tảng, Phương pháp luận, Kết quả & Thảo luận, Kết luận & Hướng phát triển. "
            "Mục tiêu là biến nội dung dự án từ dạng nhật ký kỹ thuật thành một luận văn khoa học có logic rõ ràng."
        ),
        h(1, "Thông tin báo cáo"),
        tbl(
            "Bảng 0. Thông tin cần điền trước khi nộp.",
            ["Trường", "Giá trị"],
            [
                ["Tên đề tài", "Few-Shot Open-Set Keyword Spotting dựa trên Embedding và Prototype Inference"],
                ["Sinh viên", "[Điền họ tên sinh viên]"],
                ["Mã sinh viên", "[Điền mã sinh viên]"],
                ["Giảng viên hướng dẫn", "[Điền tên giảng viên hướng dẫn]"],
                ["Đơn vị", "[Điền khoa/trường]"],
                ["Ngày", "2026-06-14"],
            ],
        ),
        h(1, "Lời cam đoan"),
        p(
            "Em xin cam đoan nội dung trong báo cáo này phản ánh quá trình thực hiện đồ án của bản thân, "
            "các kết quả thực nghiệm được trích từ log, checkpoint, result JSON và báo cáo trong dự án. "
            "Các tài liệu tham khảo, paper và mã nguồn bên ngoài cần được chuẩn hóa citation theo quy định của nhà trường trước khi nộp."
        ),
        h(1, "Lời cảm ơn"),
        p(
            "Em xin gửi lời cảm ơn tới giảng viên hướng dẫn, các thầy cô và các anh chị đã hỗ trợ về chuyên môn, "
            "tài nguyên tính toán, server, Google Colab và phản hồi kỹ thuật trong quá trình thực hiện đồ án. "
            "Dự án yêu cầu nhiều vòng thử nghiệm với dữ liệu lớn, vì vậy sự hỗ trợ về GPU, lưu trữ và định hướng nghiên cứu có vai trò rất quan trọng."
        ),
        h(1, "Tóm tắt"),
        p(
            "Đồ án nghiên cứu bài toán few-shot open-set keyword spotting, trong đó hệ thống phải nhận diện các từ khóa mới "
            "từ một số ít mẫu enrollment và đồng thời từ chối các âm thanh không thuộc tập từ khóa đã đăng ký. "
            "Khác với KWS closed-set, hệ thống phải kiểm soát false acceptance rate vì false accept có thể kích hoạt sai lệnh thoại."
        ),
        p(
            "Hướng tiếp cận của đồ án là embedding-based KWS. Audio được chuẩn hóa, trích xuất MFCC hoặc mel-PCEN, "
            "đưa qua encoder để tạo embedding L2-normalized. Với mỗi keyword, prototype được tính bằng trung bình embedding "
            "của support samples. Query được so với các prototype bằng khoảng cách L2 và được accept/reject theo threshold tại target FAR."
        ),
        p(
            "Thí nghiệm chính so sánh hai backbone DSCNN-L và EdgeSpotFull T4, hai frontend MFCC và PCEN, "
            "cùng bốn objective Triplet, SCAF, GE2E và SCAF+GE2E. Fixed 16-pipeline cap620 tạo bảng ablation sạch, "
            "sau đó development run tăng episode budget và chọn checkpoint theo composite metric ACC@1%FAR, AUC và F1."
        ),
        p(
            "Kết quả mới nhất cho thấy DSCNN-L + PCEN + GE2E đạt 86.36 ± 1.29% ACC@1%FAR trên GSC test100, "
            "là cấu hình accuracy-oriented tốt nhất. EdgeSpotFull T4 + PCEN + GE2E đạt 82.87 ± 1.22% ACC@1%FAR, "
            "là cấu hình compact tốt nhất và cạnh tranh với mốc EdgeSpot-4 paper, nhưng chưa phải reproduction đầy đủ vì chưa chạy KD."
        ),
        h(1, "Abstract"),
        p(
            "This thesis studies few-shot open-set keyword spotting, where a system must recognize newly enrolled keywords "
            "from a small number of support examples while rejecting non-enrolled speech. The proposed system uses "
            "audio preprocessing, MFCC or mel-PCEN frontends, neural embedding encoders, prototype inference and "
            "threshold-based open-set rejection."
        ),
        p(
            "The experiments compare DSCNN-L and EdgeSpotFull T4 backbones, MFCC and PCEN frontends, and Triplet, SCAF, "
            "GE2E and SCAF+GE2E objectives. The latest cap620 development run reaches 86.36 ± 1.29% ACC@1%FAR with "
            "DSCNN-L + PCEN + GE2E and 82.87 ± 1.22% with EdgeSpotFull T4 + PCEN + GE2E."
        ),
        h(1, "Danh mục thuật ngữ viết tắt"),
        tbl(
            "Bảng 1. Thuật ngữ viết tắt.",
            ["Ký hiệu", "Ý nghĩa"],
            [
                ["KWS", "Keyword Spotting, phát hiện từ khóa trong audio"],
                ["GSC", "Google Speech Commands v2, benchmark evaluation chính"],
                ["MSWC", "Multilingual Spoken Words Corpus, nguồn training chính trong đồ án"],
                ["MFCC", "Mel-Frequency Cepstral Coefficients"],
                ["PCEN", "Per-Channel Energy Normalization"],
                ["FAR", "False Acceptance Rate"],
                ["FRR", "False Rejection Rate"],
                ["EER", "Equal Error Rate"],
                ["AUC", "Area Under ROC Curve"],
                ["DET", "Detection Error Tradeoff"],
                ["GE2E", "Generalized End-to-End loss"],
                ["SCAF", "Sub-center ArcFace-style loss"],
                ["KD", "Knowledge Distillation"],
            ],
        ),
        h(1, "Mục lục"),
        p(
            "Trong file Word, mục lục được chèn bằng field TOC. Khi mở bằng Microsoft Word, chọn mục lục và nhấn Update Field để cập nhật số trang."
        ),
    ]


def chapter_1() -> list[base.Block]:
    return [
        h(1, "Chương 1. Giới thiệu"),
        h(2, "1.1. Bối cảnh và động lực"),
        p(
            "Keyword Spotting là bài toán phát hiện sự xuất hiện của một hoặc nhiều từ khóa trong luồng âm thanh. "
            "Trong các hệ thống trợ lý giọng nói, điều khiển thiết bị, smart home hoặc thiết bị nhúng, tầng KWS thường là lớp đầu tiên "
            "quyết định liệu hệ thống có cần phản hồi hay không. Vì chạy ở tuyến đầu, KWS cần vừa nhanh, vừa ổn định, vừa hạn chế kích hoạt sai."
        ),
        p(
            "Các hệ thống KWS truyền thống thường được mô hình hóa như closed-set classification với vocabulary cố định. "
            "Cách tiếp cận này không phù hợp khi người dùng muốn thêm keyword mới bằng một vài mẫu giọng nói cá nhân. "
            "Few-shot learning giải quyết vấn đề này bằng cách học embedding space, nơi class mới có thể được biểu diễn bởi prototype "
            "mà không cần train lại classifier cuối."
        ),
        p(
            "Bên cạnh khả năng nhận keyword mới, hệ thống còn cần open-set recognition: nếu âm thanh không thuộc keyword đã enroll, "
            "hệ thống phải trả về unknown. Đây là phần quan trọng vì mọi query đều có prototype gần nhất; nếu không có threshold reject, "
            "unknown speech sẽ bị ép thành một keyword hợp lệ."
        ),
        h(2, "1.2. Mục tiêu"),
        bullets(
            [
                "Xây dựng pipeline few-shot open-set KWS dựa trên embedding, prototype inference và threshold rejection.",
                "So sánh có hệ thống backbone, frontend và loss function trong cùng protocol.",
                "Làm rõ vai trò của PCEN, GE2E, Triplet và SCAF trong không gian embedding.",
                "Đánh giá bằng GSC `gsc_edgespot_exact` dev30/test100 ở FAR 1% và FAR 5%.",
                "So sánh thận trọng với EdgeSpot-4 paper, tách rõ result chính, ablation, demo và future work.",
            ]
        ),
        h(2, "1.3. Câu hỏi nghiên cứu"),
        tbl(
            "Bảng 2. Câu hỏi nghiên cứu.",
            ["ID", "Câu hỏi", "Bằng chứng chính"],
            [
                ["RQ1", "PCEN có cải thiện so với MFCC trong few-shot open-set KWS không?", "Fixed 16-pipeline cap620"],
                ["RQ2", "GE2E có phù hợp hơn với prototype inference so với Triplet/SCAF không?", "Fixed 16-pipeline và development run"],
                ["RQ3", "DSCNN-L và EdgeSpotFull T4 đánh đổi accuracy/compactness như thế nào?", "Bảng kết quả test100 và số tham số"],
                ["RQ4", "Compact model của project đã cạnh tranh với EdgeSpot-4 chưa?", "So sánh ACC@1%FAR với mốc 82.0%"],
                ["RQ5", "Demo UI có thể dùng làm bằng chứng benchmark không?", "Audit demo và phân biệt với GSC test100"],
            ],
        ),
        h(2, "1.4. Đóng góp"),
        bullets(
            [
                "Xây dựng pipeline few-shot open-set KWS end-to-end từ dữ liệu, training, evaluation đến demo.",
                "Thực hiện fixed 16-pipeline cap620 để so sánh backbone, frontend và objective trong cùng điều kiện.",
                "Phát triển run tăng budget và composite checkpoint selection, đưa DSCNN-L + PCEN + GE2E lên 86.36% ACC@1%FAR.",
                "Đưa EdgeSpotFull T4 + PCEN + GE2E lên 82.87% ACC@1%FAR, cạnh tranh với EdgeSpot-4 nhưng giữ claim không overstate.",
                "Phân tích failure modes của SCAF collapse và hard-triplet collapse để định hướng ablation tiếp theo.",
            ]
        ),
        h(2, "1.5. Cấu trúc báo cáo"),
        p(
            "Chương 2 trình bày nền tảng về ProtoNet, open-set KWS, audio features, loss functions, dataset và metrics. "
            "Chương 3 mô tả phương pháp luận của dự án theo hướng reference: phân tích baseline, direct L2 decision, data utilization, "
            "MSWC/GSC protocol và thiết kế thực nghiệm. Chương 4 trình bày kết quả và thảo luận. Chương 5 kết luận và nêu hướng phát triển."
        ),
    ]


def chapter_2() -> list[base.Block]:
    return [
        h(1, "Chương 2. Nền tảng"),
        h(2, "2.1. Few-shot open-set keyword spotting"),
        p(
            "Few-shot KWS giả định rằng ở thời điểm triển khai, mỗi keyword chỉ có một số ít support samples. "
            "Thay vì học classifier cố định, hệ thống học encoder `f_theta` để ánh xạ audio vào embedding space. "
            "Một keyword mới được biểu diễn bởi prototype, còn query được phân loại bằng khoảng cách đến prototype."
        ),
        h(2, "2.2. Prototypical Network và episodic training"),
        p("Gọi `f_theta: R^D -> R^M` là encoder biến feature audio thành embedding M chiều. Với support set `S_k` của class `k`, prototype được tính như sau:"),
        formula(r"c_k = \frac{1}{|S_k|}\sum_{(x_i,y_i)\in S_k} f_\theta(x_i)"),
        p("Baseline ProtoNet dùng softmax trên khoảng cách âm để tính xác suất class của query:"),
        formula(
            r"p_\theta(y=k|x_q)=\frac{\exp(-d(f_\theta(x_q),c_k))}{\sum_{k'}\exp(-d(f_\theta(x_q),c_{k'}))}"
        ),
        p("Trong episodic training, mỗi episode chọn `N` class và `K` support samples mỗi class. Query set được dùng để tính loss."),
        formula(
            r"C_e \subset C_{train},\ |C_e|=N"
            "\n"
            r"S_e=\bigcup_{c\in C_e}\{(x_i^c,c)\}_{i=1}^{K}"
            "\n"
            r"Q_e=\bigcup_{c\in C_e}\{(x_j^c,c)\}_{j=K+1}^{K+Q}"
        ),
        formula(
            r"L_e=-\frac{1}{NQ}\sum_{(x_q,y_q)\in Q_e}\log p_\theta(y=y_q|x_q)"
        ),
        h(2, "2.3. Workflow hệ thống"),
        p(
            "Workflow few-shot open-set KWS gồm ba pha. Pha training học encoder trên source dataset lớn, trong đồ án là MSWC English. "
            "Pha enrollment dùng một số mẫu support để tính prototype cho từng keyword. Pha inference encode query, so sánh với prototype, "
            "rồi accept hoặc reject bằng threshold."
        ),
        formula(
            r"z=\operatorname{normalize}(f_\theta(x))"
            "\n"
            r"p_c=\operatorname{normalize}\left(\frac{1}{K}\sum_{i=1}^{K}f_\theta(x_i^c)\right)"
            "\n"
            r"d_c(x)=\|z-p_c\|_2"
            "\n"
            r"c^*=\arg\min_c d_c(x)"
        ),
        h(2, "2.4. Audio processing: MFCC và PCEN"),
        p("MFCC là baseline truyền thống trong speech processing. Chuỗi tính toán gồm pre-emphasis, framing/windowing, STFT, power spectrum, mel filterbank, log và DCT."),
        formula(
            r"x[n]=y[n]-\alpha y[n-1]"
            "\n"
            r"x_m[n]=x[n+mH]"
            "\n"
            r"x_m[n]=x_m[n]w[n]"
            "\n"
            r"X_m[k]=\sum_{n=0}^{N-1}x_m[n]e^{-j2\pi kn/N}"
            "\n"
            r"P_m[k]=|X_m[k]|^2"
            "\n"
            r"S_m[r]=\sum_k P_m[k]H_r[k]"
            "\n"
            r"C[q]=\sum_r \log(S_m[r])\cos\left(\frac{\pi q(2r+1)}{2R}\right)"
        ),
        p("PCEN thay log compression bằng adaptive gain control theo từng kênh mel, giúp ổn định hơn trước biến thiên âm lượng và noise."),
        formula(
            r"M(t,f)=(1-s)M(t-1,f)+sE(t,f)"
            "\n"
            r"PCEN(t,f)=\left(\frac{E(t,f)}{(\epsilon+M(t,f))^\alpha}+\delta\right)^r-\delta^r"
        ),
        h(2, "2.5. Encoder: DSCNN-L và EdgeSpotFull T4"),
        p(
            "DSCNN-L dùng depthwise separable convolution để giảm chi phí so với convolution thường. "
            "EdgeSpotFull T4 là compact encoder lấy cảm hứng từ EdgeSpot-style KWS, dùng mel-PCEN và temporal blocks để tạo embedding 64 chiều."
        ),
        formula(
            r"Y^{(c)}=X^{(c)} * K^{(c)}"
            "\n"
            r"Z_k=\sum_{c=1}^{C}Y^{(c)} * W_k^{(c)}"
        ),
        h(2, "2.6. Loss functions"),
        p("Triplet loss tối ưu khoảng cách anchor-positive nhỏ hơn anchor-negative ít nhất một margin `m`:"),
        formula(r"L_{triplet}=\frac{1}{N_t}\sum_i \max(0,d(a_i,p_i)-d(a_i,n_i)+m)"),
        p("GE2E tính centroid từ support trong episode rồi dùng cosine logits cho query classification:"),
        formula(
            r"c_k=\operatorname{normalize}\left(\frac{1}{|S_k|}\sum_{x_i\in S_k}z_i\right)"
            "\n"
            r"l_{q,k}=w\cos(z_q,c_k)+b"
            "\n"
            r"L_{GE2E}=CE(\operatorname{softmax}(l_q),y_q)"
        ),
        p("SCAF/Sub-center ArcFace dùng nhiều sub-center cho mỗi class và angular margin để tăng tách biệt class trong embedding space:"),
        formula(
            r"\cos(\theta_j)=\max_{r=1..K}\langle \operatorname{normalize}(z),\operatorname{normalize}(W_{j,r})\rangle"
            "\n"
            r"\phi_y=\cos(\theta_y+m)"
            "\n"
            r"logit_y=s\phi_y,\quad logit_j=s\cos(\theta_j), j\ne y"
            "\n"
            r"L_{SCAF}=CE(logits,y)"
        ),
        h(2, "2.7. Dataset"),
        tbl(
            "Bảng 3. Vai trò dataset.",
            ["Dataset", "Vai trò trong đồ án", "Ghi chú"],
            [
                ["MSWC English", "Training chính", "Tạo train/val words, manifest cap20/cap220/cap620 và train encoder bằng episodic sampling."],
                ["GSC v2", "Evaluation chính", "Dùng `gsc_edgespot_exact` dev30/test100 với support/query split rõ ràng."],
                ["DEMAND", "Noise augmentation", "Trộn noise trong training để tăng robustness."],
                ["MSWC eval words", "Evaluation bổ sung", "Code hỗ trợ 5 positive/50 negative và 1:9 support/query, nhưng chưa là evidence chính."],
            ],
        ),
        h(2, "2.8. Metrics"),
        p("Open-set KWS cần metric vừa đo khả năng nhận keyword thật, vừa đo khả năng reject unknown. Các định nghĩa cơ bản:"),
        formula(
            r"TPR=\frac{TP}{TP+FN}"
            "\n"
            r"FPR=\frac{FP}{FP+TN}"
            "\n"
            r"FAR=FPR"
            "\n"
            r"FRR=1-TPR=\frac{FN}{TP+FN}"
        ),
        p("Threshold tại target FAR `alpha` được chọn để false accept không vượt `alpha`:"),
        formula(
            r"\tau_\alpha=\max\{\tau: FAR(\tau)\le \alpha\}"
            "\n"
            r"ACC@\alpha FAR=\frac{\#positive\ accepted\ đúng\ class+\#negative\ rejected}{\#all\ queries}"
        ),
        p("DET curve biểu diễn trade-off `(FAR(tau), FRR(tau))` khi thay đổi threshold. Mean DET nội suy các curve về cùng trục FAR rồi trung bình FRR."),
    ]


def chapter_3() -> list[base.Block]:
    return [
        h(1, "Chương 3. Phương pháp luận và thiết kế thực nghiệm"),
        h(2, "3.1. Phân tích hướng tham khảo"),
        p(
            "Reference master tập trung cải tiến giao thức đánh giá few-shot open-set KWS: so sánh probability-based scoring với direct L2, "
            "tính mean DET, random hóa word partitions và tích hợp MSWC eval để vượt giới hạn 35 từ của GSC. "
            "Dự án này kế thừa phần cốt lõi là embedding/prototype inference, direct L2 scoring và open-set threshold, "
            "nhưng mở rộng trọng tâm sang so sánh backbone/frontend/loss trên MSWC cap620."
        ),
        tbl(
            "Bảng 4. Mapping giữa reference master và dự án.",
            ["Thành phần", "Reference master", "Dự án này"],
            [
                ["Training source", "MSWC top500", "MSWC English cap620 FLAC, cap220/top500/microset trong lịch sử"],
                ["Main evaluation", "GSC fixed/random và MSWC randomized", "GSC `gsc_edgespot_exact` dev30/test100 là evidence chính"],
                ["Scoring", "Direct L2 thay probability normalization", "Direct L2 là default; probability/openmax/energy có trong code để ablation"],
                ["MSWC eval", "5 positive / 50 negative, 1:9 split", "Có code hỗ trợ; chưa dùng làm bảng final chính"],
                ["Main comparison", "Evaluation protocol improvements", "Backbone/frontend/loss ablation và EdgeSpot-4 comparison"],
            ],
        ),
        h(2, "3.2. Direct L2 decision"),
        p("Dự án dùng score trực tiếp từ khoảng cách L2 thay vì ép toàn bộ khoảng cách qua softmax probability. Với query `x`:"),
        formula(
            r"s(x)=-\min_c d_c(x)"
            "\n"
            r"\hat{y}=\begin{cases}c^*, & s(x)\ge \tau_\alpha\\ unknown, & s(x)<\tau_\alpha\end{cases}"
        ),
        p(
            "Cách diễn giải này trực quan: threshold là bán kính chấp nhận quanh prototype trong embedding space. "
            "Nếu query quá xa mọi prototype, nó bị reject thành unknown."
        ),
        h(2, "3.3. GSC `gsc_edgespot_exact` protocol"),
        p(
            "Evaluation chính dùng 10 command words của GSC cộng `_silence_` làm positive targets. "
            "Negative là 25 spoken words còn lại ngoài 10 command target. Mỗi run lấy `k=10` support samples cho mỗi target để tạo prototype, "
            "rồi đánh giá query samples."
        ),
        tbl(
            "Bảng 5. Cấu hình GSC exact protocol.",
            ["Thành phần", "Giá trị"],
            [
                ["Positive target", "yes, no, up, down, left, right, on, off, stop, go, _silence_"],
                ["Negative words", "25 spoken words còn lại: digits, bed/bird/cat/.../visual"],
                ["Support pool", "`validation_list.txt` của GSC"],
                ["Query pool", "`testing_list.txt` cho test; train files chính thức cho dev"],
                ["Silence", "Crop 1 giây từ `_background_noise_`"],
                ["Final dev", "30 repeated runs"],
                ["Final test", "100 repeated runs"],
            ],
        ),
        h(2, "3.4. Tích hợp MSWC trong dự án"),
        p(
            "Khác với reference đoạn 3.2.4, bằng chứng chính hiện tại của dự án không dùng MSWC làm final evaluation mà dùng MSWC làm source training lớn. "
            "Cap620 FLAC có khoảng 2.99 triệu train files, 52k validation files, 37,387 train words và 763 validation words. "
            "Model học embedding trên MSWC rồi được evaluate cross-dataset trên GSC."
        ),
        p(
            "Repo vẫn có nhánh MSWC randomized evaluation tương ứng reference: `EvaluationProtocol(dataset='mswc')` chọn 5 positive và 50 negative, "
            "`MSWCFewShotProvider` chia support/query theo tỷ lệ 1:9. Tuy nhiên, khi viết thesis hiện tại, nhánh này nên được mô tả là bổ sung/future work "
            "nếu chưa có bảng result test100 đầy đủ."
        ),
        h(2, "3.5. Ma trận thực nghiệm"),
        p("Fixed cap620 chạy 16 pipeline từ tích Descartes của 2 backbone, 2 frontend và 4 loss/objective."),
        tbl(
            "Bảng 6. Không gian cấu hình 16 pipeline.",
            ["Trục", "Giá trị"],
            [
                ["Backbone", "DSCNN-L, EdgeSpotFull T4"],
                ["Frontend", "MFCC, PCEN"],
                ["Loss/objective", "Triplet, SCAF, GE2E, SCAF+GE2E"],
                ["Fixed run", "40 epochs, 150 episodes/epoch, 30 classes x 10 samples"],
                ["Development run", "60 epochs, 300 episodes/epoch, composite checkpoint selection"],
            ],
        ),
        h(2, "3.6. Checkpoint selection"),
        p(
            "Fixed run chọn checkpoint theo GSC-dev ACC@1%FAR. Development run dùng composite metric để giảm rủi ro chọn checkpoint chỉ tốt tại một operating point:"
        ),
        formula(r"Composite=\frac{ACC@1\%FAR + AUC + F1}{3}"),
        h(2, "3.7. Thiết kế so sánh với EdgeSpot-4"),
        p(
            "Mốc EdgeSpot-4 paper được dùng như reference bên ngoài: 82.0% ACC@1%FAR, khoảng 128k parameters và 29.4M MACs. "
            "Dự án so sánh thận trọng vì không reproduce đầy đủ recipe của paper, đặc biệt development run chưa bật KD."
        ),
    ]


def chapter_4(rows: list[dict]) -> list[base.Block]:
    out: list[base.Block] = [
        h(1, "Chương 4. Kết quả và thảo luận"),
        h(2, "4.1. Tổng quan fixed 16-pipeline"),
        p(
            "Fixed 16-pipeline là evidence ablation sạch nhất vì tất cả cấu hình dùng cùng data profile, cùng training budget, "
            "cùng checkpoint selection và cùng final evaluation. Bảng top pipeline cho thấy tác động rõ của PCEN và GE2E."
        ),
    ]
    if rows:
        out.extend(
            [
                table_block(base.top_table(rows, "test100_far1", 8, "Bảng 7. Top 8 pipeline fixed cap620 tại test100 FAR=1%.")),
                table_block(base.fixed_table(rows, "test100_far1", "Bảng 8. Đầy đủ 16 pipeline tại test100 FAR=1%.")),
                table_block(base.fixed_table(rows, "test100_far5", "Bảng 9. Đầy đủ 16 pipeline tại test100 FAR=5%.")),
                table_block(base.delta_table(rows)),
            ]
        )
    else:
        out.append(p("Không tìm thấy CSV fixed cap620, vì vậy bảng fixed 16-pipeline không được sinh trong lần này."))

    out.extend(
        [
            h(2, "4.2. Development run cap620"),
            table_block(base.development_table()),
            p(
                "Development run cải thiện mạnh so với fixed run. DSCNN-L + PCEN + GE2E tăng lên 86.36% ACC@1%FAR, "
                "AUC 95.21%, EER 11.32% và F1 82.73%. EdgeSpotFull T4 + PCEN + GE2E tăng lên 82.87% ACC@1%FAR, "
                "AUC 92.41%, EER 14.82% và F1 77.85%."
            ),
            h(2, "4.3. Vì sao PCEN tốt hơn MFCC"),
            p(
                "PCEN làm ổn định năng lượng theo từng kênh mel, giảm ảnh hưởng của volume, speaker và background. "
                "Trong cross-dataset setting MSWC -> GSC, điều này giúp khoảng cách embedding phản ánh nội dung từ khóa thay vì điều kiện thu."
            ),
            h(2, "4.4. Vì sao GE2E phù hợp với prototype inference"),
            p(
                "GE2E huấn luyện theo centroid trong episode, còn inference cũng dùng prototype/centroid. "
                "Sự khớp giữa training objective và inference mechanism giải thích vì sao GE2E ổn định nhất trong cap620."
            ),
            h(2, "4.5. SCAF collapse"),
            p(
                "Nhiều cấu hình SCAF hoặc SCAF+GE2E bị collapse với AUC khoảng 50%, EER khoảng 50%, FRR@FAR 100% và F1 bằng 0. "
                "Điều này không chứng minh SCAF sai về bản chất; nó cho thấy scale, margin, loss weight và warmup hiện tại chưa phù hợp với vocabulary lớn."
            ),
            h(2, "4.6. Triplet và hard mining"),
            p(
                "Triplet vẫn có giá trị trong fixed run, đặc biệt với EdgeSpotFull T4 + PCEN. Tuy nhiên development branch Triplet hard bị collapse, "
                "cho thấy hard mining quá gắt có thể phá cấu trúc embedding. Cần ablation với semi-hard mining hoặc hard-pair probability thấp hơn."
            ),
            h(2, "4.7. So sánh với EdgeSpot-4"),
            tbl(
                "Bảng 10. So sánh với mốc EdgeSpot-4 paper.",
                ["Hệ thống", "Nguồn/profile", "Kích thước", "ACC@1%FAR", "Nhận xét"],
                [
                    ["EdgeSpot-4 paper", "Paper EdgeSpot", "128k params, 29.4M MACs", "82.0%", "Mốc công bố, không phải result chạy lại trong repo."],
                    ["DSCNN-L + PCEN + GE2E", "Project cap620 development", "~412.9k params", "86.36 ± 1.29%", "Vượt mean paper rõ nhưng model lớn hơn."],
                    ["EdgeSpotFull T4 + PCEN + GE2E", "Project cap620 development", "~130.6k params", "82.87 ± 1.22%", "Cạnh tranh và nhỉnh hơn mean paper, nhưng margin nhỏ và chưa KD."],
                    ["EdgeSpotFull T4 + PCEN + GE2E", "Project cap620 fixed", "~130.6k params", "79.98 ± 0.98%", "Fixed ablation trước development, chưa vượt paper."],
                ],
            ),
            h(2, "4.8. Demo system trong phạm vi luận văn"),
            p(
                "Demo UI minh họa enrollment, single detection, long-audio analysis, open-set sampled evaluation và calibration. "
                "Tuy nhiên demo chỉ là công cụ triển khai/debug; benchmark chính vẫn là GSC test100. "
                "Per-class threshold và close-word guard nên được trình bày là tính năng experimental, không phải bằng chứng khoa học chính."
            ),
        ]
    )
    for path, caption in base.FIGURES:
        if path.exists():
            out.append(fig(path, caption))
    return out


def chapter_5() -> list[base.Block]:
    return [
        h(1, "Chương 5. Kết luận và hướng phát triển"),
        h(2, "5.1. Kết luận"),
        p(
            "Đồ án đã xây dựng và đánh giá pipeline few-shot open-set KWS dựa trên embedding, prototype inference và threshold rejection. "
            "Bản thesis sau khi tái cấu trúc theo reference đã tách rõ nền tảng, phương pháp luận, thiết kế thực nghiệm, kết quả và claim hợp lệ."
        ),
        p(
            "Kết quả chính hiện tại là DSCNN-L + PCEN + GE2E đạt 86.36 ± 1.29% ACC@1%FAR, còn EdgeSpotFull T4 + PCEN + GE2E đạt 82.87 ± 1.22%. "
            "PCEN và GE2E là hai thành phần ổn định nhất. SCAF cần ablation riêng trước khi dùng trên vocabulary lớn. "
            "EdgeSpotFull T4 đã cạnh tranh với mốc EdgeSpot-4 nhưng chưa nên claim reproduction đầy đủ."
        ),
        h(2, "5.2. Hướng phát triển"),
        bullets(
            [
                "Chạy KD hoặc teacher-guided objective cho EdgeSpotFull T4 để so sánh công bằng hơn với EdgeSpot paper.",
                "Chạy SCAF ablation với scale/margin/weight/warmup và subset nhỏ trước khi quay lại full cap620.",
                "Chạy MSWC randomized evaluation 5-positive/50-negative nếu muốn bám sát hơn mục 3.2.4 của reference.",
                "Thử semi-hard Triplet thay vì hard mining quá gắt.",
                "Chuẩn hóa demo UI: global calibrated threshold là default, per-class/guard ở Advanced.",
                "Bổ sung streaming benchmark với false alarms per hour, latency và miss rate trên audio dài.",
            ]
        ),
        h(2, "5.3. Claim nên dùng"),
        tbl(
            "Bảng 11. Claim hợp lệ và claim cần tránh.",
            ["Nên viết", "Không nên viết"],
            [
                ["DSCNN-L + PCEN + GE2E là best accuracy hiện tại.", "Mọi model trong project đều vượt paper."],
                ["EdgeSpotFull T4 + PCEN + GE2E cạnh tranh và nhỉnh hơn mean EdgeSpot-4.", "Đã reproduce đầy đủ EdgeSpot-4."],
                ["SCAF chưa ổn định trong setting cap620 hiện tại.", "SCAF là loss vô dụng."],
                ["UI sampled evaluation là demo/debug.", "UI test thay thế benchmark test100."],
            ],
        ),
    ]


def appendix_blocks() -> list[base.Block]:
    return [
        h(1, "Phụ lục A. Lệnh tái lập"),
        h(2, "A.1. Fixed 16-pipeline cap620"),
        p("MAX_SECONDS=172800 SYNC_SECONDS=300 bash colab/run_mswc_cap620_16_pipeline_e40_fixed.sh"),
        h(2, "A.2. Development run"),
        p(
            "MAX_SECONDS=172800 SYNC_SECONDS=300 RUN_ACCURACY=1 RUN_COMPACT=1 RUN_KD=0 RUN_SCAF_ABLATION=0 "
            "ACC_EPOCHS=60 ACC_EPISODES=300 COMPACT_EPOCHS=60 COMPACT_EPISODES=300 GSC_SELECT_METRIC=composite "
            "bash colab/run_mswc_cap620_development_experiments.sh"
        ),
        h(2, "A.3. Evaluate checkpoint"),
        p(
            "python scripts/evaluate_edgespot_protocol.py --checkpoint checkpoints/<run_tag>/best.pt "
            "--model-family auto --feature-type auto --k-shot 10 --n-runs 100 --gsc-query-split test "
            "--output-dir results/<tag>/test100_far1"
        ),
        h(1, "Phụ lục B. File code chính"),
        tbl(
            "Bảng 12. File code quan trọng.",
            ["File", "Vai trò"],
            [
                ["scripts/train.py", "Training encoder với Triplet/SCAF/GE2E/KD và checkpoint selection."],
                ["scripts/evaluate.py", "Evaluation nhiều protocol và scoring methods."],
                ["scripts/evaluate_edgespot_protocol.py", "Wrapper canonical cho GSC exact test100."],
                ["src/evaluation/protocols.py", "Positive/negative partition, prototype enrollment, metric aggregation."],
                ["src/evaluation/gsc.py", "GSC support/query provider và silence crops."],
                ["src/evaluation/mswc.py", "MSWC randomized support/query provider 1:9."],
                ["src/models/ge2e.py", "GE2E loss."],
                ["src/models/arcface.py", "ArcFace/Sub-center ArcFace/SCAF loss."],
                ["src/features/pcen.py", "Trainable PCEN frontend."],
                ["src/demo/api_server.py", "Demo backend và calibration endpoints."],
            ],
        ),
        h(1, "Phụ lục C. Checklist tái lập"),
        bullets(
            [
                "Kiểm tra checkpoint `best.pt` đúng run id.",
                "Kiểm tra result JSON có `n_runs=100` và `target_far=0.01` hoặc `0.05` đúng bảng.",
                "Không trộn Microset, Top500, cap620 fixed và cap620 development trong cùng ranking nếu không ghi rõ profile.",
                "Không dùng UI sampled result thay cho `gsc_edgespot_exact test100`.",
                "Lưu checkpoint, result JSON, DET curve, CSV summary, run log và manifest split.",
            ]
        ),
    ]


def build_blocks() -> list[base.Block]:
    rows = base.load_cap620_rows()
    blocks: list[base.Block] = []
    blocks.extend(title_blocks())
    blocks.extend(chapter_1())
    blocks.extend(chapter_2())
    blocks.extend(chapter_3())
    blocks.extend(chapter_4(rows))
    blocks.extend(chapter_5())
    blocks.extend(appendix_blocks())
    return blocks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blocks = build_blocks()
    OUT_MD.write_text(base.render_markdown(blocks), encoding="utf-8")
    doc = base.build_docx(blocks)
    doc.core_properties.title = "Few-Shot Open-Set KWS - Reference Style Vietnamese Thesis"
    doc.core_properties.subject = "Reference-style thesis draft"
    doc.core_properties.author = "Codex"
    doc.save(OUT_DOCX)
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_DOCX}")


if __name__ == "__main__":
    main()
