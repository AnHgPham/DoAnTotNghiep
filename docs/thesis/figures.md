# Thesis Figures

## Figure 1 - End-To-End Training And Evaluation

```mermaid
flowchart LR
  A["MSWC manifests"] --> B["Audio loader"]
  B --> C["Mel-PCEN extractor"]
  C --> D["EdgeSpotFull T4"]
  D --> E["SCAF + GE2E"]
  E --> F["Checkpoint"]
  F --> G["GSC evaluation"]
  G --> H["Result JSON + DET curve"]
```

Caption VI: Luồng huấn luyện và đánh giá từ manifest MSWC đến checkpoint và kết quả GSC.

Caption EN: Training and evaluation flow from MSWC manifests to checkpoints and GSC results.

## Figure 2 - Prototype Inference

```mermaid
flowchart TB
  A["Enrollment samples"] --> B["Embeddings"]
  B --> C["Prototype mean"]
  D["Query sample"] --> E["Query embedding"]
  E --> F["L2 to prototypes"]
  F --> G{"Accept policy"}
  G -->|distance ok + margin ok| H["keyword"]
  G -->|otherwise| I["unknown"]
```

Caption VI: Cơ chế inference dựa trên prototype và open-set rejection.

Caption EN: Prototype-based inference and open-set rejection.

## Figure 3 - Demo Web Architecture

```mermaid
flowchart TB
  UI["React/Vite UI"] --> API["FastAPI"]
  API --> M["Model profiles"]
  API --> E["Enrollment"]
  API --> D["Detection"]
  API --> O["Open-set calibration"]
  API --> W["Streaming WebSocket"]
  M --> A["Local artifacts"]
```

## Figure 4 - Streaming State Machine

```mermaid
stateDiagram-v2
  [*] --> idle
  idle --> speech_detected
  speech_detected --> scoring
  scoring --> detected
  scoring --> rejected
  detected --> cooldown
  rejected --> cooldown
  cooldown --> idle
```

## Figure 5 - Colab Artifact Pipeline

```mermaid
flowchart LR
  A["Colab training"] --> B["Save every epoch"]
  B --> C["Drive checkpoints"]
  C --> D["Evaluate dev/test"]
  D --> E["Drive result package"]
  E --> F["Local server artifacts"]
  F --> G["Demo + thesis evidence"]
```
