# Bachelor Thesis Defense Script

## Slide 1 - Cover

Good morning everyone. My name is Pham Hoang An, and today I will present my bachelor thesis on few-shot open-set keyword spotting.

The topic of my project is to study keyword spotting systems that can recognize new keywords from only a few examples, while also rejecting unknown words. In this work, I compare DSCNN, EdgeSpot-style models, MFCC, PCEN, and several metric-learning objectives.

## Slide 2 - Outline

My presentation has five main parts.

First, I will introduce the motivation and objective of the project. Then I will briefly explain the background of few-shot open-set keyword spotting. After that, I will describe the methodology and system pipeline. The fourth part presents the experiments and results. Finally, I will conclude the work and discuss future directions.

## Slide 3 - Motivation

Keyword spotting is the task of detecting short spoken commands or keywords from audio. Traditional keyword spotting systems usually work with a fixed list of keywords and require enough training data for each word.

However, in real applications, users may want to add new keywords from only a few recordings. At the same time, the system must avoid accepting unrelated words as valid keywords. This makes the problem both few-shot and open-set.

## Slide 4 - Project Objective

The main objective of this project is to build and evaluate a flexible few-shot keyword spotting pipeline.

Instead of testing only one model, I compare multiple combinations of encoder, audio frontend, and training objective. The encoders are DSCNN-L and EdgeSpotFull T4. The frontends are MFCC and PCEN. The objectives include Triplet loss, SCAF, GE2E, and SCAF plus GE2E.

The goal is to identify the best overall configuration and also the best compact configuration for edge-oriented deployment.

## Slide 5 - Few-Shot Open-Set KWS

In the few-shot setting, the model receives a small support set. In this project, I mainly use 10-shot evaluation, which means 10 examples for each target keyword.

The system computes a prototype for each keyword from the support examples. Then, for each query audio, it compares the query embedding with the prototypes. If the distance is below a threshold, the query is accepted as a keyword. Otherwise, it is rejected as unknown.

This is important because open-set performance depends not only on recognizing known keywords, but also on controlling false accepts.

## Slide 6 - Evaluation Metrics

For evaluation, I focus on open-set metrics.

FAR measures how often unknown words are falsely accepted. FRR measures how often real keywords are rejected. ACC at 1 percent FAR is the main metric because it measures accuracy under a strict false-accept constraint.

I also report ACC at 5 percent FAR, AUC, EER, keyword accuracy, precision, recall, and F1 score to get a more complete picture of performance.

## Slide 7 - Dataset and Protocol

The training data is based on English clips from the Multilingual Spoken Words Corpus. The evaluation protocol uses Google Speech Commands in a few-shot open-set setting.

The training vocabulary is used to learn a general embedding space, while the evaluation keywords are sampled separately. This allows the system to test whether the learned embedding can generalize to new keyword sets.

The reported results are averaged over repeated runs to reduce the effect of random support-set sampling.

## Slide 8 - System Pipeline

This slide shows the overall pipeline.

First, the audio waveform is converted into acoustic features using MFCC or PCEN. Then the encoder, either DSCNN-L or EdgeSpotFull T4, maps the features into an embedding vector.

During inference, the system compares the query embedding with keyword prototypes. The final decision is based on nearest-prototype distance and a threshold for open-set rejection.

## Slide 9 - Feature Frontends

I compare two feature frontends.

MFCC is a common baseline for speech processing and keyword spotting. It is compact and well understood, but it can be sensitive to noise and loudness variation.

PCEN is based on mel features with adaptive compression. In this project, PCEN works better in the strongest configurations, especially when combined with GE2E.

## Slide 10 - Encoder Architectures

The first encoder is DSCNN-L, a depthwise separable convolutional model. It has stronger capacity and gives the best overall accuracy in the final results.

The second encoder is EdgeSpotFull T4, which is a compact EdgeSpot-style architecture. It has fewer parameters, so it is more suitable as a compact model.

This comparison helps separate the best accuracy-oriented model from the best compact model.

## Slide 11 - Training Objectives

I evaluate several metric-learning objectives.

Triplet loss tries to pull positive examples closer and push negative examples farther apart. SCAF uses sub-centers to model intra-class variation. GE2E compares utterances to class centroids and encourages class-level consistency.

I also test the hybrid SCAF plus GE2E objective. However, the experiments show that combining losses does not always improve performance.

## Slide 12 - Pipeline Matrix

This slide summarizes the 16-pipeline screening experiment.

Each row is a combination of encoder and frontend. Each column is a training objective. The number in each cell is open-set accuracy at 1 percent FAR on the test100 protocol.

From this screening, DSCNN-L plus PCEN plus GE2E is the strongest overall configuration. For the compact EdgeSpot branch, EdgeSpotFull T4 plus PCEN plus GE2E is the strongest configuration.

## Slide 13 - Main Result

After the screening experiment, I ran stronger development experiments with longer training and composite checkpoint selection.

The best overall result is DSCNN-L plus PCEN plus GE2E. It reaches 86.36 percent ACC at 1 percent FAR, with 95.21 AUC and 82.73 F1.

The best compact result is EdgeSpotFull T4 plus PCEN plus GE2E. It reaches 82.87 percent ACC at 1 percent FAR, with 92.41 AUC and 77.85 F1.

## Slide 14 - Discussion

The results show that PCEN and GE2E are the most useful components in this project.

PCEN improves robustness at the feature level, while GE2E helps the model learn a more stable embedding space for few-shot prototypes. On the other hand, SCAF is not stable in the large-vocabulary setting and often collapses to near-random open-set behavior.

Therefore, the final selected configurations are PCEN plus GE2E for both DSCNN-L and EdgeSpotFull T4.

## Slide 15 - Demo System

Besides the benchmark experiments, I also built a demo system for short audio and long audio.

For long audio, the system segments the waveform, detects candidate words, and compares them with expected timing labels if available. The UI shows detected words, false accepts, misses, and detailed candidates.

However, the demo is mainly for qualitative inspection. The main scientific claims still come from the fixed evaluation protocol.

## Slide 16 - Conclusion and Future Work

To conclude, this project builds a few-shot open-set keyword spotting pipeline and evaluates multiple combinations of model architecture, feature frontend, and training objective.

The best overall configuration is DSCNN-L plus PCEN plus GE2E. The best compact configuration is EdgeSpotFull T4 plus PCEN plus GE2E. The experiments also show that SCAF is not reliable in the current large-vocabulary setting.

For future work, I would like to reproduce EdgeSpot-style knowledge distillation more completely, improve long-audio calibration, and evaluate the system under more realistic noisy conditions.

## Slide 17 - Thank You

That concludes my presentation.

Thank you for listening. I am happy to answer your questions.
