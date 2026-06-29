# Email To Advisor - 2026-05-22

Dear Teacher,

This week I focused on re-running and comparing configurations for the few-shot open-set keyword spotting system. I first used MSWC Microset to compare the main directions: DSCNN-L + Triplet baseline, EdgeSpotFull T4 + SCAF, and EdgeSpotFull T4 + SCAF+GE2E.

From these Microset experiments, **EdgeSpotFull T4 + SCAF+GE2E** is currently the best configuration among the runs I completed. The locked GSC test100 result is:

- ACC@5%FAR: 86.12%;
- Keyword ACC: 77.66%;
- F1: 82.41%;
- AUC: 95.61%;
- EER: 11.54%.

I then used this configuration to train on the larger Top500 setup. Top500 shows promising signals, but the experiments were affected by Colab session/unit limits. One earlier run had promising logs but was not fully packaged locally before the session was lost, so I am not treating it as the final result. In the later run I improved the pipeline to save checkpoints every epoch to Drive, but it stopped at epoch13 due to resource limits. The reliable local checkpoint I currently have is `epoch_13.pt`; its dev30 ACC@5%FAR is about 88.87%.

I also improved the demo system:

- model switching between Microset and Top500;
- GSC keyword enrollment;
- single-audio detection;
- long-audio testing with labels/timing;
- miss reason explanations;
- open-set testing with 17 known and 17 unknown GSC words;
- threshold/guard calibration.

For the thesis, I will use the Microset result as the main locked result. The Top500 epoch13 checkpoint will be used for demo and preliminary scale-up reporting. When Colab/GPU resources are available again, I will continue Top500 training and produce a complete test100 artifact.
