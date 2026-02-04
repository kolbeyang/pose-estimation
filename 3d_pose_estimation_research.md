# 3D Pose Estimation: Video Datasets & Evaluation Metrics

## Video Datasets for 3D Pose Estimation

### Major Benchmarks

| Dataset | Environment | Frames | Cameras | Subjects | Key Features | Link |
|---------|-------------|--------|---------|----------|--------------|------|
| **Human3.6M** | Indoor lab | 3.6M | 4 fixed | 11 | Gold standard, 17 joints, MoCap ground truth | [Website](http://vision.imar.ro/human3.6m/) |
| **MPI-INF-3DHP** | Real scene | 1.3M | 14 | 8 | Markerless MoCap, more diverse actions | [Website](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) |
| **3DPW** | Outdoor | 51K | 1 | 7 | In-the-wild, IMU-based ground truth | [Website](https://virtualhumans.mpi-inf.mpg.de/3DPW/) |
| **AIST++** | Indoor | 10.1M | 9 | 30 | Dance motions, diverse poses | [Website](https://google.github.io/aistplusplus_dataset/) |

### Dataset Details

#### Human3.6M
- **Website**: [http://vision.imar.ro/human3.6m/](http://vision.imar.ro/human3.6m/)
- **Paper**: [TPAMI 2014](https://vision.imar.ro/human3.6m/pami-h36m.pdf)
- 3.6 million accurate 3D poses from 11 subjects (5 female, 6 male)
- 4 synchronized viewpoints at 30 FPS
- 15 action categories (taking photos, talking on phone, eating, etc.)
- **Standard split**: Train on subjects 1, 5, 6, 7, 8; Test on subjects 9, 11
- 17 joint skeleton
- **Access**: Requires account registration

#### MPI-INF-3DHP
- **Website**: [https://vcai.mpi-inf.mpg.de/3dhp-dataset/](https://vcai.mpi-inf.mpg.de/3dhp-dataset/)
- **Paper**: [3DV 2017](https://vcai.mpi-inf.mpg.de/3dhp-dataset/)
- Commercial markerless motion capture (no special suits required)
- 8 actors (4 female, 4 male), 8 action sets (~1 min each)
- Test set: 2929 frames from 6 subjects, 7 actions
- More action diversity than Human3.6M
- Results typically reported in 3DPCK (higher is better)
- **Size**: ~25GB training, ~7GB test

#### 3DPW (3D Poses in the Wild)
- **Website**: [https://virtualhumans.mpi-inf.mpg.de/3DPW/](https://virtualhumans.mpi-inf.mpg.de/3DPW/)
- **Paper**: [ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Timo_von_Marcard_Recovering_Accurate_3D_ECCV_2018_paper.pdf)
- Outdoor, monocular in-the-wild dataset
- Ground truth from IMU sensors
- Often used as primary benchmark for real-world performance
- 60 sequences across 4 scenes
- Includes SMPL model parameters

#### AIST++
- **Website**: [https://google.github.io/aistplusplus_dataset/](https://google.github.io/aistplusplus_dataset/)
- **Download**: [Instructions](https://google.github.io/aistplusplus_dataset/download.html)
- **Paper**: [ICCV 2021](https://google.github.io/aichoreographer/)
- Largest 3D human dance dataset
- 1,408 sequences, 30 subjects, 10 dance genres
- 10M+ frames with corresponding images
- Built from [AIST Dance Video DB](https://aistdancedb.ongaaccel.jp/)

### Recent Datasets (2024-2025)

#### VOccl3D (2025)
- **Paper**: [arXiv:2508.06757](https://arxiv.org/abs/2508.06757)
- 250,000+ frames, 2.5+ hours of video
- Focus: Real occlusion scenarios
- Includes: bounding boxes, body part segmentations, silhouettes
- Diverse body shapes and textures

#### Human3.6M-C (CVPR 2024 Workshop)
- **Paper**: [CVPR 2024 Workshop](https://openaccess.thecvf.com/content/CVPR2024W/TCV2024/papers/Hoang_Improving_the_Robustness_of_3D_Human_Pose_Estimation_A_Benchmark_CVPRW_2024_paper.pdf)
- Synthetically corrupted version of Human3.6M
- Includes temporal occlusion, motion blur
- Tests robustness of lifter-based 3D HPE models

#### FreeMan (CVPR 2024)
- **Paper**: [arXiv:2309.05073](https://arxiv.org/html/2309.05073v4)
- Real-world conditions benchmark
- Addresses limitations of lab-based datasets

### 2D Pose Estimation Datasets

| Dataset | Images | Keypoints | Notes | Link |
|---------|--------|-----------|-------|------|
| **COCO** | 200K+ | 17 | Multi-person, in-the-wild, most popular | [Website](https://cocodataset.org/) |
| **MPII** | 25K | 16 | Single-person, 410 activities | [Website](http://human-pose.mpi-inf.mpg.de/) |
| **LSP** | 2K | 14 | Sports poses | [Website](https://sam.johnson.io/research/lsp.html) |
| **CrowdPose** | 20K | 14 | Crowded scenes, occlusion | [GitHub](https://github.com/Jeff-sjtu/CrowdPose) |
| **OCHuman** | 5K | 17 | Heavy occlusion benchmark | [GitHub](https://github.com/liruilong940607/OCHumanApi) |

---

## Evaluation Metrics

### Spatial Accuracy Metrics

| Metric | Full Name | Description | Units |
|--------|-----------|-------------|-------|
| **MPJPE** | Mean Per Joint Position Error | Euclidean distance between predicted and GT joints | mm |
| **PA-MPJPE** | Procrustes-Aligned MPJPE | MPJPE after rigid alignment (translation, rotation, scale) | mm |
| **MPVE/PVE** | Mean Per Vertex Error | For mesh recovery methods (SMPL) **(Not applicable)** | mm |
| **PCK** | Percentage of Correct Keypoints | Joint within threshold (typically 50mm) | % |
| **3DPCK** | 3D PCK | Used primarily with MPI-INF-3DHP | % |

### Temporal Consistency Metrics

| Metric | Full Name | Description | Units |
|--------|-----------|-------------|-------|
| **Accel** | Acceleration Error | Difference in acceleration between predicted and GT | mm/s² |
| **MPJVE** | Mean Per Joint Velocity Error | Velocity accuracy | mm/s |
| **MPJAE** | Mean Per Joint Acceleration Error | Second-order differential accuracy | mm/s² |
| **MPJJE** | Mean Per Joint Jitter Error | First-order differential of acceleration | mm/s³ |

### 2D Pose Estimation Metrics

| Metric | Full Name | Description | Benchmark |
|--------|-----------|-------------|-----------|
| **PCK** | Percentage of Correct Keypoints | % of keypoints within threshold of GT (normalized by torso size) | LSP |
| **PCKh@0.5** | PCK with Head normalization | % of keypoints within 50% of head segment length | MPII |
| **OKS** | Object Keypoint Similarity | Similarity score (0-1) based on keypoint distance normalized by object scale | COCO |
| **AP** | Average Precision | Mean AP across OKS thresholds 0.50-0.95 | COCO |
| **AP50** | AP at OKS=0.50 | Precision at loose threshold | COCO |
| **AP75** | AP at OKS=0.75 | Precision at strict threshold | COCO |
| **AR** | Average Recall | Mean recall across OKS thresholds | COCO |

---

## Key Papers & Citations

### Foundational Body Model

1. **SMPL** (SIGGRAPH Asia 2015) ⭐ Foundational
   - Loper et al., "SMPL: A Skinned Multi-Person Linear Model"
   - [Paper](https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf) | [Website](https://smpl.is.tue.mpg.de/)
   - **Contribution**: Learned parametric 3D human body model; basis for most mesh recovery methods
   - **Impact**: 4000+ citations; enables HMR, SPIN, VIBE, and all SMPL-based methods

---

### Foundational Dataset Papers

2. **Human3.6M** (TPAMI 2014) ⭐ Gold Standard Benchmark
   - Ionescu et al., "Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments"
   - [Paper](https://vision.imar.ro/human3.6m/pami-h36m.pdf)
   - **Dataset**: 3.6M frames, 4 cameras, 11 subjects, 15 actions
   - **Metrics**: MPJPE (Protocol 1), PA-MPJPE (Protocol 2)
   - **Standard Split**: Train S1,5,6,7,8; Test S9,11

3. **3DPW** (ECCV 2018) ⭐ In-the-Wild Standard
   - von Marcard et al., "Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera"
   - **Dataset**: 51K frames, outdoor/in-the-wild, IMU ground truth
   - **Metrics**: PA-MPJPE, MPJPE, Acceleration Error
   - **Use**: Primary benchmark for real-world generalization

---

### Landmark 2D Pose Estimation

4. **Stacked Hourglass** (ECCV 2016) ⭐ Foundational Architecture
   - Newell et al., "Stacked Hourglass Networks for Human Pose Estimation"
   - [arXiv:1603.06937](https://arxiv.org/abs/1603.06937) | [GitHub](https://github.com/princeton-vl/pytorch_stacked_hourglass)
   - **Contribution**: Multi-scale feature processing via repeated bottom-up/top-down architecture
   - **Datasets**: MPII, FLIC
   - **Metrics**: PCKh@0.5
   - **Performance**: State-of-the-art on MPII/FLIC at time of publication
   - **Impact**: 1500+ citations; foundational for later 2D detectors

5. **HRNet** (CVPR 2019) ⭐ Current 2D Standard
   - Sun et al., "Deep High-Resolution Representation Learning for Human Pose Estimation"
   - [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.pdf) | [GitHub](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
   - **Contribution**: Maintains high-resolution representations throughout network
   - **Datasets**: COCO, MPII, PoseTrack
   - **Metrics**: AP (Average Precision)
   - **Performance (COCO test-dev)**:
     - HRNet-W48: 75.5 AP
   - **Impact**: De facto standard 2D detector for lifting methods

---

### Landmark Lifting Methods (2D → 3D)

6. **SimpleBaseline** (ICCV 2017) ⭐ Foundational Lifting Paper
   - Martinez et al., "A Simple Yet Effective Baseline for 3D Human Pose Estimation"
   - [arXiv:1705.03098](https://arxiv.org/abs/1705.03098) | [GitHub](https://github.com/una-dinosauria/3d-pose-baseline)
   - **Contribution**: Simple feedforward network lifts 2D keypoints to 3D; showed lifting is easier than end-to-end
   - **Dataset**: Human3.6M
   - **Metrics**: MPJPE (Protocol 1), PA-MPJPE (Protocol 2)
   - **Performance (Human3.6M)**:
     - GT 2D input: ~45mm MPJPE
     - Stacked Hourglass input: ~62mm MPJPE (Protocol 1)
     - Fine-tuned detector: ~53mm MPJPE
   - **Impact**: Established lifting paradigm; 30% improvement over prior art

7. **VideoPose3D** (CVPR 2019) ⭐ Temporal Lifting Standard
   - Pavllo et al., "3D Human Pose Estimation in Video with Temporal Convolutions and Semi-Supervised Training"
   - [arXiv:1811.11742](https://arxiv.org/abs/1811.11742) | [GitHub](https://github.com/facebookresearch/VideoPose3D)
   - **Contribution**: Dilated temporal convolutions; semi-supervised back-projection training
   - **Dataset**: Human3.6M, HumanEva-I
   - **Metrics**: MPJPE, PA-MPJPE
   - **Performance (Human3.6M, CPN detections)**:
     - MPJPE: 46.8mm (Protocol 1)
     - PA-MPJPE: 36.5mm (Protocol 2)
   - **Impact**: 11% improvement over prior art; influential temporal modeling

8. **PoseFormer** (ICCV 2021) ⭐ First Transformer-Based
   - Zheng et al., "3D Human Pose Estimation with Spatial and Temporal Transformers"
   - [arXiv:2103.10455](https://arxiv.org/abs/2103.10455) | [GitHub](https://github.com/zczcwh/PoseFormer)
   - **Contribution**: Pure transformer architecture for spatial-temporal pose modeling
   - **Dataset**: Human3.6M, MPI-INF-3DHP
   - **Metrics**: MPJPE, PA-MPJPE, PCK, AUC
   - **Performance (Human3.6M, detected 2D)**:
     - MPJPE: 44.3mm (Protocol 1)
     - GT 2D input: 31.3mm MPJPE
   - **Performance (MPI-INF-3DHP)**:
     - PCK: 88.6%, AUC: 56.4%, MPJPE: 77.1mm

9. **MixSTE** (CVPR 2022)
   - Zhang et al., "MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video"
   - [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_MixSTE_Seq2seq_Mixed_Spatio-Temporal_Encoder_for_3D_Human_Pose_Estimation_CVPR_2022_paper.html) | [GitHub](https://github.com/JinluZhang1126/MixSTE)
   - **Contribution**: Alternating spatial/temporal transformer blocks; seq2seq output
   - **Dataset**: Human3.6M, MPI-INF-3DHP, HumanEva
   - **Metrics**: MPJPE, PA-MPJPE
   - **Performance (Human3.6M, CPN detections)**:
     - MPJPE: 40.9mm (Protocol 1) — 7.6% better than PoseFormer
     - PA-MPJPE: 32.6mm (Protocol 2)
   - **Performance (HumanEva)**: 16.1mm MPJPE

10. **MotionBERT** (ICCV 2023) ⭐ Current State-of-the-Art
    - Zhu et al., "MotionBERT: A Unified Perspective on Learning Human Motion Representations"
    - [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV_2023_paper.html) | [GitHub](https://github.com/Walter0807/MotionBERT)
    - **Contribution**: Dual-stream spatio-temporal transformer; unified motion pretraining
    - **Dataset**: Human3.6M, AMASS, PoseTrack, InstaVariety
    - **Metrics**: MPJPE, PA-MPJPE, velocity error
    - **Performance (Human3.6M, detected 2D)**:
      - MPJPE: 37.5mm (from scratch)
      - MPJPE: 35.8mm (with pretraining) — current SOTA

---

### Landmark Mesh Recovery Methods (Image → SMPL)

11. **HMR** (CVPR 2018) ⭐ First End-to-End Mesh Recovery
    - Kanazawa et al., "End-to-End Recovery of Human Shape and Pose"
    - [arXiv:1712.06584](https://arxiv.org/abs/1712.06584) | [Project](https://akanazawa.github.io/hmr/)
    - **Contribution**: First end-to-end regression of SMPL parameters from single image
    - **Datasets**: Human3.6M, MPI-INF-3DHP, LSP, COCO
    - **Metrics**: MPJPE, Reconstruction Error (PA-MPJPE)
    - **Performance (Human3.6M)**:
      - Reconstruction Error: ~56mm
    - **Impact**: Established end-to-end mesh recovery paradigm

12. **SPIN** (ICCV 2019) ⭐ Optimization-in-the-Loop
    - Kolotouros et al., "Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop"
    - [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kolotouros_Learning_to_Reconstruct_3D_Human_Pose_and_Shape_via_Model-Fitting_ICCV_2019_paper.pdf) | [GitHub](https://github.com/nkolot/SPIN)
    - **Contribution**: Self-improving network with SMPLify optimization in training loop
    - **Datasets**: Human3.6M, 3DPW, MPI-INF-3DHP
    - **Metrics**: PA-MPJPE, MPJPE, PVE
    - **Performance (3DPW)**:
      - PA-MPJPE: 59.2mm
      - MPJPE: 96.9mm
    - **Impact**: Major improvement over HMR; widely used backbone

---

### Video-Based Mesh Recovery Methods

13. **VIBE** (CVPR 2020)
    - Kocabas et al., "VIBE: Video Inference for Human Body Pose and Shape Estimation"
    - [arXiv:1912.05656](https://arxiv.org/abs/1912.05656) | [GitHub](https://github.com/mkocabas/VIBE)
    - **Contribution**: Temporal encoder with adversarial motion discriminator (AMASS)
    - **Datasets**: 3DPW, Human3.6M, MPI-INF-3DHP
    - **Metrics**: PA-MPJPE, MPJPE, PVE, Acceleration Error
    - **Performance (3DPW)**:
      - PA-MPJPE: 51.9mm
      - MPJPE: 82.9mm
      - Accel: 23.4 mm/s²
    - **Performance (Human3.6M)**:
      - PA-MPJPE: 53.3mm
      - Accel: 27.3 mm/s²

14. **TCMR** (CVPR 2021)
    - Choi et al., "Beyond Static Features for Temporally Consistent 3D Human Pose and Shape from a Video"
    - [arXiv:2011.08627](https://ar5iv.labs.arxiv.org/html/2011.08627) | [GitHub](https://github.com/hongsukchoi/TCMR_RELEASE)
    - **Contribution**: Reduces dependency on static features; PoseForecast module
    - **Datasets**: 3DPW, Human3.6M, MPI-INF-3DHP
    - **Metrics**: PA-MPJPE, MPJPE, Acceleration Error
    - **Performance (Human3.6M)**:
      - PA-MPJPE: 52.0mm
      - MPJPE: 73.6mm
      - Accel: 3.9 mm/s² — **7× better than VIBE**
    - **Performance (3DPW)**:
      - PA-MPJPE: 52.7mm
      - Accel: 6.8 mm/s²

---

### Post-Processing / Refinement Methods

15. **SmoothNet** (ECCV 2022) ⭐ Post-Processing Standard
    - Zeng et al., "SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos"
    - [arXiv:2112.13715](https://arxiv.org/abs/2112.13715) | [GitHub](https://github.com/cure-lab/SmoothNet) | [Project](https://ailingzeng.site/smoothnet)
    - **Contribution**: Temporal-only refinement; works with any backbone
    - **Datasets**: Human3.6M, 3DPW, AIST++, MPI-INF-3DHP, MuPoTS-3D, Sub-JHMDB
    - **Metrics**: MPJPE, PA-MPJPE, Acceleration Error
    - **Supported Backbones**: SPIN, TCMR, VIBE, CPN, FCN, Hourglass, HRNet, RLE, VideoPose3D, TposeNet, EFT, PARE, SimplePose
    - **Key Results**:
      - Consistently reduces acceleration error across all backbones
      - Improves or maintains MPJPE/PA-MPJPE
      - Strong cross-dataset and cross-backbone transferability

---

### Recent Work (2024-2025)

16. **BioPose** (WACV 2025)
    - Koleini et al., "Biomechanically-accurate 3D pose estimation from monocular videos"
    - **Focus**: Biomechanical constraints for plausible poses

17. **STRIDE** (2024)
    - Lal et al., "Single-video based temporally continuous occlusion robust 3D pose estimation"
    - **Focus**: Occlusion handling with temporal continuity

18. **SynSP** (CVPR 2024)
    - Wang et al., "SynSP: Synergy of Smoothness and Precision in Pose Sequences Refinement"
    - [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_SynSP_Synergy_of_Smoothness_and_Precision_in_Pose_Sequences_Refinement_CVPR_2024_paper.pdf)
    - **Focus**: Improved post-processing balancing smoothness and precision

---

### Surveys & Reviews

19. **Comprehensive Survey 2025**
    - "A Survey of the State of the Art in Monocular 3D Human Pose Estimation: Methods, Benchmarks, and Challenges"
    - Sensors 2025 | [Paper](https://www.mdpi.com/1424-8220/25/8/2409)
    - First comprehensive analysis including diffusion models and SSMs

20. **Deep Learning Survey 2024**
    - "Deep Learning for 3D Human Pose Estimation and Mesh Recovery: A Survey"
    - [arXiv:2402.18844](https://arxiv.org/html/2402.18844v1)
    - Covers 200+ papers; single/multi-person pose and mesh recovery

---

### Quick Reference: Benchmark Comparisons

#### Human3.6M MPJPE (Protocol 1, detected 2D) — Lower is Better

| Method | Year | MPJPE (mm) | Notes |
|--------|------|------------|-------|
| SimpleBaseline | 2017 | ~62 | First lifting baseline |
| VideoPose3D | 2019 | 46.8 | Temporal convolutions |
| PoseFormer | 2021 | 44.3 | First transformer |
| MixSTE | 2022 | 40.9 | Alternating ST blocks |
| **MotionBERT** | 2023 | **35.8** | Current SOTA |

#### 3DPW PA-MPJPE — Lower is Better

| Method | Year | PA-MPJPE (mm) | Accel (mm/s²) |
|--------|------|---------------|---------------|
| SPIN | 2019 | 59.2 | — |
| VIBE | 2020 | 51.9 | 23.4 |
| TCMR | 2021 | 52.7 | 6.8 |

#### Human3.6M Acceleration Error — Lower is Better

| Method | Accel (mm/s²) |
|--------|---------------|
| VIBE | 27.3 |
| TCMR | 3.9 |
| + SmoothNet | Further reduced |

---

### Paper × Metric Usage Matrix

| Paper | Method | Year | MPJPE | PA-MPJPE | PVE | PCK | AUC | Accel | AP | PCKh |
|-------|--------|------|:-----:|:--------:|:---:|:---:|:---:|:-----:|:--:|:----:|
| [Stacked Hourglass](https://arxiv.org/abs/1603.06937) | Multi-scale CNN with bottom-up/top-down processing | 2016 | | | | | | | | ✅ |
| [SimpleBaseline](https://arxiv.org/abs/1705.03098) | Feedforward network lifting 2D→3D | 2017 | ✅ | ✅ | | | | | | |
| [HMR](https://arxiv.org/abs/1712.06584) | End-to-end CNN regressing SMPL params | 2018 | ✅ | ✅ | | ✅ | ✅ | | | |
| [HRNet](https://arxiv.org/abs/1902.09212) | High-resolution representations throughout | 2019 | | | | | | | ✅ | ✅ |
| [VideoPose3D](https://arxiv.org/abs/1811.11742) | Dilated temporal convolutions on 2D keypoints | 2019 | ✅ | ✅ | | | | | | |
| [SPIN](https://arxiv.org/abs/1909.12828) | SMPLify optimization in training loop | 2019 | ✅ | ✅ | ✅ | | | | | |
| [VIBE](https://arxiv.org/abs/1912.05656) | GRU temporal encoder + motion discriminator | 2020 | ✅ | ✅ | ✅ | | | ✅ | | |
| [PoseFormer](https://arxiv.org/abs/2103.10455) | Spatial-temporal transformer on 2D keypoints | 2021 | ✅ | ✅ | | ✅ | ✅ | | | |
| [TCMR](https://arxiv.org/abs/2011.08627) | Temporal encoding without static feature dominance | 2021 | ✅ | ✅ | ✅ | | | ✅ | | |
| [MixSTE](https://arxiv.org/abs/2203.00859) | Alternating spatial/temporal transformer blocks | 2022 | ✅ | ✅ | | ✅ | ✅ | | | |
| [SmoothNet](https://arxiv.org/abs/2112.13715) | Temporal-only post-processing refinement | 2022 | ✅ | ✅ | | | | ✅ | | |
| [MotionBERT](https://arxiv.org/abs/2210.06551) | Dual-stream transformer with motion pretraining | 2023 | ✅ | ✅ | | | | ✅ | | |

**Legend**: ✅ = Metric reported in paper

---

## Recommended Evaluation Protocol

For comparing a base model with post-processing improvements:

### Datasets
1. **Human3.6M** - Controlled environment, standard benchmark
2. **3DPW** - In-the-wild generalization

### Metrics to Report

| Category | Metrics |
|----------|---------|
| Spatial Accuracy | MPJPE, PA-MPJPE |
| Temporal Consistency | Acceleration Error |

---

## Resources

- [SmoothNet GitHub](https://github.com/cure-lab/SmoothNet) - Post-processing baseline
- [VIBE GitHub](https://github.com/mkocabas/VIBE) - Video pose estimation
- [3DPW Evaluation Code](https://github.com/miraymen/3dpw-eval) - ECCV 2020 workshop
- [Human Pose Estimation 101](https://github.com/cbsudux/Human-Pose-Estimation-101) - Basics overview
