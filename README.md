<div align="center">
   <h1 align="center">RelaxFlow: Text-Driven Amodal 3D Generation
   </h1>

   <p>
      <a href=https://arxiv.org/abs/xxx target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv height=25px></a>
      <a href="https://jyzhu.top/RelaxFlow_Webpage/" target="_blank"><img src= https://img.shields.io/badge/Project%20Page-bb8a2e.svg?logo=github height=25px></a>
      <br>
      <!-- <img src="https://img.shields.io/github/languages/top/viridityzhu/RelaxFlow?style&color=5D6D7E" alt="GitHub top language" />
      <img src="https://img.shields.io/github/languages/code-size/viridityzhu/RelaxFlow?style&color=5D6D7E" alt="GitHub code size in bytes" /> -->
   </p>
</div>

<p align="center">
  <a href="https://jyzhu.top/" target="_blank">Jiayin Zhu</a><sup>1</sup>,&nbsp;</a>
  <a href="https://scholar.google.com/citations?user=a8CLpC0AAAAJ&hl=en" target="_blank">Guoji Fu</a><sup>1</sup>,&nbsp;</a>
  <a href="https://github.com/xiaolul2" target="_blank">Xiaolu Liu</a><sup>2 1</sup>,&nbsp;</a>
  <a href="https://qy-h00.github.io/" target="_blank">Qiyuan He</a><sup>1</sup>,&nbsp;</a>
  <a href="https://yl3800.github.io/" target="_blank">Yicong Li</a><sup>1</sup>,&nbsp;</a>
  <a href="https://www.comp.nus.edu.sg/~ayao/" target="_blank">Angela Yao</a><sup>1</sup>;</a>
  <br>
  National University of Singapore <sup>1</sup>
  <br/>
  Zhejiang University <sup>2</sup>
</p>

<!-- <h3 align="center">Arxiv 2026</h3> -->

## 🎯 What We Do: Resolving Semantic Ambiguity

<p align="center"><img width="70%" src="doc/teaser.png"/></p>

Image-to-3D generation faces inherent semantic ambiguity under occlusion, where partial observation alone is often insufficient to determine the object category. For instance, a visible wooden backboard could plausibly correspond to a sofa, a bed, or a dressing table. Existing feedforward models, like SAM3D, often collapse to an "observation-overfitted" shape by uncontrolled hallucination.

We formalize **text-driven amodal 3D generation**. Our task allows users to *explicitly steer the completion of unseen regions using text prompts*, while strictly preserving the visual evidence of the input observation.

## ⚙️ How We Do It: Decoupled Control & Relaxation

<p align="center"><img src="doc/fig-pipeline.png"/></p>

These dual objectives demand distinct control granularities: rigid control for the visible observation versus relaxed structural control for the text prompt. To solve this, we propose **RelaxFlow**, a training-free dual-branch framework:

- **Observation Branch**: Provides strict adherence to ensure visual fidelity for the observed pixels.
- **Multi-Prior Consensus**: Converts the text prompt into visual proxy reference images. Cross-attention across these priors naturally amplifies structural consensus while suppressing inconsistent, instance-specific textures.
- **Visibility-Aware Fusion**: A spatial blending mechanism ensuring the semantic guide only steers genuinely occluded regions, while the observation strictly governs the visible pixels.

### The Theory: Low-Pass Relaxation

<p align="center"><img src="doc/fig-corridor-demon.png"/></p>

A core challenge is preventing the text prompt's high-frequency details from clashing with the input image. We introduce a **Relaxation Mechanism** that smooths cross-attention logits within the generation backbone.

Theoretically, we prove this smoothing is equivalent to applying a low-pass filter on the generative vector field. This mathematically suppresses high-frequency instance details and exposes a "coarse semantic corridor," enforcing only the low-frequency global geometry needed to accommodate the observation (e.g., the general shape of a "sofa").

## 📊 Benchmarks & Results

To facilitate systematic evaluation, we introduce two new diagnostic benchmarks:

- **ExtremeOcc-3D**: Targets extreme occlusion in natural indoor scenes where visible evidence cannot identify the object category.
- **AmbiSem-3D**: Targets semantic branching, where the same visual evidence admits multiple plausible interpretations, paired with distinct text prompts.

### Results 

<p align="center"><img src="doc/fig-qual.png"/></p>

Extensive experiments demonstrate that RelaxFlow successfully steers the generation of unseen regions to match the prompt intent. It avoids the observation-overfitted collapse of existing models and produces high-quality 3D assets without compromising visual fidelity.

## 🚀 Get Started

### Installation

Follow the [setup](https://github.com/facebookresearch/sam-3d-objects/blob/afdf6a31522d038c44c68a0bb57aa68827380797/doc/setup.md) steps of SAM 3D Objects before running the following.
Based on our testing, the minimum requirement is a single GPU with 24GB of memory (e.g., NVIDIA RTX A5000).

## Quickstart

For a quick start, run `python demo_relaxflow.py` using test data:

```sh
FOLDER="test_data/A_bike_with_a_blue_front_wheel_and_a_red_rear_wheel"
OUTNAME=$(basename $FOLDER)
IMG=${FOLDER}/image.png
MSK=${FOLDER}/mask.png
# PRI="${FOLDER}/prior1.png ${FOLDER}/prior2.png ${FOLDER}/prior3.png ${FOLDER}/prior4.png"
PRI=${FOLDER}/prior.png
python demo_relaxflow.py --image $IMG --mask $MSK --prior-images $PRI --output-name $OUTNAME 
```

Another case:

```sh
FOLDER="test_data/dressing_table"
OUTNAME=$(basename $FOLDER)
IMG=${FOLDER}/input.png
PRI="${FOLDER}/prior1.png ${FOLDER}/prior2.png ${FOLDER}/prior3.png"
python demo_relaxflow.py --image $IMG --prior-images $PRI --output-name $OUTNAME 
```

Results will be saved into `outputs/`.

### Benchmarks

<will update soon>
For testing benchmarks **ExtremeOcc-3D** and **AmbiSem-3D**, please first download the datasets via [link](tbd). Then run `python demo_relaxflow_batch.py` using the prepared manifests:

```sh
python demo_relaxflow_batch.py \... #todo: publish the datasets and manifest files
```

## License

This repository is built upon the SAM 3D Objects model as a backbone; both the original SAM 3D Objects code and the modifications in this repository are licensed under the [SAM License](./LICENSE).

## Citing RelaxFlow

If you find our work useful, please use the following BibTeX entry.

< TODO: update bibtex here >

```bibtex
@article{zhu2026relaxflow,
  title={RelaxFlow: Text-Driven Amodal 3D Generation},
  author={Zhu, Jiayin and Fu, Guoji and Liu, Xiaolu and He, Qiyuan and Li, Yicong and Yao, Angela},
  journal={arXiv preprint},
  year={2026}
}
```