# CSE4261 – Neural Networks and Deep Learning

This repository consolidates the end‑to‑end academic deliverables for the course **CSE4261 – Neural Networks and Deep Learning**, including source code, experimental artifacts, and analytical reports across 15 structured assignments.

Each assignment articulates practical deep learning workflows spanning convolutional networks, transfer learning, adversarial robustness, explainability, generative modeling, transformer architectures, and knowledge distillation.

---

## Repository Overview

```
CSE4261-Neural-Networks-and-Deep-Learning/
│
├── Assignment_01_CNN_Performance/
├── Assignment_02_Activation_Kernel_Analysis/
├── Assignment_03_Feature_Extraction_TransferLearning/
├── Assignment_04_Manual_Backprop_GradientTape/
├── Assignment_05_Adversarial_FGSM/
├── Assignment_06_Explainability_GradCAM_IG/
├── Assignment_07_YOLO_Object_FaceDetection/
├── Assignment_08_UNet_MCNN_CrowdCounting/
├── Assignment_09_Autoencoder_FeatureStudy/
├── Assignment_10_AE_DAE_VAE_ImageGeneration/
├── Assignment_11_FaceVerification_LossComparison/
├── Assignment_12_KnowledgeDistillation/
├── Assignment_13_GAN_FacialSynthesis/
├── Assignment_14_ViT_Classifier_Analysis/
├── Assignment_15_BERT_FromScratch/
│
└── docs/Assignmet_Question
```

---

## Assignment Portfolio

### Assignment 01 – CNN Performance Benchmark

* **Dataset:** CIFAR‑100 (20 selected classes)
* **Models:** 10 pretrained ImageNet CNNs
* **Focus:** Accuracy, inference latency, parameter footprint, FLOPs, memory utilization

### Assignment 02 – Activation & Kernel Analysis

* **Dataset:** Same as Assignment 01
* **Scope:** Activation function comparison, dilated/deformable/depthwise kernels, feature map visualization

### Assignment 03 – Transfer Learning & Feature Extraction

* **Dataset:** MNIST
* **Task:** Contrast pretrained feature extraction vs fine‑tuned feature extraction; visualize embeddings (PCA, t‑SNE, UMAP)

### Assignment 04 – Manual Backpropagation Formulation

* **Dataset:** MNIST
* **Model:** Custom 3‑layer neural network
* **Deliverables:** Computation graph derivation, analytical gradient formulation, tf.GradientTape vs model.fit performance

### Assignment 05 – FGSM Adversarial Attack

* **Dataset:** Single ImageNet‑class image
* **Objective:** Implement FGSM; compare Gaussian noise failure behavior

### Assignment 06 – Explainability on Adversarial Inputs

* **Tools:** Grad‑CAM, Integrated Gradients
* **Purpose:** Region attribution comparison between clean and adversarial samples

### Assignment 07 – YOLO Object & Face Detection

* **Datasets:** Custom video (object), WIDER FACE (face)
* **Models:** YOLOv1, YOLOv8/11/12
* **Deliverables:** Inference‑speed and mAP comparison, fine‑tuned face detector

### Assignment 08 – U‑Net vs MCNN for Segmentation & Crowd Counting

* **Datasets:** Segmentation (Oxford Pet, ADE20k, or equivalent), Crowd counting (ShanghaiTech, UCF‑CC)
* **Outcome:** Benchmark performance divergence between architectures

### Assignment 09 – Autoencoder vs CNN Representation Study

* **Dataset:** CIFAR‑10
* **Comparison:** AE representation vs CNN embeddings, dimensionality reduction, augmentation impact on accuracy

### Assignment 10 – Image Generation via AE / DAE / VAE

* **Dataset:** CIFAR‑10
* **Deliverable:** Train three generative variants and synthesize images from identical Gaussian priors

### Assignment 11 – Face Verification & Loss Comparison

* **Dataset:** LFW or similar
* **Models:** Siamese networks (BCE, contrastive, triplet), VAE with BCE vs MSE reconstruction

### Assignment 12 – Knowledge Distillation Pipeline

* **Dataset:** Any 10‑class dataset excluding MNIST/Fashion‑MNIST
* **Components:** Two pretrained teacher models → distilled lightweight CNN

### Assignment 13 – GAN–Based Face Synthesis

* **Dataset:** CelebA
* **Models:** DCGAN, Conditional GAN, CycleGAN
* **Theory:** BCE vs Minimax justification in TF GAN layers

### Assignment 14 – Vision Transformer Analysis

* **Dataset:** 20 ImageNet classes
* **Focus:** Patch embedding variants, attention head scaling, positional encoding selection

### Assignment 15 – BERT Pretraining & Fine‑Tuning

* **Datasets:** WikiText (MLM), IMDB/SST (sentiment), SQuAD (QA), SNLI (NLI)
* **Models:** Scratch‑built BERT vs public pretrained baseline

---

## Technology Stack

* **Frameworks:** TensorFlow, PyTorch, Keras
* **Tooling:** NumPy, Pandas, Matplotlib, Scikit‑learn
* **Explainability:** Grad‑CAM, Integrated Gradients
* **Architectures:** YOLOv8/11/12, U‑Net, MCNN, VAE, CycleGAN
* **NLP:** HuggingFace Transformers

---

## Execution Workflow

```
cd Assignment_XX
pip install -r requirements.txt
python main.py
```

Or launch the notebook equivalent if available.

---

## License

This repository is intended for academic and research enablement under fair‑use guidelines.
