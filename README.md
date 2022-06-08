# HiResCAM

This repository contains a small demo of the HiResCAM and Grad-CAM gradient-based
neural network explanation methods.

A more sophisticated demo is coming soon.

Due to the gradient averaging step, sometimes Grad-CAM creates misleading
explanations that highlight regions
a model did not actually use for prediction.
HiResCAM fixes this problem. HiResCAM produces faithful explanations that reliably
show which regions of an image were used for each prediction.

## Papers

This paper introduces the HiResCAM method:

[Rachel Lea Draelos, MD PhD, and Lawrence Carin, PhD. Use HiResCAM instead of Grad-CAM for
faithful explanations of convolutional neural networks. arXiv preprint. 21 Nov 2021. Under review.](https://arxiv.org/abs/2011.08891)

This paper applies the HiResCAM method as part of an explainable method
for multiple abnormality prediction in CT scans:

[Rachel Lea Draelos, MD PhD, and Lawrence Carin, PhD. Explainable multiple
abnormality classification of chest CT volumes with deep learning. arXiv preprint.
24 Nov 2021. Under review.](https://arxiv.org/abs/2111.12215)

## Requirements

Installing requirements with conda:

```
conda env create -f requirements.yml
```

## Demo Code

To run the demo of HiResCAM and Grad-CAM:

```
python demo.py
```

As explained further in the papers,
* For CNNs ending in global average pooling followed by one fully connected
layer (the "CAM architecture," e.g. a ResNet), the visualizations produced by CAM,
Grad-CAM, and HiResCAM are identical, as Grad-CAM and HiResCAM are alternative
generalizations of CAM.
* For CNNs ending in one fully connected layer but no preceding global average
pooling (e.g. the AxialNet model used in this demo),
HiResCAM and Grad-CAM produce different explanations. The HiResCAM
explanation provably reflects the model's computations while the Grad-CAM explanation
does not.

The key difference between HiResCAM and Grad-CAM can be seen by comparing
the HiResCAM and GradCAM functions in cams.py.
The difference between HiResCAM and GradCAM is small in terms of code,
but has significant implications for the faithfulness of the explanation as
discussed further in the papers.
