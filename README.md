# 🌕 Astronomical objects craters detection by deep learning

### 📑 About

This project is an Engineering thesis aimed at developing **a system to detect craters on celestial bodies**. It features a modified **Attention U-Net** convolutional network and utilizes **Combo loss** function to enhance identification accuracy. Designed to support both planetary research and navigation system development, it has been effectively tested across various planetary environments. 

The solution is implemented in *Python 3.9.17* using PyTorch 2.0.1 and CUDA 11.7. To run the system, clone the repo, install dependencies from `requirements.txt`, optionally change parameters in `settings.py` and execute `main.py`, which will display a menu with available options.

> For additional information or to access specific references related to this project, please contact the authors: **Tomash Mikulevich** (tommikulevich@gmail.com) and/or **Radosław Dębiński** (radekdebinski00@gmail.com).

### 🗂️ Datasets

The **Moon** dataset consisted of 40000 training images, 10000 validation images, and 10000 test images, with each sample sized at $256×256$ pixels representing a $50×50$ km area. This dataset was compiled using [Moon Crater Database v1 Robbins](https://astrogeology.usgs.gov/search/map/Moon/Research/Craters/lunar_crater_database_robbins_2018) and [WAC Global Morphologic Map](https://wms.lroc.asu.edu/lroc/view_rdr/WAC_GLOBAL). 

For **Mars**, we have the test set included 10000 images, each also $256×256$ pixels but representing a $60×60$ km area, compiled using [Mars Crater Database](https://craters.sjrdesign.net/) and [Global CTX Mosaic of Mars](https://murray-lab.caltech.edu/CTX/index.html). 

### 🧠 NN Architecture

To effectively handle the broad range and complexity of craters, we selected the **Attention U-Net** architecture, originally designed for detailed medical image segmentation. This model is particularly useful for identifying small and complex structures. Detailed architecture schemes are provided below.

<p align="center">
  <img src="data/_readme-img/1-AttentionUNet.png?raw=true" alt="Attention U-Net">
</p>

<p align="center">
  <img src="data/_readme-img/2-AttentionGate.png?raw=true" alt="Attention Gate">
</p>

### ⏱️ Training

We decided to use **Combo loss** function, that combines the advantages of cross-entropy and Dice's coefficient:

$$L_{Combo} = \alpha \cdot L_{mCE} - (1 - \alpha) \cdot DSC,$$

where:

$$L_{mCE} = - \frac{1}{N} \sum_{i=1}^{N} \left[ \beta \cdot y_i \ln(p_i) + (1 - \beta) \cdot \left(1 - y_i\right) \ln\left(1 - p_i\right) \right],$$

$$DSC = \sum_{i=1}^{N} \frac{2 y_i p_i}{y_i + p_i}.$$

It allows control not only over the contribution of each component by setting $\alpha$ coefficient but also over the relative weights assigned to falsely identified and missed objects, which is managed by $\beta$ parameter. In our study, we chose $\alpha=0.7$ and $\beta=0.3$.

During training, we used the Adam optimizer with a learning rate of $0.0005$ and weight decay of $10^{-5}$. The model was trained over $20$ epochs with batch sizes of $16$. It included $8$ to $96$ filters and $15\%$ dropout to avoid overfitting. 

### 📈 Results

The following table compares the number of parameters optimized by the network (including weights) as well as performance metrics: **precision**, **recall**, and **F1-score**. As expected, increasing the number of filters led to more parameters. There was no clear trend of improvement or decline in performance metrics; however, a noticeable drop in precision starting from 16 filters and a rising trend in recall up to 32 filters were observed. Based on the F1-score, the model with 32 filters proved to be the most optimal.

|Number of filters $F$|8|16|32|64|96|
|:-------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|Number of parameters|125 820|500 076|1 993 932|7 963 020|17 907 276|
|Precision [%]|69.32|**69.84**|68.93|67.74|66.98|
|Recall [%]|40.08|41.94|**45.79**|44.04|45.29|
|F1-score [%]|50.7|52.31|**54.94**|53.29|53.94|

Below is a graph showing how the loss function changed during training and validation for number of filters $F=32$.

<p align="center">
  <img src="data/_readme-img/3-Loss_F32.png?raw=true" width="600" alt="Loss plot">
</p>

There are example network results for **Moon** photos (*red* — areas predicted correctly by the model only; *blue* — areas correct only in the target mask; *green* — areas where the model's predictions and the target overlap):
- The first row showcases an image with many small craters, evenly lit, where the model performs very well.
- The second row represents a typical (average) outcome, with the model accurately predicting densely packed large craters under challenging light conditions.
- The third row reveals the model spotting several possible craters not recorded in the catalog, demonstrating its ability to generalize but at the cost of reducing performance metrics.
- The fourth row shows a difficult case with minimal correct predictions caused by uneven lighting, which makes it hard for even humans to see clearly.

<p align="center">
  <img src="data/_readme-img/4-Results-Moon.png?raw=true" alt="Samples of Moon">
</p>

Below are example results from the network for **Mars** photos, which were only used for testing and not during training:
- In the first row, the most favorable scenario is presented, similar to the lunar data, where small crater diameters and lighting play a significant role.
- The second row shows interesting model behavior. It seems that unenclosed craters pose a problem, particularly highlighted by an undetected crater in the bottom right corner of the original photo, which entirely escaped the model's detection despite moderate lighting conditions.
- The third row is associated with a different phenomenon. It was found that the described area is covered with mountains and plateaus, a landscape quite uncommon on the Moon's surface, likely confusing the model. Many approximately circular elevations were identified as craters due to the lighting conditions.
- The last row aims to showcase a scenario related to the incompleteness of the test set. Even though the optical data used represents most of Mars' surface, there are gaps. Unmapped areas needed for the mosaic show up in samples as black stripes.

<p align="center">
  <img src="data/_readme-img/5-Results-Mars.png?raw=true" alt="Samples of Mars">
</p>