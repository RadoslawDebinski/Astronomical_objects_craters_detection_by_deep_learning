# 🌕 Astronomical objects craters detection by deep learning

### 📑 About

<!-- Write that this is an engineering thesis and that in order to get acquainted with the details (in Polish) user should write to the authors -->
<!-- Description of the project -->

> The application is written in **Python 3.9.17**, with PyTorch 2.0.1 and CUDA 11.7, in PyCharm 2023.1.1 Professional Edition.

### 🗂️ Dataset

<!-- Moon: trening set 40000, validation set 10000, test set 10000 (catalog [29], WAC [34], 256x256, 50x50km) -->
<!-- Mars: test set 10000 (catalog [31], mosaic [5], 256x256, 60x60km) -->

### 🧠 NN Architecture

<!-- Attention U-Net - description -->

<p align="center">
  <img src="data/_readme-img/1-AttentionUNet.png?raw=true" alt="Attention U-Net">
</p>

<p align="center">
  <img src="data/_readme-img/2-AttentionGate.png?raw=true" alt="Attention Gate">
</p>

### ⏱️ Training

<!-- More info about Combo loss (with alpha and beta) -->

$$L_{Combo} = \alpha \cdot L_{mCE} - (1 - \alpha) \cdot DSC$$

where:

$$L_{mCE} = - \frac{1}{N} \sum_{i=1}^{N} \left[ \beta \cdot y_i \ln(p_i) + (1 - \beta) \cdot \left(1 - y_i\right) \ln\left(1 - p_i\right) \right]$$

$$DSC = \sum_{i=1}^{N} \frac{2 y_i p_i}{y_i + p_i}$$

<!-- Optimizer Adam (with learning rate and weight decay) -->
<!-- Number of epochs, batch size -->
<!-- Network params (number of filters, dropout probability) -->
<!-- Metrics -->

### 📈 Results

<!-- Description of loss plot for F=32 -->

<p align="center">
  <img src="data/_readme-img/3-Loss_F32.png?raw=true" width="600" alt="Loss plot">
</p>

<!-- Metrics comparison for different number of filters -->

|Number of filters $F$|8|16|32|64|96|
|---------|:-----:|:-----:|:-----:|:-----:|:-----:|
|Number of parameters|125 820|500 076|1 993 932|7 963 020|17 907 276|
|Precision [%]|69.32|**69.84**|68.93|67.74|66.98|
|Recall [%]|40.08|41.94|**45.79**|44.04|45.29|
|F1-score [%]|50.7|52.31|**54.94**|53.29|53.94|

<!-- Moon: images -->

<p align="center">
  <img src="data/_readme-img/4-Results-Moon.png?raw=true" alt="Samples of Moon">
</p>

<!-- Mars: results (metrics for test set) -->

<p align="center">
  <img src="data/_readme-img/5-Results-Mars.png?raw=true" alt="Samples of Mars">
</p>