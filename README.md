# EEG-Classifiers-Ensemble: Real-Time Data Classification System

## Table of Contents
1. [Processing Methods](https://github.com/AlmaCuevas/voting_system_platform/tree/main#processing-methods)
2. [Datasets used](https://github.com/AlmaCuevas/voting_system_platform/tree/main#datasets-used)
3. [References](https://github.com/AlmaCuevas/voting_system_platform/tree/main#references)

## Processing Methods

### LSTM
Abdulghani, Walters, and Abed [2], Agarwal and Kumar [3], and Kumar and Scheme [4] use Long-Short Term Memory (LSTM) to
classify speech imagery. An LSTM, a recurrent neural network, can learn long-term dependencies between the discrete
steps in a time series data. The code used was obtained from GitHub by [5].

### GRU

### DiffE

### ShallowFBCSPNet

### Spatial Features

#### XDAWN+RG
The XDAWN spatial filter and Riemannian Geometry classifier (RG) algorithm [7] achieved an accuracy of 0.836. Riemannian
Geometry represents data as symmetric positive definite covariance matrices and maps them onto a specific geometric space.
It can be computationally intensive when dealing with high-dimensional data, so dimensionality reduction techniques like
XDAWN spatial filters are used alongside it. The approach is based on the idea that tasks like the P300-speller, EEG signals,
and mental states have a degree of invariance that covariance matrices can capture. Due to its logarithmic nature, the Riemann
distance is robust to noise. This method can potentially reduce or eliminate the calibration phase, especially when limited
training data is available.

#### Common Spatial Patterns

#### Covariances

### Time Features

These features are organized into columns with descriptive names, facilitating feature selection. The
resulting table of features serves as the input for classifiers, enabling the analysis of EEG signals.

#### EEGExtract
EEGExtract is a feature extraction code designed to process EEG data, here the input is segmented into various frequency
bands before feeding it to the extraction process. For each frequency band, EEGExtract computes a set of features,
including entropy, mobility, complexity, ratio, and Lyapunov exponent.

#### Statistical variables

##### Mean
The average value of the signal.
##### Skewness
A measure of the asymmetry of the probability distribution of the signal values.
##### Kurtosis
A measure of the “tailedness” of the probability distribution of the signal values.
##### Standard Deviation (Std)
A measure of the amount of variation or dispersion of the signal values.
##### Variance
The square of the standard deviation, representing the spread of the signal values.

##  Datasets used:
### [Aguilera](https://data.mendeley.com/datasets/57g8z63tmy/1) [15]
### [Nieto](https://openneuro.org/datasets/ds003626/versions/2.1.2) [16]

Their code is embedded in this repository to load their dataset. The original is [Inner_Speech_Dataset](https://github.com/N-Nieto/Inner_Speech_Dataset)

Ten Argentinian participants were involved in this experimental setup, and data from 136 channels were recorded, with 128 dedicated to EEG readings and 8 to muscle activity. The experiment focused on eliciting four specific commands from the subjects, namely ”arriba,” ”abajo,” ”derecha,” and ”izquierda,” corresponding to ”up,” ”down,” ”right,” and ”left.” To explore inner speech processes, each participant was instructed to engage in a mental exercise involving repeatedly imagining their voice and uttering the respective word.

### [Coretto](https://drive.google.com/file/d/0By7apHbIp8ENZVBLRFVlSFhzbHc/view?resourcekey=0-JVHv2UiRsxim41Wioro0EA) [17]
The Coretto dataset consists of 15 Argentinian subjects who are native Spanish speakers, with an average age of 25 years old. These subjects repeated words, including vowels and directional words, 50 times each at a sample frequency of 1024Hz. The words were visually presented, and the recordings were single takes. The dataset used the Geschwind-Wernicke model, focusing on specific electrode locations to minimize myoelectric noise interference during speech.

### Torres (Data available on request from the original authors) [18]
This dataset comprises the EEG signals of 27 right-handed subjects performing internal pronunciation of words without emitting sounds or doing facial movements. It is focused on
the recognition of five Spanish words corresponding to the English words “up,” “down,” “left,” “right,” and “select,” with which a computer cursor could be controlled. Unlike the other datasets, this one is not open-access and was kindly made available by interinstitutional agreement.

### [2020 International BCI Competition](https://osf.io/pq7vb/) [19]


# References

1. Miao, Z., Zhang, X., Zhao, M. & Ming, D. Lmda-net: a lightweight multi-dimensional attention network for general eeg-based brain-computer interface paradigms and interpretability. 10.48550/ARXIV.2303.16407 (2023).
2. Abdulghani, M. M., Walters, W. L. & Abed, K. H. Imagined speech classification using eeg and deep learning. Bioengineering 10, 649, 10.3390/bioengineering10060649 (2023).
3. Agarwal, P. & Kumar, S. Electroencephalography-based imagined speech recognition using deep long short-term memory network. ETRI J. 44, 672–685, 10.4218/etrij.2021-0118 (2022).
4. Kumar, P. & Scheme, E. A deep spatio-temporal model for eeg-based imagined speech recognition. ICASSP 2021 - 2021 IEEE Int. Conf. on Acoust. Speech Signal Process. (ICASSP) 10.1109/icassp39728.2021.9413989 (2021).
5. C. Brunner, G. R. M.-P. A. S. G. P., R. Leeb. Bigproject.
6. Nouri, M., Moradi, F., Ghaemi, H. & Motie Nasrabadi, A. Towards real-world bci: Ccspnet, a compact subject-independent motor imagery framework. Digit. Signal Process. 133, 103816, 10.1016/j.dsp.2022.103816 (2023).
7. Barachant, A. et al. pyriemann/pyriemann: v0.5, 10.5281/zenodo.8059038 (2023).
8. Saba-Sadiya, S., Chantland, E., Alhanai, T., Liu, T. & Ghassemi, M. M. Unsupervised eeg artifact detection and correction.
Front. Digit. Heal. 2, 57 (2020).
9. Kim, S., Lee, Y.-E., Lee, S.-H. & Lee, S.-W. Diff-e: Diffusion-based learning for decoding imagined speech eeg. (2023). 2307.14389.
10. Liu, X. et al. Tcacnet.
11. Liu, X. et al. Tcacnet: Temporal and channel attention convolutional network for motor imagery classification of eeg-based
bci. Inf. Process. amp; Manag. 59, 103001, 10.1016/j.ipm.2022.103001 (2022).
12. Lawhern, V. J. et al. Eegnet: a compact convolutional neural network for eeg-based brain–computer interfaces. J. Neural Eng. 15, 056013 (2018).
13. Tibor, S. R. et al. Deep learning with convolutional neural networks for eeg decoding and visualization. Hum. Brain Mapp. 38, 5391–5420, 10.1002/hbm.23730.
14. project, A. R. L. A. E. Army research laboratory (arl) eegmodels.
15. Aguilera-Rodríguez, E. Imagined speech datasets applying traditional and gamified acquisition paradigms, 10.17632/57G8Z63TMY.1 (2024).
16. Nieto, N., Peterson, V., Rufiner, H., Kamienkowski, J. & Spies, R. "inner speech", doi:10.18112/openneuro.ds003626.v2.1.2 (2022)
17. Coretto, G. A. P., Gareis, I. E. & Rufiner, H. L. Open access database of eeg signals recorded during imagined speech. In Symposium on Medical Information Processing and Analysis (2017)
18. A. A. Torres-García, C. A. Reyes-García, L. Villaseñor-Pineda, and J. M. Ramírez-Cortes, “Análisis de Señales Electroencefalográficas para la Clasificación de Habla Imaginada,” Revista Mexicana de Ingeniería Biomédica, vol. 34, no. 1, pp. 23–39, 2013. ISSN: 0188-9532.
19. Committee, B. 2020 international bci competition. Open Sci. Framew. 10.17605/OSF.IO/PQ7VB (2022)
