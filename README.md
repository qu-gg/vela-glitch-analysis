<h1 align='center'>Single Pulse Analyses of the Vela Pulsar <br> using Machine Learning Techniques<br>
   [<a href='https://academic.oup.com/mnras/article-abstract/509/4/5790/6433657'>Vela-Pulsar</a>, <a href=''>Vela-Glitch</a>]</h2>
<p>This repository holds the experiments and models as explored in the works, "Vela pulsar: single pulses analysis with machine learning techniques" and "First glitching Pulsars monitoring program at the Argentine Institute of Radioastronomy" as published in <a href="https://academic.oup.com/mnras">MNRAS</a>. We provide training and visualization scripts. Data generated by our calculations or observations are available from the corresponding authors upon reasonable request.</p>

### Overview

We leverage Variational Autoencoders (VAEs) and Self-Organizing Maps (SOMs) to enable individual pulse reconstruction and clustering from the Vela Pulsar (SR B0833-45 / J0835-4510) given daily observations ranging from 1-3 hours and 50-100k pulses. The first works presents the initial presentation of the technique when applied to pulsar data and highlights the results on 4 days of observation from January and March of 2021. Using these techniques, we were able to isolate 'mini-giant' pulses effectively  into clusters and highlight the trend of increasing signal peak amplitude with earlier pulse arrival times, supporting earlier pulsar models suggesting pulsar emitting regions of different heights in thheir magnetospheres.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/182661125-c86a6805-7bf8-4b53-8f1c-6488a6b103a9.png" alt="framework schematic" )/></p>
<p align='center'>Fig 1. Schematic of the prposed VAE-SOM model used in analysis.</p>
