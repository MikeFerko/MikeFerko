How to Reach me: 
  - Email: mike.w.ferko@gmail.com
  - Twitter: [@MikeFerko_](https://twitter.com/MikeFerko_)

💼 My Recent Projects Include: 

<!-- This is the V2X Research Seciton -->

<div class = "dropdown">
  <details>
    <summary>
      <head>Research: <a href="https://github.com/MikeFerko/Deep-Reinforcement-Learning-for-V2V-Communication">Reinforcement Learning for V2V Communication</a>
      </head>
    </summary>
       <ul>
         <li>Artifically Intelligent (AI) form of electronic communications between a vehicle and everything (V2X): </li>
          <ul>
            <li><a href="https://youtu.be/9g32v7bK3Co">Markov Decision Processeses (MDP)</a>
            <li><a href="https://ieeexplore.ieee.org/document/8450518">Deep Reinforcement Learning Framework (DRL a.k.a. DQL)</a>
            <li><a href="https://ieeexplore.ieee.org/document/8052521">Orthogonal Frequency Division Multiplexing (OFDM)</a>
            <li><a href="https://arxiv.org/abs/1710.02298">State-of-the-Art DQN C51 Rainbow</a>
            <li><a href="https://ojs.aaai.org/index.php/AAAI/article/view/11791">Google Deep Mind's progress in Quantile Regression</a>
          </ul>
    </ul>
       <p align="center">
        <a href="https://github.com/MikeFerko/Deep-Reinforcement-Learning-for-V2V-Communication/blob/main/Distributed%20Deep%20Reinforcement%20Learning%20for%20V2V%20Communication.pdf">
           <img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/structureOfVehiclularComunicationsNetwork.png"
           width="75%" height="75%">
        </a>
        <br> Structure of Vehiclular Communicaitons Network
   </p>
   </details>
</div>
  
<!-- This is the Machine Learning Seciton -->

<div class = "dropdown">
  <details>
      <ul>
        <sumamry>
          <a href="https://github.com/MikeFerko/Taipei-Taiwan-Regression-Modeling-of-Housing-Prices">Multiple Linear Perceptron Modeling of the Saddle and Ackley Functions</a>
        </summary>
          <ul>
            <li>Algorithm: Regression with Multilayer Perceptrons (MLP)</li>
            <ul>
              <li>Generate a data set with the simple Saddle Point or the Ackley Function</li>
                <ul>
                  <li>Saddle Point:</li>
                    <img src="https://latex.codecogs.com/gif.latex?z%28x%2Cy%29%20%3D%20x%5E%7B2%7D%20&plus;%20y%5E%7B2%7D">
                    </img>
                  <li>Ackley:</li> 
                    <img src="https://latex.codecogs.com/gif.latex?z%28x%2Cy%29%20%3D%20-20e%5E%7B%5Cfrac%7B1%7D%7B5%7D%20%5Csqrt%7B%5Cfrac%7B1%7D%7B2%7D%20%28x%5E%7B2%7D%20&plus;%20y%5E%7B2%7D%29%7D%7D%20-%20e%5E%7B%5Cfrac%7B1%7D%7B2%7D%28cos%7B%28%5Cpi%20x%7D%29%20&plus;cos%7B%28%5Cpi%20y%7D%29%29%7D">
                    </img>
            - 2) Add uniform random noise and visualize the 3D meshgrid 
            - 2) Reshape the generated data to be a tensor input vector (shape will be: sample rows by feature columns)
            - 3) Create and Train/Test a sequential MLP model
            - 4) Plot the Results

              <p>
                <a href="https://drive.google.com/file/d/17p5fgVgv836Nup1Jq5vYwrFuBrS3THVM/view?usp=sharing">
                   <img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/MLPModel.png"
                   width="75%" height="75%">
                </a>
                <br> Multiple Linear Perceptron (MLP) Model </br>
              </p>

              <p>
                <br></br>
                <a href="https://drive.google.com/file/d/17p5fgVgv836Nup1Jq5vYwrFuBrS3THVM/view?usp=sharing">
                   <img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/SaddlePointPredictions.png"
                   width="100%" height="100%">
                </a>
                <br> Saddle Point Results Left-to-Right, Top-to-Bottom: </br>
                <ul>
                      <li>Real vs. Predicted Saddle</li>
                      <li>z-x cross section @ y = 2</li>
                      <li>z-x cross section @ y = 0</li>
                      <li>Model Loss Vs. Epochs</li>
                      <li>Topological Heat Map</li>
              </ul>
              </p>
  </details>
</div>
<!--             
            <p>
              <a href="https://drive.google.com/file/d/17p5fgVgv836Nup1Jq5vYwrFuBrS3THVM/view?usp=sharing">
                 <img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/AckleyPredictions.png"
                 width="100%" height="100%">
              </a>
              <br> Ackley Results Left-to-Right, Top-to-Bottom: </br>
              <ul>
                    <li>Real vs. Predicted Ackley</li>
                    <li>z-x cross section @ y = 2</li>
                    <li>z-x cross section @ y = 0</li>
                    <li>Model Loss Vs. Epochs</li>
                    <li>Topological Heat Map</li>
              </ul>
            </p>


    - b) [Classification of MNIST 70,000 Handwritten Digits 0-9 Image Data Set](https://github.com/MikeFerko/Classification-of-Fashion-MNIST)
      - Algorithm: Supervised Categorical Cross Entropy
        - 1) Train/Test split the 70,000 images into 60,000 and 10,000 images respectively.
        - 2) vectorize the images by reshaping the images from 28 x 28 pixels to 784 x 1 pixels


      <p align="left">
  
        <img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/encodingHandWrittenDigits.png" 
             title="Encoding Handwritten Digits Images as Binary "
             width="25%"
             height="25%" img/>
  
        <img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/firstLayerWeightVisualization.png" 
                   width="40%"
                   height="40%" img/>
      </p>
      
    - c) Classification of Fashion MNIST Image Data Set: [Jupyter Notebook](https://github.com/MichaelFerko/Classification-of-Fashion-MNIST/blob/main/Lab%202%20Assignment%20Notebook.ipynb) 
      
   
    - d) Autoencoders for compression and denoising of MNIST Handwritten Digits Image Data Set: [Jupyter Notebook](https://github.com/MichaelFerko/Auto-encoders-Compression-Denoising-MNIST-Digits-0-9/blob/main/LAB3-Autoencoders_for_Compression_and_Denoising_final.ipynb) 
    - e) Classification of CIFAR-10/100 with Convolutional Neural Networks: [Jupyter Notebook](https://github.com/MichaelFerko/Classification-of-CIFAR-10-and-100-with-Convolutional-Neural-Networks/blob/main/Copy%20of%20LAB%204-Classification%20with%20CNN.ipynb) -->


<!-- This is the Pattern Recognition Seciton -->
  - Pattern Recognition Projects:
    - a) Bayesian Binary Classification on Multidimensional Multivariate Distributions: [Paper](https://github.com/MichaelFerko/Bayesian_Classification_on_parametric_distributions/blob/main/Bayesian_Classification_on_parametric_distributions.pdf), [Jupyter Notebook](https://github.com/MichaelFerko/Bayesian_Classification_on_parametric_distributions/blob/main/Bayesian_Classification_on_parametric_distributions.ipynb)
    - b) Density Estimation and Applications in Estimation, Clustering and Segmentation: [Paper](https://github.com/MichaelFerko/Density_Estimation_and_Basics_of_Segmentation/blob/main/Density%20Estimation%20and%20Basics%20of%20Segmentation%20by%20EM%20Method.pdf), [Jupyter Notebook](https://github.com/MichaelFerko/Density_Estimation_and_Basics_of_Segmentation/blob/main/Density%20Estimation%20and%20Applications%20in%20Estimation%2C%20Clustering%20%20and%20Segmentation.ipynb)
    - c) Feature Extraction and Object Recognition: [Paper](https://github.com/MichaelFerko/Feature_Extraction_and_Object_Recognition/blob/main/Feature%20Extraction%20and%20Object%20Recognition.pdf), [Jupyter Notebook](https://github.com/MichaelFerko/Feature_Extraction_and_Object_Recognition/blob/main/Feature_Extraction_and_Object_Recognition.ipynb)
    - d) Principle Component Analysis (PCA) and Linear Discriminant Analysis (LDA): [Paper](https://github.com/MichaelFerko/PCA_and_LDA/blob/main/Principal%20Component%20and%20Linear%20Discriminant%20Analyses.pdf), [Jupyter Notebook](https://github.com/MichaelFerko/PCA_and_LDA/blob/main/PCA_and_LDA.ipynb)
