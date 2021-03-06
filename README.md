Hi <img src="https://github.com/TheDudeThatCode/TheDudeThatCode/blob/master/Assets/Hi.gif" width="29px">, I'm Mike! 
------------------
<a href="https://www.linkedin.com/in/michael-ferko-24811997/">
  <img align="left" width="24px" src="https://cdn-icons-png.flaticon.com/512/174/174857.png"  />
</a>
<a href="https://twitter.com/MikeFerko_">
  <img align="left" width="26px" src="https://logodownload.org/wp-content/uploads/2014/09/twitter-logo-6.png" />
</a>
<a href="mailto:mike.w.ferko@gmail.com">
  <img align="left" width="26px" src="https://cdn-icons-png.flaticon.com/512/281/281769.png" />
</a>
<a href="https://www.instagram.com/michael.ferko/">
  <img align="left" width="26px" src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Instagram_icon.png/1024px-Instagram_icon.png" />
</a>
<br><br>
Click the arrows to see more!</br>
<br>
💼 My Recent Projects Include: 

<!DOCTYPE html>
<html>
<!--This is the V2X Research Seciton -->

<body>
  <div class = "dropdown-submenu">
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
    <summary>Machine Learning Projects:</summary>
<!-- 1. MLP Section -->
       <div>
       <details>
         <summary>1. Regression with Multiple Linear Perceptron (MLP) Modeling of the Saddle and Ackley Functions</summary>
         <li>Google Coolab Notebook: <a href="https://drive.google.com/file/d/17p5fgVgv836Nup1Jq5vYwrFuBrS3THVM/view?usp=sharing">Jupyter Notebook</a></li>
         <li>Github Repository: <a href="https://github.com/MikeFerko/Multiple-Linear-Perceptron-Modeling-of-the-Saddle-and-Ackley-Functions">Respository</a></li>
         <li>Paper: <a href="https://github.com/MikeFerko/Multiple-Linear-Perceptron-Modeling-of-the-Saddle-and-Ackley-Functions/blob/main/Regression%20with%20Multiple%20Linear%20Perceptron%20(MLP)%20Modeling%20of%20the%20Saddle%20and%20Ackley%20Functions.pdf">Regression with Multiple Linear Perceptron (MLP) Modeling of the Saddle and Ackley Functions</a></li>
         <li>MLP Machine Learning Algorithm:</li>
          <ol type="1">
            <li>Generate a data set with the simple Saddle Point or the Ackley Function</li>
              <ul>
                <li>Saddle Point:</li>
                <img src="https://latex.codecogs.com/gif.latex?z%28x%2Cy%29%20%3D%20x%5E%7B2%7D%20&plus;%20y%5E%7B2%7D"></img>
                <li>Ackley:</li>
                <img src="https://latex.codecogs.com/gif.latex?z%28x%2Cy%29%20%3D%20-20e%5E%7B%5Cfrac%7B1%7D%7B5%7D%20%5Csqrt%7B%5Cfrac%7B1%7D%7B2%7D%20%28x%5E%7B2%7D%20&plus;%20y%5E%7B2%7D%29%7D%7D%20-%20e%5E%7B%5Cfrac%7B1%7D%7B2%7D%28cos%7B%28%5Cpi%20x%7D%29%20&plus;cos%7B%28%5Cpi%20y%7D%29%29%7D"></img>
              </ul>
            <li>Add uniform random noise and visualize the 3D meshgrid</li>
            <li>Reshape the generated data to be a tensor input vector (shape will be: sample rows by feature columns)</li>
            <li>Regression MLP Model Parameters:</li>
            <ul>
              <li><a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent">Stochastic Gradient Descent optimizer</a></li>
              <ul>
                <li>Neural Network Architecture:</li>
                <ul>
                  <li>Input Layer = 10 neurons with a sigmoid activation function</li>
                  <li>Output Layer = 1 neuron</li>
                </ul>
                <li>Learing Rate = 0.1</li>
                <li>Exponential Decay Factor = 0</li>
                <li>Momentum = 0.1</li>
                <li>Train Duration: 50 Epochs</li>
                <li>Batch Size = 10</li>
               </ul>
              <li><a href="https://en.wikipedia.org/wiki/Mean_squared_error">Mean Square Error Loss Function</a></li>
            </ul>
            <li>Create a predicted Saddle point and Ackley Function from the Regression MLP trained Neural Network</li>
            <li>Plot the Results</li>
          </ol>
          <p align="center">
            <a href="https://drive.google.com/file/d/17p5fgVgv836Nup1Jq5vYwrFuBrS3THVM/view?usp=sharing">
            <img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/MLPModel.png" width="50%" height="50%">
            </a>
            <br>Multiple Linear Perceptron (MLP) Model</br>
          </p>
          <p align="center">
            <a href="https://drive.google.com/file/d/17p5fgVgv836Nup1Jq5vYwrFuBrS3THVM/view?usp=sharing">
            <img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/SaddlePointPredictions.png" width="75%" height="75%">
            </a>
            <br> Results of Saddle Function Predictions </br>
            <ol>
              <br>Results are shown in the above image Left-to-Right, Top-to-Bottom</br>
              <li>Real vs. Predicted Saddle</li>
              <li>z-x cross section @ y = 2</li>
              <li>z-x cross section @ y = 0</li>
              <li>Model Loss Vs. Epochs</li>
              <li>Topological Heat Map</li>
            </ol>
          </p>
          <p align="center">
            <a href="https://drive.google.com/file/d/17p5fgVgv836Nup1Jq5vYwrFuBrS3THVM/view?usp=sharing">
            <img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/AckleyPredictions.png" width="75%" height="75%">
            </a>
            <br> Results of Ackley Function Predictions </br>
            <ol>
              <br>Results are shown in the above image Left-to-Right, Top-to-Bottom</br>
              <li>Real vs. Predicted Ackley</li>
              <li>z-x cross section @ y = 2</li>
              <li>z-x cross section @ y = 0</li>
              <li>Model Loss Vs. Epochs</li>
              <li>Topological Heat Map</li>
            </ol>
          </p>
    </details>
    </div>

<!-- 2. Parametric Regression in Taipei Taiwan Section -->
   <div>
   <details>
     <summary>2. Real Estate Evaluation of housing prices in Taipei Taiwan</summary>
     <p>
       <ul>
         <li>Google Coolab Notebook: <a href="https://drive.google.com/file/d/17p5fgVgv836Nup1Jq5vYwrFuBrS3THVM/view?usp=sharing">Jupyter Notebook</a></li>
         <li>Github Repository: <a href="https://github.com/MikeFerko/Multiple-Linear-Perceptron-Modeling-of-the-Saddle-and-Ackley-Functions">Respository</a></li>
         <li>Paper: <a href="https://github.com/MikeFerko/Taipei-Taiwan-Regression-Modeling-of-Housing-Prices/blob/main/Real%20Estate%20Evaluation%20of%20housing%20prices%20in%20Taipei%20Taiwan.pdf">Real Estate Evaluation of housing prices in Taipei Taiwan</a></li>
         <li>We are using the same sequential MLP model used for the Saddle Point and Ackley Function preditctions.</li>
       </ul>
     </p>
     <p align="center">
       <a href="https://drive.google.com/file/d/17p5fgVgv836Nup1Jq5vYwrFuBrS3THVM/view?usp=sharing">
       <img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/MLPModel.png" width="50%" height="50%">
       </a>
       <br>Multiple Linear Perceptron (MLP) Model</br>
     </p>
     <p>We will be <a href="https://drive.google.com/file/d/1vAsnXHDkRoNFSS2KrWP39lFF4fS3J43O/view?usp=sharing">examining real estate valuation</a> which will help us understand where people tend to live in a city. The higher the price, the greater the demand to live in the property. Predicting real estate valuation can help urban design and urban policies, as it could help identify what factors have the most impact on property prices. Our aim is to predict real estate value, based on several features.
     </p>
     <br></br>
     <p>
      <ul>
        <li>Regression MLP Machine Learning on Taipei Taiwan Algorithm:</li>
        <ol type="1">
          <li>Load the Real estate valuation data set</li>
          <li>Independent feature vector containing:</li>
          <ol type="1" start="2">
            <li>X2 house age</li>
            <li>X3 distance to the nearest MRT station</li>
            <li>X4 number of convenience stores</li>
            <li>X5 latitude</li>
            <li>X6 longitude</li>
          </ol>
          <li>Train/Test split the data at a ratio of 80:20, respectively</li>
          <li>Min/Max Scale the dataset with a range of 0 to 1</li>
          <li>Normalise the scaled features</li>
          <li>Regression MLP Model Parameters:</li>
          <ul>
            <li>Neural Network Architecture:</li>
            <ul>
              <li>Input Layer = 10 neurons with a sigmoid activation function</li>
              <li>Output Layer = 1 neuron</li>
            </ul>
            <li><a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent">Stochastic Gradient Descent optimizer</a></li>
            <ul>
              <li>Learing Rate = 0.1</li>
              <li>Exponential Decay Factor = 0</li>
              <li>Momentum = 0.1</li>
             </ul>
            <li><a href="https://en.wikipedia.org/wiki/Mean_squared_error">Mean Square Error Loss Function</a></li>
            <li>Train Duration: 50 Epochs</li>
            <li>Batch Size = 10</li>
          </ul>
          <li>Create a predicted House Price Prediction of the unit area from the Regression MLP trained Neural Network</li>
          <li>Plot the Results</li>
        </ol>
      </ul>
     </p>
     
   <p align="center">
    <a href="https://drive.google.com/file/d/1i49EBOacHkSxA84ghQZCQ3JSvItULceT/view?usp=sharing">
    <img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/MLPRegressionLoss.png" width="50%" height="50%">
    </a>
    <br>Regression MLP Model Loss</br>
   </p>
   
   <p align="center">
    <a href="https://drive.google.com/file/d/1i49EBOacHkSxA84ghQZCQ3JSvItULceT/view?usp=sharing">
    <img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/RegressionPrediction.png" width="75%" height="75%">
    </a>
    <br>Regression MLP Predictions in New Taiwan Dollars (NT$)</br>
   </p>
     
  </details>
  </div>

<!--  3. Classification of MNIST 70,000 Handwritten Digits 0-9 Image Data Set -->
 
<div>
<details>
 <summary>3. Classification of MNIST 70,000 Handwritten Digits 0-9 Image Data Set</summary>     
<p>
  <ul>
    <li>Google Coolab Notebook: <a href="https://drive.google.com/file/d/1D7tv0AckARCQMVxbaBTvdb7UxYvrIIPe/view?usp=sharing">Jupyter Notebook</a></li>
    <li>Github Repository: <a href="https://github.com/MikeFerko/Classification-of-MNIST-70k-Handwritten-Digits-0-9-Image-Data-Set">Respository</a></li>
    <li>Paper: <a href="https://github.com/MikeFerko/Classification-of-MNIST-70k-Handwritten-Digits-0-9-Image-Data-Set/blob/main/Classification%20of%20MNIST%2070%2C000%20Handwritten%20Digits%200-9%20Image%20Data%20Set.pdf">Classification of MNIST 70,000 Handwritten Digits 0-9 Image Data Set</a></li>
    <li>Categorical Cross Entropy Algorithm:</li>
    <ol type="1">
      <li>Load the Modified National Institute of Standards and Technology (MNIST) Handwritten digits 0-9 data set</li>
      <li>Train/Test split the data at a ratio of 6:1, respectively</li>
      <li>Reshape the images from 28x28 pixels to 784x1 pixels</li>
      <li>Normalise the image pixels by dividing by the gray scale image intensity level set L:</li>
      <p align="center">
       <a href="https://drive.google.com/file/d/17p5fgVgv836Nup1Jq5vYwrFuBrS3THVM/view?usp=sharing">
       <img src="https://latex.codecogs.com/gif.latex?L%3D%5B0%2C2%5E%7Bk%7D-1%5D%3B%20k%3D8%20%5Crightarrow%20L%3D%5B0%2C255%5D%3B" width="35%" height="35%"></img>
       </a>
      </p>
      <li>Create 10 Categories for the 10 digits 0-9 to be classified</li>
      <br></br>
      <p align="center">
        <a href="https://drive.google.com/file/d/1D7tv0AckARCQMVxbaBTvdb7UxYvrIIPe/view?usp=sharing"><img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/encodingHandWrittenDigits.png" width="50%" height="50%"></img>
        </a>
        <br>Creating classes 10 classes for the 10 digits 0-9 of handwritten digits</br>
      </p>
      <br></br>
      <li>Categorical Cross Entropy (CE) Model Parameters:</li>
          <ul>
            <li><a href="https://en.wikipedia.org/wiki/Cross_entropy">categorical cross entropy (CE) Loss Function: </a></li>
            <p align="center">
            <a href="https://drive.google.com/file/d/1D7tv0AckARCQMVxbaBTvdb7UxYvrIIPe/view?usp=sharing">
            <img src="https://latex.codecogs.com/gif.latex?CE%20%3D%20-%5Csum_%7Bi%7D%5E%7BC%7Dt_%7Bi%7Dln%28s_%7Bi%7D%29" width="25%" height="25%"></img>
            </a>
            <br>Where: The formula can be seen as above, where  ti  refers to the  i -th element of the target vector and  si  refers to the  i -th element of the models output vector, and C the number of classes.</br>
            </p>
            <p align="center">
            <a href="https://drive.google.com/file/d/1D7tv0AckARCQMVxbaBTvdb7UxYvrIIPe/view?usp=sharing">
            <img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/logLossCrossEntropy.png" width="75%" height="75%"></img>
            </a>
            <br>Visualization of Log Loss (Cross Entropy)</br>
            </p>
            <p align="center">
            <a href="https://drive.google.com/file/d/1D7tv0AckARCQMVxbaBTvdb7UxYvrIIPe/view?usp=sharing">
            <img src="https://github.com/MikeFerko/Classification-of-MNIST-70k-Handwritten-Digits-0-9-Image-Data-Set/blob/main/LAB-2-Fig2.png" width="50%" height="50%"></img>
            </a>
            <br>Cross Entropy between probability distributions for each Class</br>
            </p>
            <li><a href="https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Accuracy">Model Accuracy: </a></li>
            <p align="center">
            <a href="https://drive.google.com/file/d/1D7tv0AckARCQMVxbaBTvdb7UxYvrIIPe/view?usp=sharing">
            <img src="https://latex.codecogs.com/gif.latex?Acc%3D%5Cfrac%7B1%7D%7BM%7D%5Csum_%7Bk%7D%5E%7BM%7Dargmax%28t_%7Bk%7D%29%20%3D%3D%20argmax%28s_%7Bk%7D%29" width="35%" height="35%"></img>
            </a>
            <br>Where: M is the number of samples in the dataset, tk is the target vector for the k-th sample, and sk is the models output vector for the k-th sample.</br>
            </p>
            <li>Neural Network Architecture:</li>
            <ul>
              <li>Input Layer = 16 hyperbolic tangent activation (tanh) neurons with an input shape of 784x1</li>
              <li>Hidden Layer = 16 hyperbolic tangent activation (tanh) neurons with an input shape of 16x1</li>
              <li>Output Layer = 10 softmax neurons</li>
              <p align="center">
              <a href="https://drive.google.com/file/d/1D7tv0AckARCQMVxbaBTvdb7UxYvrIIPe/view?usp=sharing">
              <img src="https://github.com/MikeFerko/Classification-of-MNIST-70k-Handwritten-Digits-0-9-Image-Data-Set/blob/main/LAB-2-Fig1.png" width="75%" height="75%"></img>
              </a>
              <br>Classification Neural Network Architecture</br>
              </p>
            </ul>
            <li><a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent">Stochastic Gradient Descent optimizer</a></li>
            <ul>
              <li>Learing Rate = 0.4</li>
              <li>Exponential Decay Factor = 0</li>
              <li>Momentum = 0.5</li>
            </ul>
            <li>Train Duration: 10 Epochs</li>
            <li>Batch Size = 128</li>
            <li>training samples = 60,000</li>
            <li>testing samples = 10,000</li>  
          </ul>
  </ul>
</p>
<br>7. Show Results: </br>
<p align="center">

<a href="https://drive.google.com/file/d/1D7tv0AckARCQMVxbaBTvdb7UxYvrIIPe/view?usp=sharing">
<img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/MNIST_handwritten_Digits_Results.png" width="50%" height="50%"></img>
</a>
<br>Visualization of Model Loss and Accuracy (0.1532 and 95.49% Respectively)</br>
</p>

<p align="center">
<a href="https://drive.google.com/file/d/1D7tv0AckARCQMVxbaBTvdb7UxYvrIIPe/view?usp=sharing">
<img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/VisualizationOfFirstLayerWeightsW1.png" width="50%" height="50%"></img>
</a>
<br>Visualization of First Layer Weights W1 from Neural Network Architecture</br>
</p>

<p align="center">
<a href="https://drive.google.com/file/d/1D7tv0AckARCQMVxbaBTvdb7UxYvrIIPe/view?usp=sharing">
<img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/VisualizationOfSecondLayerWeightsW2.png" width="50%" height="50%"></img>
</a>
<br>Visualization of Second Layer Weights W2 from Neural Network Architecture</br>
</p>

<p align="center">
<a href="https://drive.google.com/file/d/1D7tv0AckARCQMVxbaBTvdb7UxYvrIIPe/view?usp=sharing">
<img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/VisualizationOfThirdLayerWeightsW3.png" width="50%" height="50%"></img>
</a>
<br>Visualization of Third Layer Weights W3 from Neural Network Architecture</br>
</p>
</details>
</div>
 
 
 <!-- 4. Classification of MNIST Fashion Data set -->
 
<div>
<details>
 <summary>4. Classification of Fashion MNIST Image Data Set</summary>     
<p>
  <ul>
    <li>Google Coolab Notebook: <a href="https://drive.google.com/file/d/197UP-kVRMQzCfOPd9AiP7Z4qxwMn9O54/view?usp=sharing">Jupyter Notebook</a></li>
    <li>Github Repository: <a href="https://github.com/MikeFerko/Classification-of-Fashion-MNIST">Respository</a></li>
    <li>Paper: <a href="https://github.com/MikeFerko/Classification-of-Fashion-MNIST/blob/main/Lab%202%20MNIST-Fashion%20Assignment_Michael%20Ferko.pdf">Classification of Fashion MNIST Image Data Set</a></li>
    <li>Categorical Cross Entropy Algorithm:</li>
    <ol type="1">
      <li>Load the Modified National Institute of Standards and Technology (MNIST) Fashion data set</li>
      <li>Train/Test split the data at a ratio of 6:1, respectively</li>
      <li>Reshape the images from 28x28 pixels to 784x1 pixels</li>
      <li>Normalise the image pixels by dividing by the gray scale image intensity level set L:</li>
      <p align="center">
       <a href="https://drive.google.com/file/d/197UP-kVRMQzCfOPd9AiP7Z4qxwMn9O54/view?usp=sharing">
       <img src="https://latex.codecogs.com/gif.latex?L%3D%5B0%2C2%5E%7Bk%7D-1%5D%3B%20k%3D8%20%5Crightarrow%20L%3D%5B0%2C255%5D%3B" width="35%" height="35%"></img>
       </a>
      </p>
      <li>Create 10 Categories for class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']</li>
      <br></br>
      <p align="center">
        <a href="https://drive.google.com/file/d/197UP-kVRMQzCfOPd9AiP7Z4qxwMn9O54/view?usp=sharing"><img src="https://github.com/MikeFerko/Classification-of-Fashion-MNIST/blob/main/Fashion-MNIST-Dataset-GrayScale.png" width="50%" height="50%"></img>
        </a>
        <br>Creating 10 classes for the 10 types of clothing in the Image Data Set</br>
      </p>
      <br></br>
      <li>Categorical Cross Entropy (CE) Model Parameters:</li>
          <ul>
            <li><a href="https://en.wikipedia.org/wiki/Cross_entropy">categorical cross entropy (CE) Loss Function: </a></li>
            <p align="center">
            <a href="https://drive.google.com/file/d/197UP-kVRMQzCfOPd9AiP7Z4qxwMn9O54/view?usp=sharing">
            <img src="https://latex.codecogs.com/gif.latex?CE%20%3D%20-%5Csum_%7Bi%7D%5E%7BC%7Dt_%7Bi%7Dln%28s_%7Bi%7D%29" width="25%" height="25%"></img>
            </a>
            <br>Where: The formula can be seen as above, where  ti  refers to the  i -th element of the target vector and  si  refers to the  i -th element of the models output vector, and C the number of classes.</br>
            </p>
            <p align="center">
            <a href="https://drive.google.com/file/d/197UP-kVRMQzCfOPd9AiP7Z4qxwMn9O54/view?usp=sharing">
            <img src="https://github.com/MikeFerko/MikeFerko/blob/main/images/logLossCrossEntropy.png" width="75%" height="75%"></img>
            </a>
            <br>Visualization of Log Loss (Cross Entropy)</br>
            </p>
            <p align="center">
            <a href="https://drive.google.com/file/d/197UP-kVRMQzCfOPd9AiP7Z4qxwMn9O54/view?usp=sharing">
            <img src="https://github.com/MikeFerko/Classification-of-MNIST-70k-Handwritten-Digits-0-9-Image-Data-Set/blob/main/LAB-2-Fig2.png" width="50%" height="50%"></img>
            </a>
            <br>Cross Entropy between probability distributions for each Class</br>
            </p>
            <li><a href="https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Accuracy">Model Accuracy: </a></li>
            <p align="center">
            <a href="https://drive.google.com/file/d/197UP-kVRMQzCfOPd9AiP7Z4qxwMn9O54/view?usp=sharing">
            <img src="https://latex.codecogs.com/gif.latex?Acc%3D%5Cfrac%7B1%7D%7BM%7D%5Csum_%7Bk%7D%5E%7BM%7Dargmax%28t_%7Bk%7D%29%20%3D%3D%20argmax%28s_%7Bk%7D%29" width="35%" height="35%"></img>
            </a>
            <br>Where: M is the number of samples in the dataset, tk is the target vector for the k-th sample, and sk is the models output vector for the k-th sample.</br>
            </p>
            <li>Neural Network Architecture:</li>
            <ul>
              <li>Input Layer = 64 ReLu activation neurons with an input shape of 784x1</li>
              <li>Hidden Layer = 64 ReLu activation neurons with an input shape of 64x1</li>
              <li>Output Layer = 10 softmax neurons</li>
              <p align="center">
              <a href="https://drive.google.com/file/d/197UP-kVRMQzCfOPd9AiP7Z4qxwMn9O54/view?usp=sharing">
              <img src="https://github.com/MikeFerko/Classification-of-Fashion-MNIST/blob/main/Fashion-MNIST-Neural-Network-Architecture.png" width="75%" height="75%"></img>
              </a>
              <br>Classification Neural Network Architecture</br>
              </p>
            </ul>
            <li><a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent">Stochastic Gradient Descent optimizer</a></li>
            <ul>
              <li>Learing Rate = 0.1</li>
              <li>Exponential Decay Factor = 0</li>
              <li>Momentum = 0</li>
            </ul>
            <li>Train Duration: 10 Epochs</li>
            <li>Batch Size = 128</li>
            <li>training samples = 60,000</li>
            <li>testing samples = 10,000</li>  
          </ul>
  </ul>
</p>
<br>7. Show Results: </br>
<p align="center">

<a href="https://drive.google.com/file/d/197UP-kVRMQzCfOPd9AiP7Z4qxwMn9O54/view?usp=sharing">
<img src="https://github.com/MikeFerko/Classification-of-Fashion-MNIST/blob/main/Fashipn-MNIST-Training.png" width="50%" height="50%"></img>
</a>
<br>Visualization of Model Loss and Accuracy (0.3090 and 88.66% Respectively)</br>
</p>

<p align="center">
<a href="https://drive.google.com/file/d/197UP-kVRMQzCfOPd9AiP7Z4qxwMn9O54/view?usp=sharing">
<img src="https://github.com/MikeFerko/Classification-of-Fashion-MNIST/blob/main/Fashion-MNIST-FirstLayerWeightVisualization.png" width="50%" height="50%"></img>
</a>
<br>Visualization of First Layer Weights W1 from Neural Network Architecture</br>
</p>

<p align="center">
<a href="https://drive.google.com/file/d/197UP-kVRMQzCfOPd9AiP7Z4qxwMn9O54/view?usp=sharing">
<img src="https://github.com/MikeFerko/Classification-of-Fashion-MNIST/blob/main/Fashion-MNIST-SecondLayerWeightVisualization.png" width="50%" height="50%"></img>
</a>
<br>Visualization of Second Layer Weights W2 from Neural Network Architecture</br>
</p>

<p align="center">
<a href="https://drive.google.com/file/d/197UP-kVRMQzCfOPd9AiP7Z4qxwMn9O54/view?usp=sharing">
<img src="https://github.com/MikeFerko/Classification-of-Fashion-MNIST/blob/main/Fashion-MNIST-ThirdLayerWeightVisualization.png" width="50%" height="50%"></img>
</a>
<br>Visualization of Third Layer Weights W3 from Neural Network Architecture</br>
</p>
     
</details>
</div>
    
   </details>
</div>

<!--             
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
    - a) Bayesian Binary Classification on Multidimensional Multivariate Distributions: [Paper](https://github.com/MikeFerko/Bayesian_Classification_on_parametric_distributions-main/blob/main/Bayesian_Classification_on_parametric_distributions.pdf), [Jupyter Notebook](https://drive.google.com/file/d/1PrarAQY7YOEo5WYrurrUJ6qez1bfK7Go/view?usp=sharing)
    - b) Density Estimation and Applications in Estimation, Clustering and Segmentation: [Paper](https://github.com/MikeFerko/Density_Estimation_and_Basics_of_Segmentation/blob/main/Density%20Estimation%20and%20Basics%20of%20Segmentation%20by%20EM%20Method.pdf), [Jupyter Notebook](https://drive.google.com/file/d/10jKmI9C6KkQMauWRrTybLOt6lAKuWJzh/view?usp=sharing)
    - c) Feature Extraction and Object Recognition: [Paper](https://github.com/MikeFerko/Feature_Extraction_and_Object_Recognition/blob/main/Feature%20Extraction%20and%20Object%20Recognition.pdf), [Jupyter Notebook](https://drive.google.com/file/d/1u2sCnsTK3nsf0bK9xaXRHGNrXJo7Sq6R/view?usp=sharing)
    - d) Principle Component Analysis (PCA) and Linear Discriminant Analysis (LDA): [Paper](https://github.com/MikeFerko/PCA_and_LDA/blob/main/Principal%20Component%20and%20Linear%20Discriminant%20Analyses.pdf), [Jupyter Notebook](https://drive.google.com/file/d/1zBjnGCHLn8ziEZCQEgotm3c_JaAfJ_aG/view?usp=sharing)

</html>
