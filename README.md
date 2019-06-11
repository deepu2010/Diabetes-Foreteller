# Diabetes-Foreteller
Diabetes Foreteller is a mobile application used to calculate the probability of diabetes for a person based on the key symptoms that influence diabetes.

I have developed an android application named ‘Diabetes Foreteller’ which helps the user to assess his/her probability of having diabetes by analyzing the various symptoms of diabetes and the lifestyle of the user. Suggesting the probability of having diabetes to the user, creates an awareness and cautions him about the detrimental health effects caused by the diabetes. In addition to this, my application also recommends the preventive measures[15] for avoiding the diabetes in one’s lifetime.

**About the Dataset:**


For my application, I have made use of PIMA indians diabetes dataset. PIMA is a top-class medical institute released a diabetes dataset with the key factors that influence diabetes. The dataset provides accurate information of all the patients and provides reliability for the students/professionals to use this data in real time projects. 

The dataset includes data from 768 women with 8 characteristics, in particular:
1. Number of times pregnant
2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. Diastolic blood pressure (mm Hg)
4. Triceps skin fold thickness (mm)
5. 2-Hour serum insulin (mu U/ml)
6. Body mass index (weight in kg/(height in m)^2)
7. Diabetes pedigree function
8. Age (years)


**Context Diagram of Diabetes Foreteller**
![Alt Text](https://raw.githubusercontent.com/deepu2010/Diabetes-Foreteller/master/Images/Context%20diagram.JPG)

**Use Case Diagram of Diabetes Foreteller application**
![Alt Text](https://raw.githubusercontent.com/deepu2010/Diabetes-Foreteller/master/Images/Use%20case%20diagram.JPG)

**Machine Learning Model Details:**
Rather than relying on popular machine learning frameworks such as Keras, TensorFlow, I wanted to challenge myself by building a neural network completely on my own. 
Based on my experience with Deep Learning, I have followed a methodology I learnt from ‘Deep Learning Specialization’ by Andrew Ng.

**METHODOLOGY FOLLOWED:**
1. Initialize parameters / Define hyperparameters
2. Loop for number of iterations:
    * Forward propagation
    * Compute cost function
    * Backward propagation
    * Update parameters (using parameters, and grads from backprop) 
4. Use trained parameters to predict labels

###### Architecture of neural network for diabetes foreteller
* I have built a deep neural network with 6 layers.
* Hence, my model has 5 hidden layers and one output layer
* My model is mentioned as,
   **[LINEAR -> RELU]  ×  (L-1) -> LINEAR -> SIGMOID**
   

![Alt Text](https://raw.githubusercontent.com/deepu2010/Diabetes-Foreteller/master/Images/Architecture%20of%20Neural%20Network.JPG)

**Model Training details:**
1. Learning Rate = 0.0075
2. Num of iterations = 8000
3. Regularization = ‘Dropout Regularization’
4. Early stopping = True
5. Optimization = ‘Adam Optimization’
6. Cost function used: Entropy loss function

**Results Achieved**

* **Accuracy:** 86%
* **F1 Score (training set):** 1.0
* **F1 Score (test set):** 0.79

