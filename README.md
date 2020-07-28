# Banking Purchase Predictor
## Udacity Machine Learning Nanodegree Capstone Project


This repository consists of a project conducted in July 2020, based on a 2018 Kaggle competition. The repository includes a zip file of the original Kaggle dataset, several notebooks, and a directory including processing scripts.

The competition can be found here: https://www.kaggle.com/c/santander-value-prediction-challenge/overview

The competition description is as follows:

> According to Epsilon research, 80% of customers are more likely to do business with you if you provide personalized service. Banking is no exception.
>
> The digitalization of everyday lives means that customers expect services to be delivered in a personalized and timely manner… and often before they´ve even realized they need the service. In their 3rd Kaggle competition, Santander Group aims to go a step beyond recognizing that there is a need to provide a customer a financial service and intends to determine the amount or value of the customer's transaction. This means anticipating customer needs in a more concrete, but also simple and personal way. With so many choices for financial services, this need is greater now than ever before.
>
> In this competition, Santander Group is asking Kagglers to help them identify the value of transactions for each potential customer. This is a first step that Santander needs to nail in order to personalize their services at scale.

Each notebook comprises a step in taken in the process of exploring and building a predictive model. They are intended to be used procedurally, with necessary files saved and uploaded to be used in subsequent notebooks as necessary. The intended sequence of use is as follows: 

![Banking Purchase Predictor Project Process](\Users\argon\OneDrive\Pictures\Workflow_Process.JPG "Banking Purchase Predictor Workflow")

Each notebook creates additional directories as needed. The first, called 'input data' holds train and test data unzipped from the main folder, as well as subsequent transformations to that data. The second created folder is called "models" and holds the artifacts of models created in the Modeling stage. The last created folder is called 'Predictions' and holds the final output from the Submission notebook. 

Because of computational limitations of my personal computer, i ran some algorithms through the Amazon SageMaker API. Documentation on how to access and use it can be found here: https://docs.aws.amazon.com/sagemaker/latest/dg/gs.html 

