# Serverless Machine Learning Inference with AWS Lambda + Amazon EFS
This repository contains supporting code for the medium article:
https://medium.com/@shitijkarsolia/setup-serverless-ml-inference-with-aws-lambda-efs-738546fa2e03.

   ![](https://i.ibb.co/S62Tzvf/Untitled-Diagram.png)

- The ```SageMaker_EFS_Lambda_Integration.ipynb``` is the SageMaker notebook used for training the ML model using SageMaker and storing it on EFS.
- The ```lambda_function.py``` is the AWS Lambda function code that is used for inference.
- The ```request_command``` file is an example of the final curl request that is to be sent to the API Endpoint which triggers the lambda function. The request can also be sent using Postman.

# Setup:
1. Clone this repository in your SageMaker Notebook:
```sh
$ git clone https://github.com/shitijkarsolia/ServerlessML-AWS-Lambda-EFS.git
```
2. Select the **Python3 Kernel** and follow the blog to run the cells in the notebook.

3. Copy the lambda function code to the function editor in your inference lambda function as mentioned in the article.

