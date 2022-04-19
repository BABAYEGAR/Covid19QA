# Covid-19 Question & Answer


## Summary
This repository contains [Haruna Salim](https://github.com/BABAYEGAR)'s final project for George Washington University's DATS 6312: Natural Language Processing course.
My project objective was to apply transformers question answering and summarizer to answer questions related to Covid-19. For this project applied techniques such as cosine similarity to rank articles, transformers to generate and summarize answers to questions.

## Data
[Data Source 1](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge)   

## Folders
* Code contains all of our code and data used in the project.
* Proposal contains the proposal for the project.
* Presentation contains a PowerPoint presentation of our project.
* Report contains a report of our findings from this project.

## Code Execution

1. To successfully execute the code, make sure you have following libraries installed on your python interpreter enviroment:

* sklearn
* pytorch 
* urllib
* pydub
* transformers
* abc
* pandas
* attr


2. To generate the dataset, execute the DataGenerator.py file. 

3. To start the application, run the main.py file. You can select either speech or text as user input by entering 1 or 2. if you select speech, the program will prompt when the recorder starts and the timeout is 10s. If you select text option then you simply type your question.

4. The program will check the Output folder and if the there is no saved transformer model, the program will train and write the model to file

5. The program will return a dataframe with answers ranked by the start score.
