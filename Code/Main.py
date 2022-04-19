import os
import warnings
from sys import platform

import pandas as pd
import torch
from transformers import pipeline

import GoogleTranslateWavToText
from Code import ResearchAnswers, QuestionRecorder, Retrieval

if platform == "darwin":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')


class Main(object):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.retrievers = {'abstract': Retrieval.TdifAnalysis(self.data[self.data.abstract.notna()].abstract), 'body_text': Retrieval.TdifAnalysis(self.data[self.data.body_text.notna()].body_text)}
        if not os.path.exists("Output/config.json") or not os.path.exists("Output/pytorch_model.bin"):
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-base-cased')
        else:
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'Output/')
            self.model.save_pretrained('Output')

    def run(self, asked_question, answers_article_length):
        return ResearchAnswers.search_answers_in_articles(self, asked_question, max_articles=answers_article_length)


data_url = 'Data/covid19.csv'
data = pd.read_csv(data_url)
main = Main(data_url)
if __name__ == '__main__':
    question = ""
    print("Select Input Option 1 or 2:\n")
    print("1: Speech")
    print("2: Text")
    user_input = str(input("Enter Input Option:"))
    print(user_input)
    while user_input != "1" and user_input != "2":
        print("You must select option 1 or 2.")
        user_input = str(input("Enter Input Option:"))
    if user_input == "1":
        QuestionRecorder.record_voice()
        question = GoogleTranslateWavToText.get_audio_transcription("Question/question.wav")
    else:
        question = str(input("Enter Question Text:"))
    article_length = 5
    print(main.run(question, 10))