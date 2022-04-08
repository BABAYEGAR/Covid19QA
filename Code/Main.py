import os
import warnings
from sys import platform

import pandas as pd
import torch

import SpeechToText
import TdifAnalysis
from Code import ResearchAnswers, QuestionRecorder

if platform == "darwin":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')


class Main(object):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        if not os.path.exists("Output/config.json") or not os.path.exists("Output/pytorch_model.bin"):
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering',
                                        'bert-base-cased')
        else:
            self.model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'Output/')
            self.model.save_pretrained('Output')
        self.retrievers = {
            'abstract': TdifAnalysis.TdifAnalysis(self.data[self.data.abstract.notna()].abstract),
            'body_text': TdifAnalysis.TdifAnalysis(self.data[self.data.body_text.notna()].body_text)
        }
        self.question_dict = dict()

    def run(self, asked_question, answers_article_length):
        answers = ResearchAnswers.find_answers(self, asked_question, max_articles=answers_article_length)
        print(answers)


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
        question = SpeechToText.get_audio_transcription("Question/question.wav")
    else:
        question = str(input("Enter Question Text:"))
    article_length = 5
    main.run(question, 5)