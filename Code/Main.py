import warnings
from sys import platform

import pandas as pd
import torch

from NLPTermProject import ResearchAnalysis, SpeechToText
from NLPTermProject.TdifAnalysis import TdifAnalysis

if platform == "darwin":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')


class Main(object):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
        self.model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-base-uncased')
        self.retrievers = {
            'abstract': TdifAnalysis(self.data[self.data.abstract.notna()].abstract),
            'body_text': TdifAnalysis(self.data[self.data.body_text.notna()].body_text)
        }
        self.main_question_dict = dict()
        self.model.save_pretrained('Output')

    def run(self, question, article_length):
        answers = ResearchAnalysis.retrieve_answers(self, question, max_articles=article_length)
        answers.to_csv("answer.csv")
        print(answers)


data_url = 'Data/covid19.csv'
data = pd.read_csv(data_url)
main = Main(data_url)
if __name__ == '__main__':
    #QuestionRecorder.record_voice()
    question = SpeechToText.get_audio_transcription("Question/question.wav")
    article_length = 5
    main.run(question, 5)
