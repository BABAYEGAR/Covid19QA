import pandas as pd
import torch
from torch import cuda
from Code.Answers import Answer

device = 'cuda' if cuda.is_available() else 'cpu'


def summarize_answer(self, text):
    summarized_answer = self.summarizer(text, max_length=100, min_length=30, do_sample=False)[0]
    return summarized_answer['summary_text']


def search_answers_in_articles(self, question, max_articles):
    batch_size = 8
    body_text_listing = []
    indices = []
    # get top N candidate articles based on similarity score
    section = 'abstract'
    # get top N candidate articles based on similarity score
    top_candidates = self.data.loc[self.retrievers[section.split('/')[0]].retrieve(question, max_articles).index]
    top_candidates = top_candidates.head(max_articles)
    for index, row in top_candidates.iterrows():
        if section.split('/')[0] == 'body_text':
            text = row.body_text
        else:
            text = row[section]
        if text and isinstance(text, str):
            body_text_listing.append(text)
            indices.append(index)
    batch_num = len(body_text_listing) // batch_size
    retrieved_answers = []
    for i in range(batch_num):
        batch = body_text_listing[i * batch_size:(i + 1) * batch_size]
        answers = get_answers_from_transformer(self, question, batch)
        retrieved_answers.extend(answers)
    last_batch = body_text_listing[batch_size * batch_num:]
    if last_batch:
        retrieved_answers.extend(get_answers_from_transformer(self, question, last_batch))
    columns = ['doi', 'authors', 'journal', 'publish_time', 'title']
    processed_answers = []
    for i, j in enumerate(retrieved_answers):
        if j:
            row = top_candidates.loc[indices[i]]
            append_row = [j.text, j.start_score, j.end_score, j.input_text]
            append_row.extend(row[columns].values)
            processed_answers.append(append_row)
    answers_datatable = pd.DataFrame(processed_answers,
                                     columns=(['answer', 'start_score', 'end_score', 'context'] + columns)).sort_values(
        ['start_score', 'end_score'], ascending=False)
    answers_datatable["summarized_answer"] = answers_datatable["answer"].apply(lambda x: summarize_answer(self, x))
    answers_datatable.to_csv("Data/answer.csv")
    return answers_datatable


def get_answers_from_transformer(self, question, text_list):
    tokenizer = self.tokenizer
    inputs = tokenizer.batch_encode_plus([(question, text) for text in text_list], add_special_tokens=True, return_tensors='pt', truncation_strategy='only_second', pad_to_max_length=True)
    if device == "cpu":
        input_ids = inputs['input_ids']
    else:
        input_ids = inputs['input_ids'].to(device, dtype=torch.long)
    output = self.model(input_ids)
    # Get the most likely beginning of each answer with the argmax of the score
    if device == "cpu":
        answer_start = torch.argmax(output.start_logits, dim=1).detach().numpy()
    else:
        answer_start = torch.argmax(output.start_logits, dim=1)
    # Get the most likely end of each answer with the argmax of the score
    if device == "cpu":
        answer_end = (torch.argmax(output.end_logits, dim=1) + 1).detach().numpy()
    else:
        answer_end = (torch.argmax(output.end_logits, dim=1) + 1)
    answers = []
    for i, text in enumerate(text_list):
        input_text = tokenizer.decode(input_ids[i, :], clean_up_tokenization_spaces=True)
        input_text = input_text.split('[SEP] ', 2)[1]
        answer = tokenizer.decode(input_ids[i, answer_start[i]:answer_end[i]], clean_up_tokenization_spaces=True)
        if device == "cpu":
            score_start = output.start_logits.detach().numpy()[i][answer_start[i]]
            score_end = output.end_logits.detach().numpy()[i][answer_end[i] - 1]
        else:
            score_start = output.start_logits[i][answer_start[i]]
            score_end = output.end_logits[i][answer_end[i] - 1]
        if answer and not '[CLS]' in answer:
            answers.append(Answer(answer, score_start, score_end, input_text))
        else:
            answers.append(None)
    return answers

