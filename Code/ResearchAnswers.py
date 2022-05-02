from collections import OrderedDict

import numpy as np
import torch
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'
MAX_LEN = 512


def summarize_answer(self, text):
    summarized_answer = self.summarizer(text)[0]
    return summarized_answer['summary_text']


def chunkify_inputs(inputs, token_type_ids, input_ids):
    # create question mask based on token_type_ids
    # value is 0 for question tokens, 1 for context tokens
    qt = torch.masked_select(input_ids, token_type_ids)
    chunk_size = MAX_LEN - qt.size()[0] - 1  # the "-1" accounts for
    # having to add an ending [SEP] token to the end
    # create a dict of dicts; each sub-dict mimics the structure of pre-chunked model input
    chunked_input = OrderedDict()
    for key, value in inputs.items():
        if device == "cpu":
            q = torch.masked_select(value, token_type_ids)
            c = torch.masked_select(value, ~token_type_ids)
        else:
            q = torch.masked_select(value.to(device), token_type_ids).to(device)
            c = torch.masked_select(value.to(device), ~token_type_ids).to(device)
        chunks = torch.split(c, chunk_size)
        for i, chunk in enumerate(chunks):
            if i not in chunked_input:
                chunked_input[i] = {}
            if device == "cpu":
                thing = torch.cat((q, chunk))
            else:
                thing = torch.cat((q, chunk)).to(device)
            if i != len(chunks) - 1:
                if key == 'input_ids':
                    if device == "cpu":
                        thing = torch.cat((thing, torch.tensor([102])))
                    else:
                        thing = torch.cat((thing, torch.tensor([102]).to(device))).to(device)
                else:
                    if device == "cpu":
                        thing = torch.cat((thing, torch.tensor([1])))
                    else:
                        thing = torch.cat((thing, torch.tensor([1]).to(device))).to(device)

            chunked_input[i][key] = torch.unsqueeze(thing, dim=0).to(device)
    return chunked_input


def search_answers_in_articles(self, question, max_articles):
    # get top N candidate articles based on similarity score
    top_candidates = self.data.loc[self.articles.retrieve(question).index]
    top_candidates = top_candidates.head(max_articles)
    tokenizer = self.tokenizer
    top_candidates["start_score"] = None
    top_candidates["end_score"] = None
    top_candidates["answer"] = None
    top_candidates["summarized_answer"] = None
    for index, data in top_candidates.iterrows():
        start_score = 0
        end_score = 0
        score_count = 1
        final_answer = ''
        context = str(data.body_text)
        inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt',
                                       truncation_strategy='only_second', pad_to_max_length=True)
        if device == "cpu":
            input_ids = inputs['input_ids']
            token_type_ids = inputs['token_type_ids']
        else:
            input_ids = inputs['input_ids'].to(device, dtype=torch.long)
            token_type_ids = inputs['token_type_ids'].to(device, dtype=torch.bool)
        if len(input_ids) > MAX_LEN:
            chunked_inputs = chunkify_inputs(inputs, token_type_ids, input_ids)
            for k, chunk in chunked_inputs.items():
                score_count += 1
                output = self.model(**chunk)
                if device == "cpu":
                    answer_start = np.argmax(output.start_logits.cpu().detach().numpy(), axis=1)[0]
                    answer_end = np.argmax(output.end_logits.cpu().detach().numpy(), axis=1)[0] + 1
                else:
                    answer_start = torch.argmax(output.start_logits, dim=1)
                    answer_end = (torch.argmax(output.end_logits, dim=1) + 1)
                answer = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
                if answer != '[CLS]':
                    final_answer += str(answer) + "  "
                start_score += np.argmax(output.start_logits.cpu().detach().numpy(), axis=1)[0]
                end_score += np.argmax(output.end_logits.cpu().detach().numpy(), axis=1)[0]

            start_score = start_score / score_count
            end_score = start_score / score_count

        else:
            output = self.model(input_ids)
            # Get the most likely beginning of each answer with the argmax of the score
            if device == "cpu":
                answer_start = np.argmax(output.start_logits.cpu().detach().numpy(), axis=1)[0]
                answer_end = np.argmax(output.end_logits.cpu().detach().numpy(), axis=1)[0]
            else:
                answer_start = torch.argmax(output.start_logits, dim=1)
                answer_end = torch.argmax(output.end_logits, dim=1) + 1
            start_score = np.argmax(output.start_logits.cpu().detach().numpy(), axis=1)[0]
            end_score = np.argmax(output.end_logits.cpu().detach().numpy(), axis=1)[0]
            # Get the most likely end of each answer with the argmax of the score
            answer = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
            if answer != '[CLS]':
                final_answer += str(answer) + "  "
        top_candidates.loc[index, ['answer']] = final_answer
        top_candidates.loc[index, ['start_score']] = start_score
        top_candidates.loc[index, ['end_score']] = end_score
    top_candidates["summarized_answer"] = top_candidates["answer"].apply(lambda x: summarize_answer(self, x))
    answers = top_candidates[
        ['title', 'source_x', 'authors', 'url', 'publish_time', 'doi', 'start_score', 'end_score',
         'summarized_answer']].copy()
    answers = answers.sort_values(['start_score', 'end_score'], ascending=False)
    answers.to_csv("Data/answer.csv")
    return answers
