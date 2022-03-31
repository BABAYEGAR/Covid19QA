import pandas as pd
import torch

from NLPTermProject.Answers import Answer


def retrieve_answers(self, question, section='abstract', keyword=None, max_articles=1000, batch_size=4):
    text_tokens = []
    indices = []
    section_path = section.split('/')
    if keyword:
        candidates = self.data[self.data[section_path[0]].str.contains(keyword, na=False, case=False)]
    else:
        # get top N candidate articles based on similarity score
        candidates = self.data.loc[self.retrievers[section_path[0]].retrieve(question, max_articles).index]
    if max_articles:
        candidates = candidates.head(max_articles)

    for idx, row in candidates.iterrows():
        if section_path[0] == 'body_text':
            text = get_body_section(self, row.body_text, section_path[1])
        else:
            text = row[section]
        if text and isinstance(text, str):
            text_tokens.append(text)
            indices.append(idx)

    num_batches = len(text_tokens) // batch_size
    all_answers = []
    for i in range(num_batches):
        batch = text_tokens[i * batch_size:(i + 1) * batch_size]
        answers = get_answers_from_text_tokens(self, question, batch)
        all_answers.extend(answers)

    last_batch = text_tokens[batch_size * num_batches:]
    if last_batch:
        all_answers.extend(get_answers_from_text_tokens(self, question, last_batch))

    columns = ['doi', 'authors', 'journal', 'publish_time', 'title']
    processed_answers = []
    for i, j in enumerate(all_answers):
        if j:
            row = candidates.loc[indices[i]]
            append_row = [j.text, j.start_score, j.end_score, j.input_text]
            append_row.extend(row[columns].values)
            processed_answers.append(append_row)
    answer_data = pd.DataFrame(processed_answers, columns=(['answer', 'start_score', 'end_score', 'context'] + columns))
    return answer_data.sort_values(['start_score', 'end_score'], ascending=False)


def get_body_section(body_text, section_name):
    sections = body_text.split('<SECTION>\n')
    for section in sections:
        lines = section.split('\n')
        if len(lines) > 1:
            if section_name.lower() in lines[0].lower():
                return section


def get_answers_from_text_tokens(self, question, text_list, max_tokens=512):
    tokenizer = self.tokenizer
    inputs = tokenizer.batch_encode_plus(
          [(question, text) for text in text_list], add_special_tokens=True, return_tensors='pt',
          max_length=max_tokens, truncation_strategy='only_second', pad_to_max_length=True)
    input_ids = inputs['input_ids']
    output = self.model(input_ids)
    answer_start_scores = output.start_logits
    answer_end_scores = output.end_logits
    # Get the most likely beginning of each answer with the argmax of the score
    answer_start = torch.argmax(answer_start_scores, dim=1).detach().numpy()
    # Get the most likely end of each answer with the argmax of the score
    answer_end = (torch.argmax(answer_end_scores, dim=1) + 1).detach().numpy()
    answers = []
    for i, text in enumerate(text_list):
        input_text = tokenizer.decode(input_ids[i, :], clean_up_tokenization_spaces=True).split('[SEP] ', 2)[1]
        answer = tokenizer.decode(input_ids[i, answer_start[i]:answer_end[i]], clean_up_tokenization_spaces=True)
        score_start = answer_start_scores.detach().numpy()[i][answer_start[i]]
        score_end = answer_end_scores.detach().numpy()[i][answer_end[i] - 1]
        if answer and not '[CLS]' in answer:
            answers.append(Answer(answer, score_start, score_end, input_text))
        else:
            answers.append(None)
    return answers