import torch

from transformers import BartTokenizer, AutoModelForSeq2SeqLM
tokenizer_2 = BartTokenizer.from_pretrained("Salesforce/bart-large-xsum-samsum")
model_2 = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/bart-large-xsum-samsum")

def inputOutput(input_text):

    # tokenize without truncation
    inputs_no_trunc = tokenizer_2(input_text, max_length=None, return_tensors='pt', truncation=False)

    # get batches of tokens corresponding to the exact model_max_length
    chunk_start = 0
    chunk_end = tokenizer_2.model_max_length  # == 1024 for Bart
    inputs_batch_lst = []
    while chunk_start <= len(inputs_no_trunc['input_ids'][0]):
        inputs_batch = inputs_no_trunc['input_ids'][0][chunk_start:chunk_end]  # get batch of n tokens
        inputs_batch = torch.unsqueeze(inputs_batch, 0)
        inputs_batch_lst.append(inputs_batch)
        chunk_start += tokenizer_2.model_max_length  # == 1024 for Bart
        chunk_end += tokenizer_2.model_max_length  # == 1024 for Bart

    # generate a summary on each batch
    summary_ids_lst = [model_2.generate(inputs, num_beams=4, max_length=100, early_stopping=True) for inputs in
                       inputs_batch_lst]

    # decode the output and join into one string with one paragraph per summary batch
    summary_batch_lst = []
    for summary_id in summary_ids_lst:
        summary_batch = [tokenizer_2.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                         summary_id]
        summary_batch_lst.append(summary_batch[0])
    summary_all = '\n'.join(summary_batch_lst)

    return summary_all