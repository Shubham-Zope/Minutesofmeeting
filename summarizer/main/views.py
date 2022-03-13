import django
from django.shortcuts import render, redirect
from django.views.generic import View
from django.http import FileResponse
# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
import pandas as pd
import numpy as np
import json
import joblib
import torch

from transformers import BartTokenizer, AutoModelForSeq2SeqLM
import PyPDF2

class SummarizeData(APIView):
    authentication_classes = []
    permission_classes = []

    def get(self, request, format=None):
        tokenizer_2 = BartTokenizer.from_pretrained("Salesforce/bart-large-xsum-samsum")
        model_2 = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/bart-large-xsum-samsum")
        summarizer = pipeline("summarization", model="Salesforce/bart-large-xsum-samsum")
        txt = ''
        pdfFileObj = open('main/testingtranscript2.pdf', 'rb') 
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        for i in range(int(pdfReader.numPages)):
            pageObj = pdfReader.getPage(i)
            txt += pageObj.extractText()
        # print(txt)
        pdfFileObj.close()
        inputs_no_trunc = tokenizer_2(txt, max_length=None, return_tensors='pt', truncation=False)

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
        OUTPUT = summarizer(summary_all)
        print(OUTPUT)