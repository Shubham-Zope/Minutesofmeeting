
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
import re
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import JSONRenderer, TemplateHTMLRenderer
import json
from transformers import BartTokenizer, AutoModelForSeq2SeqLM
import PyPDF2

def handle():
    return redirect('/summarize/')

class SummarizeData(APIView):
    authentication_classes = []
    permission_classes = []

    def get(self, request, format=None):
        tokenizer_2 = BartTokenizer.from_pretrained("knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM-AMI")
        model_2 = AutoModelForSeq2SeqLM.from_pretrained("knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM-AMI")
        # summarizer = pipeline("summarization", model="Salesforce/bart-large-xsum-samsum")
        Transcript = ''
        pdfFileObj = open('main/testingtranscript.pdf', 'rb') 
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        for i in range(int(pdfReader.numPages)):
            pageObj = pdfReader.getPage(i)
            Transcript += pageObj.extractText()
        with open("sample.txt",'w') as file:
            file.write(Transcript)
        pdfFileObj.close()
        txt = re.sub(r'[^.\w\s]', '', "".join(Transcript))
        inputs_no_trunc = tokenizer_2(txt, max_length=None, return_tensors='pt', truncation=False)
        # get batches of tokens corresponding to the exact model_max_length
        chunk_start = 0
        chunk_end = tokenizer_2.model_max_length if (len(txt.split()) > tokenizer_2.model_max_length) else int(len(txt.split())/3) # == tokenizer_2.model_max_length for Bart
        inputs_batch_lst = []
        while chunk_start <= len(inputs_no_trunc['input_ids'][0]):
            inputs_batch = inputs_no_trunc['input_ids'][0][chunk_start:chunk_end]  # get batch of n tokens
            inputs_batch = torch.unsqueeze(inputs_batch, 0)
            inputs_batch_lst.append(inputs_batch)
            chunk_start += tokenizer_2.model_max_length if (len(txt.split()) > tokenizer_2.model_max_length) else int(len(txt.split())/3)  # == tokenizer_2.model_max_length for Bart
            chunk_end += tokenizer_2.model_max_length if (len(txt.split()) > tokenizer_2.model_max_length) else int(len(txt.split())/3)  # == tokenizer_2.model_max_length for Bart

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
        # OUTPUT = summarizer(summary_all)
        print(summary_all)

        return render(request, "index.html", {"summary": summary_all}) 


temp = ""
attendes = [] 
meetcode = ""
date = ""
attendesstr = ""
@api_view(('POST','GET',))
def MOM(request):
    if request.method == 'POST':
        # print(request.body)
        txt = json.loads(request.body) 
        global temp
        global attendesstr
        global attendes
        global meetcode
        global date 
        temp = ""
        attendes = [] 
        meetcode = ""
        date = ""
        attendesstr = ""
        temp = txt['text'] 
        meetcode = txt['code']
        date = txt['date']
        t = temp.split('\n')
        for i in t:
            at = i.split(':')
            print(at[0])
            attendes.append(at[0])
        attendes = list(dict.fromkeys(attendes)) 
        attendes.remove(attendes[len(attendes) - 1])
        attendesstr = ', '.join(attendes)
        # print(txt['text'])
        # print(temp)
        
        return redirect('/mom/')
        

    elif request.method == 'GET':
        print("MOM first step done")
        return render(request, 'index.html', {'trans': temp,'attendees':attendesstr , 'code': meetcode, 'date': date}) 

    
finaltrans = ''
finalsummary = ''
finalcode = ''
finaldate = ''
finalattendes = ''
@api_view(('POST','GET',))
def summarize(request):
    if request.method == 'POST':
        # print(request.body)
        txt = json.loads(request.body) 
        global finaltrans
        global finalcode 
        global finaldate
        global finalattendes
        finaltrans = txt['transcript'] 
        finalcode = txt['code']
        finaldate = txt['date']
        finalattendes = txt['attendes']
        # print(txt['transcript']) 
        tokenizer_2 = BartTokenizer.from_pretrained("knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM-AMI")
        model_2 = AutoModelForSeq2SeqLM.from_pretrained("knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM-AMI")
        # summarizer = pipeline("summarization", model="Salesforce/bart-large-xsum-samsum")
        txt = re.sub(r'[^.\w\s]', '', "".join(finaltrans))
        inputs_no_trunc = tokenizer_2(finaltrans, max_length=None, return_tensors='pt', truncation=False)
        # get batches of tokens corresponding to the exact model_max_length
        chunk_start = 0
        chunk_end = tokenizer_2.model_max_length #if (len(finaltrans.split()) > tokenizer_2.model_max_length) else int(len(finaltrans.split())/3) # == tokenizer_2.model_max_length for Bart
        inputs_batch_lst = []
        while chunk_start <= len(inputs_no_trunc['input_ids'][0]):
            inputs_batch = inputs_no_trunc['input_ids'][0][chunk_start:chunk_end]  # get batch of n tokens
            inputs_batch = torch.unsqueeze(inputs_batch, 0)
            inputs_batch_lst.append(inputs_batch)
            chunk_start += tokenizer_2.model_max_length #if (len(finaltrans.split()) > tokenizer_2.model_max_length) else int(len(finaltrans.split())/3)  # == tokenizer_2.model_max_length for Bart
            chunk_end += tokenizer_2.model_max_length #if (len(finaltrans.split()) > tokenizer_2.model_max_length) else int(len(finaltrans.split())/3)  # == tokenizer_2.model_max_length for Bart

        # generate a summary on each batch
        summary_ids_lst = [model_2.generate(inputs, num_beams=4, max_length=100, early_stopping=True) for inputs in
                           inputs_batch_lst]

        # decode the output and join into one string with one paragraph per summary batch
        global finalsummary
        summary_batch_lst = []
        for summary_id in summary_ids_lst:
            summary_batch = [tokenizer_2.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                             summary_id]
            summary_batch_lst.append(summary_batch[0])
        finalsummary = '\n'.join(summary_batch_lst)
        # OUTPUT = summarizer(summary_all)
        # print(f"Final summary in post is {finalsummary}")
        #print("done")
        return Response('hello')
       
    elif request.method == 'GET':
        print(f"Final trans is {finaltrans}")
        print(f"Final summary is {finalsummary}")
        print(f"Final code is {finalcode}")
        print(f"Final date is {finaldate}")
        print(f"Final attendes is {finalattendes}")
        return render(request, 'index.html', {'trans': finaltrans, 'summary': finalsummary, "date": finaldate, "code": finalcode, "attendees": finalattendes}) 


