# -*- coding: utf-8 -*-
from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader

def index(request):
    # return HttpResponse("Hello, world. You're at the polls index.")
    context          = {}
    context['hello'] = '短信识别结果：'
    # template = loader.get_template('polls/templates/msg.html')
    return render(request, 'msg.html', context)

