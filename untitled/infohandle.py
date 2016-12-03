# -*- coding: utf-8 -*-

from django.http import HttpResponse
from django.shortcuts import render

import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# import the LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from django.core.files import File
from time import clock


# # 表单
# def search_form(request):
#     return render_to_response('search_form.html')
from untitled.settings import BASE_DIR


class Infohandle:
    # 接收请求数据
    def handle(self, request):
        request.encoding = 'utf-8'
        if 'q' in request.GET:
            message = request.GET['q'].encode('utf-8')
        else:
            message = ''
        context = {}
        res = '结果：垃圾短信'
        context['hello'] = res
        # template = loader.get_template('polls/templates/msg.html')
        return render(request, 'msg.html', context)


# def handle(request):
#     info = Infohandle()
#     info.handle(request)
def cutdata(dataMat):
    return [' '.join(jieba.cut(line)) for line in dataMat]

def loadtrain(file_name, maxline=10):
    dataMat = []
    labelMat = []
    data = open(file_name).readlines()
    if maxline < len(data):
        data = data[:maxline]
    for line in data:
        lineArr = line.strip().split('\t')
        labelMat.append(lineArr[0])
        dataMat.append(lineArr[1])
    return labelMat, dataMat

# 加载测试数据
def loadtest(file_name, maxline=10):
    data = open(file_name).readlines()
    if maxline < len(data):
        data = data[:maxline]
    return data

def handle(request):
    request.encoding = 'utf-8'
    # f = open('/hello', 'w')
    # f.write('fe')
    # f.close()
    context = {}
    if 'q' in request.GET:
        message = request.GET['q'].encode('utf-8')
        # 加载模型

        # lr = joblib.load(BASE_DIR+'/static/datamodel/lr_model.m')
        # 载入训练集
        labelmat, datamat = loadtrain(BASE_DIR+'/static/datamodel/train.txt', maxline=20000)

        # 载入测试集
        # testdataMat = loadtest('test.txt', maxline=100)

        # 使用jieba库进行中文分词
        datamat = cutdata(datamat)
        # testdataMat = cutdata(testdataMat)

        # 将字符串标签转换为整型
        labelmat = [0 if int(x) == 0 else 1 for x in labelmat]

        word_count_vect = CountVectorizer()
        X_train_counts = word_count_vect.fit_transform(datamat)

        # 训练数据计算tf-idf矩阵，并训练出Logistc_regression模型
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        # 训练出回归系数
        # start = clock()
        clf = LogisticRegression()
        clf.fit(X_train_tfidf, labelmat)
        # joblib.dump(clf, "lr_model.m")
        # finsh = clock()
        # print (finsh - start)

        test_data = message
        testdata_mat = ' '.join(jieba.cut(test_data))
        # 载入测试集
        testdataMat_raw = loadtest(BASE_DIR+'/static/datamodel/test.txt', maxline=1)
        testdataMat_raw[0] = testdata_mat
        # print testdataMat_raw[0]
        # print testdata_mat
        testdataMat_raw = cutdata(testdataMat_raw)
        # 计算测试数据tfidf
        # word_count_vect = CountVectorizer()
        X_test_counts = word_count_vect.transform(testdataMat_raw)
        # 训练数据计算tf-idf矩阵，并训练出Logistc_regression模型
        # tfidf_transformer = TfidfTransformer()
        X_test_tfidf = tfidf_transformer.transform(X_test_counts)
        # 判别新数据
        predicted = clf.predict(X_test_tfidf)
        print predicted
        res = '结果：正常短信'
        if predicted[0] == 1:
            res = '结果：垃圾短信'
        context['hello'] = res
        context['pre_data'] = message
    else:
        context['hello'] = '请输入短信内容'
        context['pre_data']='空'


    # template = loader.get_template('polls/templates/msg.html')
    return render(request, 'msg.html', context)
