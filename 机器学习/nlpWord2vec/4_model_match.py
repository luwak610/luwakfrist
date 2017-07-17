#!/usr/bin/env python
# -*- coding: utf-8  -*-
#测试训练好的模型

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')# 忽略警告
import sys  
#reload(sys)  
#sys.setdefaultencoding('utf8')
import gensim


if __name__ == '__main__':
    fdir = '/机器学习北理工/wiki_zh_word2vec-master/'
    model = gensim.models.Word2Vec.load(fdir + 'wiki.zh.text.model')

    word = model.most_similar(u"股票")
    print('~相关',len(word),'个')
    for t in word:
        print(t[0],t[1])
    print('~doesnt',model.doesnt_match(u'投资 回报 收益 亏损 股票'.split()))
    print('~2similarity',model.similarity(u'股票',u'亏损'))
    print('~2similarity',model.similarity(u'股票',u'收益'))
    '''
    word = model.most_similar(positive=[u'皇上',u'国王'],negative=[u'皇后'])
    for t in word:
        print t[0],t[1]


    print model.doesnt_match(u'太后 妃子 贵人 贵妃 才人'.split())
    print model.similarity(u'书籍',u'书本')
    print model.similarity(u'逛街',u'书本')
    '''


