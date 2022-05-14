#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2022/5/13 8:56
# @Author: Aaron Meng
# @File  : similarity.py

import sys

sys.path.append('..')
from text2vec import Similarity

# Two lists of sentences
sentences1 = []
sentences2 = []
with open('syn_sample.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        word1, word2, cata = line.split()
        sentences1.append(word1)

        sentences2.append(word2)

f = open('ans.txt','a+')
sim_model = Similarity()
for i in range(len(sentences1)):
    result = []
    for j in range(len(sentences2)):
        score = sim_model.get_score(sentences1[i], sentences2[j])
        result.append((sentences2[j], score))
    result.sort(key=lambda x: x[1],reverse=True)
    for j in range(3):
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], result[j][0], result[j][1]))
        f.write("{} \t\t {} \t\t Score: {:.4f}\n".format(sentences1[i], result[j][0], result[j][1]))

f.close()