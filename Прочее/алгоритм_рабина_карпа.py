# -*- coding: utf-8 -*-
"""Алгоритм Рабина Карпа.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ixALJ3s4_8nxBllW8srwIbfuMsImoNaB
"""

import random 
def hash_func(S,x,p):
    sum = 0
    pol = 1
    for i in range(len(S)):
        sum += (ord(S[i]) * pol) % p
        pol*=x%p
    return sum%p

def Rabin_Karp(substring,string):
    Index = []
    p = 2**31 - 1
    x = random.randint(1,p-1)
    hash_substring = hash_func(substring,x,p)
    hash_sub_str = hash_func(string[len(string)-len(substring):len(string)],x,p)
    for i in range(len(string)-1,len(substring)-2,-1):
        part_string = string[i-len(substring)+1:i+1]
        if hash_sub_str==hash_substring and substring==part_string:
            Index.append(i-len(substring)+1)
        hash_sub_str = ((hash_sub_str - ord(string[i])*x**(len(substring)-1))*x + ord(string[i-len(substring)]))%p
    return Index