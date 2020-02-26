# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:12:15 2020

@author: yadav
"""
from nltk.translate.bleu_score import corpus_bleu



def Bleu(references, hypotheses, n = 4):
    
    """if n = 1, will calculate Bleu-1 otherwise will calculate Blue-4
    by defualt.  
    
    
    """
    
    if n == 1:
        
        bleu = corpus_bleu(references, hypotheses, weights = (1.0/1.0, ))
    
    else:
        bleu = corpus_bleu(references, hypotheses)
        
        
    return bleu