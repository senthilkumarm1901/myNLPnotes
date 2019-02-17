# -*- coding: utf-8 -*-


import nltk
import re
import pprint
from nltk import Tree

from nltk.tokenize import RegexpTokenizer

import pandas as pd

from collections import Counter
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore")

from time import time
start_time = time()

#Give the location to the input file
ip=pd.read_csv()

#ip=ip[['Content']]

#For Noun Phrase Extraction
#custom_patterns_new = """
#    NP:{<JJ>*<NN|NNS|NNP|NNPS>+<JJ>*<CD>*}
#    {<NN|NNS|NNP|NNPS>+<JJ|CD|VB|CC>*<NN|NNS|NNP|NNPS>+}
#    """

#For Adjective Phrase Extraction
pattern=r"""
AP2:
   {<JJ.*><VB.*|DT>*<NN.*>+}
AP1:
   {<NN.*>+<VB.*|DT>*<JJ.*>}
"""
custom_NPChunker = nltk.RegexpParser(pattern)
Regexp_tokenizing=RegexpTokenizer("\w+")
list_NPs=[]
Overall_list=[]

#edit this function based on data
func = lambda s: s[0].lower() + s[1:] if s else ''

count1=0
for each in range(len(ip)):
    if each%100==0: print each
    try:
                text=ip.ix[each,'content'].decode('utf8','ignore').encode('utf8','ignore') #is this needed for Python3? Check it
                #text=re.sub(r'@','AT',text)
                #text=re.sub(r'#','HASHTAG',text)
    
                sent_tokenize_list=nltk.sent_tokenize(text)
                sent_tokenize_list_proper=[func(sent) for sent in sent_tokenize_list]
                RegEx_Tokenized_Words=[Regexp_tokenizing.tokenize(sentence) for sentence in sent_tokenize_list_proper]
                Pos_Tag_list=[nltk.pos_tag(word) for word in RegEx_Tokenized_Words]
                word_tree = [custom_NPChunker.parse(word) for word in Pos_Tag_list]
                nps=[] # an empty list in which to NPs will be stored.
                for every in word_tree:
                        tree = custom_NPChunker.parse(every)
                        for subtree in tree.subtrees():
                                if subtree.label() == 'AP2':
                                        t = subtree
                                        t = ' '.join(word for word, tag in t.leaves())
                                        nps.append(t)
                list_NPs.append(nps)
                Overall_list.extend(nps)
    except:
                count1+=1
                list_NPs.append("ignored document")
                pass

print str(count1)+" documents ignored"
NPs_dict=Counter(Overall_list)
NPs_dict_sorted=OrderedDict(sorted(NPs_dict.items(),key=lambda x:x[1], reverse=True))

ip['NounPhrases']=list_NPs
ip.to_csv() #Creates an additional column to the data containing Noun or Adjective Phrases
op=pd.Series(NPs_dict_sorted)
op1=pd.DataFrame(op)
op1['NounPhrases']=op1.index
op1['Count']=op1[0]
op1.index=range(len(op1))
op1=op1[['NounPhrases','Count']]
#op1.to_csv() #Only if you need the majority of the Nouns used

end_time = time()
print "Running Time : " + str((end_time - start_time)/60*1.0) + " minutes"

#good links:
#http://nbviewer.jupyter.org/github/lukewrites/NP_chunking_with_nltk/blob/master/NP_chunking_with_the_NLTK.ipynb
#http://www.nltk.org/book/ch07.html
#https://github.com/skipgram/modern-nlp-in-python/blob/master/executable/Modern_NLP_in_Python.ipynb
