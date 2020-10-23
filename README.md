

This repository is dedicated to improving my NLP expertise.
It consists of original papers, my learning materials, my codes, among other useful links. 
With the NLP space changing ever so rapidly, this is my humble attempt to at least be aware of most of it and get to the bottom of the most important ones.

## Courses to finish:

* DL Specialization (completed by me): https://www.coursera.org/specializations/deep-learning
* High difficulty (not pursuing yet): Best NLP Course ever! Mastering this would give the latest knowledge in NLP possible  
    - http://web.stanford.edu/class/cs224n/syllabus.html
    - https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z
* Moderate Difficulty (not pursuing yet): https://www.coursera.org/learn/language-processing
* Low Difficulty: Easier than Stanford (pursuing):
    - https://www.udemy.com/natural-language-processing-with-deep-learning-in-python/
    - https://www.udemy.com/deep-learning-advanced-nlp/

## Good Resources

A Good PyTorch Book (for further exploration at any stage of the below learning)
Book: https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf 
Code: https://github.com/deep-learning-with-pytorch/dlwpt-code 

NLP-PyTorch Source1 : Official PyTorch Tutorial for NLP
https://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html

NLP-PyTorch Source2: (Stanford 2017 cs 224n Course Codes; we have lectures of 2020 winter)
https://github.com/DSKSD/DeepNLP-models-Pytorch 
http://web.stanford.edu/class/cs224n/ (only for topics I need reference; resource is heavy)

NLP-PyTorch Source3: HuggingFace Transformers Tutorials
https://huggingface.co/transformers/quickstart.html


**Good Traditional NLP Learning Resources**

Machine Learning with Text in scikit-learn
- https://github.com/justmarkham/pycon-2016-tutorial/blob/master/tutorial_with_output.ipynb
- https://github.com/justmarkham/pycon-2016-tutorial
- https://www.youtube.com/watch?v=WHocRqT-KkU

Bhargav Srinivasa Desikan - Topic Modelling (and more) with NLP framework Gensim <br>
- https://github.com/bhargavvader/personal/tree/master/notebooks/text_analysis_tutorial
- https://www.youtube.com/watch?v=ZkAFJwi-G98


Word Embeddings Tutorials in Python Theory by Macro Bonzanini
- https://www.pycon.it/media/conference/slides/word-embeddings-for-natural-language-processing-in-python.pdf
- https://www.youtube.com/watch?v=c8QF1FJjPG8

Tony Ojeda, Benjamin Bengfort, Laura Lorenz - Natural Language Processing with NLTK and Gensim
- https://github.com/DistrictDataLabs/PyCon2016/tree/master/notebooks/tutorial
- https://www.youtube.com/watch?v=itKNpCPHq3I

 
Making an Impact with Python Natural Language Processing Tools - PyCon 2016
- https://www.youtube.com/watch?v=jSdkFSg9oW8&t=4347s  
- https://github.com/totalgood/twip﻿

- Active Learning:
    - https://www.datacamp.com/community/tutorials/active-learning
    - https://www.kdnuggets.com/2018/10/introduction-active-learning.html
    - Active Learning for CNN: https://openreview.net/pdf?id=H1aIuk-RW

**Some Neural NLP Links**

Text Classification using Tensorflow and Scikit-learn
- https://developers.google.com/machine-learning/guides/text-classification/
- https://github.com/google/eng-edu/blob/master/ml/guides/text_classification/explore_data.py

4-part Text Classification series using DNN techniques
- https://medium.com/@sabber/classifying-yelp-review-comments-using-lstm-and-word-embeddings-part-1-eb2275e4066b


NLP’s ImageNET moment/ BERT:
- http://ruder.io/nlp-imagenet/
- http://jalammar.github.io/illustrated-bert/
- https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270


About Word Embeddings:
- https://towardsdatascience.com/beyond-word-embeddings-part-2-word-vectors-nlp-modeling-from-bow-to-bert-4ebd4711d0ec

**Papers and Resources**

- Sebastian Ruder Blog http://ruder.io/
- http://nlp.fast.ai/
- Definition of all things NLP: https://github.com/sebastianruder/NLP-progress
- BERT as a service: https://github.com/hanxiao/bert-as-service
- NN - common training tips: https://karpathy.github.io/2019/04/25/recipe/
- SOTA Papers to understand:
  * Attention is all you need: 
    - https://arxiv.org/pdf/1706.03762.pdf
    - https://www.youtube.com/watch?v=54uLU7Nxyv8
    - https://www.youtube.com/watch?v=VEcsf0OKhfw
- BERT paper:
    - https://arxiv.org/pdf/1810.04805.pdf
    - https://www.youtube.com/watch?v=BaPM47hO8p8
ELMo:
    - https://arxiv.org/pdf/1802.05365.pdf
USE:
    - https://arxiv.org/pdf/1803.11175.pdf

Basics to Advanced concepts in NLP on how it is applied to Text Classification:
- https://arxiv.org/pdf/1904.08067.pdf

Cross-lingual Language Model Pretraining -- a paper
Another paper on Document-level representation: 
    - https://arxiv.org/pdf/1902.08850v2.pdf

- Deep Active Learning for NER: https://arxiv.org/pdf/1707.05928.pdf

XLNET:     
- XLNET: https://medium.com/keyreply/xlnet-a-new-pre-training-method-outperforming-bert-on-20-tasks-b34daeee8edb
 
NLP Kaggle Reading Group Series:
- https://www.youtube.com/watch?v=I82arEIPP6U&list=PLqFaTIg4myu8t5ycqvp7I07jTjol3RCl9&index=25

## Some NLP Projects I have executed at varied depths in my Professional Career:

| NLP Task | Description | Type of Task |
|-|-|-|
| Sequence   Classification | Given a sentence, I have classified it   based on pre-defined set of labels (e.g. Aspect-based Sentiment   classification) | Supervised |
| Named Entity Recognition | Given a sentence,  attribute   individual words to the entity classes they belong to (e.g.: Jason   <PERSON> bought Focus <NAMEPLATE> from Dearborn Ford   <DEALERNAME>); | Supervised |
| Intent Detection and Slots Filing | (part of a retrieval-based chatbot engine) Given a sentence , identify   the intents and slots (e.g.: -- (mostly as a query in a chatbot setting)   Where can I find CAD designs for steering? INTENT - CAD_design_info; SLOT -   Part_steering_wheel) | Supervised |
| Extractive Question Answering System | Given single or multiple PDFs/ docs/ markdowns, questions relevant to   that content can be answered by retrieving the correct span of text from the   uploaded PDFs. Adapted from this [open-source   repo](https://github.com/deepset-ai/haystack) | pre-trained (no-labeled data needed) OR pre-trained+fine-tuned (custom   dataset fine-tuned) |
| Semantic Text Retrieval Engine | Given a collection of unlabeled comments, can group them into   clusters/topics based on their similarity/co-occurrence; and identify the top   words that describe the cluster | Unsupervised |
| Clustering, Topic Modeling &   Summarization: | Given a collection of unlabeled comments, can group them into   clusters/topics based on their similarity/co-occurrence; and identify the top   words that describe the cluster. Summarize the information in the clusters   using Summarization module. Adapted from open-source [code1](https://radimrehurek.com/gensim/auto_examples/tutorials/run_summarization.html#sphx-glr-auto-examples-tutorials-run-summarization-py)   and   [code2](https://huggingface.co/transformers/task_summary.html#summarization) | Unsupervised |
