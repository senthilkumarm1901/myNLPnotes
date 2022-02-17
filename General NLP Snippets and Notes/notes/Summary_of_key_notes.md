### A Summary of my own notes from `NLP Techniquest that power the World of Words.pdf`

1. Traditional NLP
- Common Traditional NLP Preprocessing Techniques:
    - Parts of Speech (NN, JJ, VBG, etc.,)
    - Dependency Parsing (nsubj, dobj)
    - Lemmatization, Stemming
    - TF-IDF: A graph of `Occurence` vs `Value` will mean TF-IDF will put rare words (with low occurence) at high value. 
          - TF(t,d) = Number of occurences of the term t in document d
          - IDF(t) = Inverse Document Frequency for the term t = log (1 + N/df(d,t)); N - total no. of docs; df(d,t) - no. of documents containing the term t


- One trend that have powered NLP's growth:
    - Better numerical representation of text through Transfer Learning models
<br>

- `Word2Vec`:
- An envolution over Feed Forward Neural Language Model but with no non-linear activation function
- A log linear model
    - CBOW: Predict focus word given a bag of words context
    - Skipgram: Predict context words given a focus word

<br>
- `Glove`:
- Generates word vectors by examining the **word occurrences** within text corpus

Evolution of DL model architectures in NLP: 
- Why RNN? Lack of memory element in MLP made it perform poorly in sequence modeling tasks;
- Why LSTM? Vanishing gradient problem in RNN when BPTT (back propagating through time)
- LSTM units in Sequence2Sequence Models: A combination of Encoder - Decoder architectures where 
     - the `Encoder` creates a "thought" or "context" vector of fixed size
- Why attention in Seq2Seq Models? 
     - A source sequence of any length was compressed to a fixed size vector. 
     - Attention solves the limitation by making the decoder look at the relevant source input hidden states
- Multiple layers of `self attention` forms the core of Transformer Architecture
     - Self Attention is used to look at the surrounding words in a sentence or document to obtain more contextually sensitive word representations


<br>
- What was BERT pre-trained on?
     - Masked Language Model
     - New Sentence Prediction  
