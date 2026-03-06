

'''
Implementing Tokenization 
Tokenizers enable machines to process and analyze human language
Highly important for LLMs

'''

# ---- Installation 

'''
!pip install nltk
!pip install transformers==4.42.1
!pip install sentencepiece
!pip install spacy
!python -m spacy download en_core_web_sm
!python -m spacy download de_core_news_sm
!pip install scikit-learn
!pip install torch==2.2.2
!pip install torchtext==0.17.2
!pip install numpy==1.26.0
'''


# imports: 
import nltk
nltk.download("punkt")
nltk.download('punkt_tab')
import spacy
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.util import ngrams
from transformers import BertTokenizer
from transformers import XLNetTokenizer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from datetime import datetime
from datetime import datetime


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# ---- WORD-BASED TOKENIZER 
text = "This is a sample sentence for word tokenization."
tokens = word_tokenize(text)
print(tokens)

# This showcases word_tokenize from nltk library
text = "I couldn't help the dog. Can't you do it? Don't be afraid if you are."
tokens = word_tokenize(text)
print(tokens)

# This showcases the use of the 'spaCy' tokenizer with torchtext's get_tokenizer function
text = "I couldn't help the dog. Can't you do it? Don't be afraid if you are."
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# Making a list of the tokens and priting the list
token_list = [token.text for token in doc]
print("Tokens:", token_list)

# Showing token details
for token in doc:
    print(token.text, token.pos_, token.dep_)

'''
A thing to note: the problem with this algorithm is that words with similar meanings 
will be assigned different IDs, resulting them being treated as entirely separate words
with distinct meanings.  For example, Unicorns vs. Unicorn 
'''

# ---- CHARACTER-BASED TOKENIZER 

'''
Involves splitting text into individual characters
has its limitations - single characters may not convey same info 
as entire words - token length increases in size - causes issues with model size
loss of performance 

'''

# ---- SUBWORD-BASED TOKENIZER 

# using BertTokenizer -> 
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.tokenize("IBM taught me tokenization.")

# using sentencepiece/unigram 
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
tokenizer.tokenize("IBM taught me tokenization.")

# tokenization with PyTorch 
dataset = [
    (1,"Introduction to NLP"),
    (2,"Basics of PyTorch"),
    (1,"NLP Techniques for Text Classification"),
    (3,"Named Entity Recognition with PyTorch"),
    (3,"Sentiment Analysis using PyTorch"),
    (3,"Machine Translation with PyTorch"),
    (1," NLP Named Entity,Sentiment Analysis,Machine Translation "),
    (1," Machine Translation with NLP "),
    (1," Named Entity vs Sentiment Analysis  NLP ")]


tokenizer = get_tokenizer("basic_english")
tokenizer(dataset[0][1])

# token indices: 
def yield_tokens(data_iter):
    for  _,text in data_iter:
        yield tokenizer(text)

my_iterator = yield_tokens(dataset) 
next(my_iterator)

# ---- DEMO: Comparative text tokenization and performance analysis 

# nltk: 
startTime = datetime.now()
nltk_tokens = word_tokenize(text)
nltk_time = datetime.now() - startTime
print(nltk_time)

# spaCy
startTime = datetime.now()
nlp = spacy.load("en_core_web_sm")
# Making a list of the tokens and priting the list
spaCy_tokens = [token.text for token in nlp(text)]
spaCy_time = datetime.now() - startTime

#Bert
startTime = datetime.now()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_tokens = tokenizer.tokenize(text)
bert_time = datetime.now() - startTime

# XLNet
startTime = datetime.now()
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
xlnet_tokens = tokenizer.tokenize(text)
xlnet_time = datetime.now() - startTime

# Display tokens, time taken for each tokenizer, and token frequencies
print(f"NLTK Tokens: {nltk_tokens}\nTime Taken: {nltk_time} seconds\n")
show_frequencies(nltk_tokens, "NLTK")

print(f"SpaCy Tokens: {spaCy_tokens}\nTime Taken: {spaCy_time} seconds\n")
show_frequencies(spaCy_tokens, "SpaCy")

print(f"Bert Tokens: {bert_tokens}\nTime Taken: {bert_time} seconds\n")
show_frequencies(bert_tokens, "Bert")

print(f"XLNet Tokens: {xlnet_tokens}\nTime Taken: {xlnet_time} seconds\n")
show_frequencies(xlnet_tokens, "XLNet")

'''
Results: 
NLTK performed the best, while SpaCy performed the worst ( slowest ).
There are tradeoffs to consider such as accuracy, linguistic features,
 and integration when it comes to choosing which library.  
'''
