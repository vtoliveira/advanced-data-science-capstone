from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
from collections import Counter
import numpy as np

def preprocess_tweet(s, preserve_case=False, strip_handles=False, reduce_len=False, 
               punctuation = False, stop_words = False, join = True):
    
    punctuation = ['”“’'] if punctuation else list(string.punctuation+'”“’')
    stop = stopwords.words('english') + punctuation + ['rt', 'via'] if stop_words else punctuation + ['rt', 'via']
    tknzr = TweetTokenizer(preserve_case=preserve_case, 
                           strip_handles=strip_handles, reduce_len=reduce_len)
    
    tokens = tknzr.tokenize(s)
    
    tokens = [token for token in tokens
              if token not in stop and
              not token.startswith(('#', '@','http', '...'))] 
    
    if join:
        return ' '.join(tokens)
    
    return tokens

# function is credited to Udacity course Pytorch Challegende Nanodegree
def preprocess_text(text):
    positive_emoticons = [':-)', ':)', '=)', ':D', ';D', ':]', ';]', ': D']
    negative_emoticons = [':-(', ':(', '=(', ';(', 'D:', 'D;', ':[', ';[', ':/']

    # Replace punctuation with tokens so we can use them in our model
    for pos_emoticon in positive_emoticons:
        text = text.replace(pos_emoticon, '<POSITIVE_FACE>')
    for neg_emoticon in negative_emoticons:
        text = text.replace(neg_emoticon, '<NEGATIVE_FACE>')

    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()
    
    # Remove all words with  5 or fewer occurences
    #word_counts = Counter(words)
    #trimmed_words = [word for word in words if word_counts[word] > 5]

    return words

def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    ## implement function
    
    features = np.zeros((len(reviews_ints), seq_length))
    
    for i, review in enumerate(reviews_ints):
        if len(review)==seq_length:
            features[i, :] = review
        elif len(review) > seq_length:
            features[i, :] = review[:seq_length]
        else:
            n_zeros_to_add = (seq_length-len(review))
            array = np.zeros((seq_length))
            array[n_zeros_to_add:] = review
            features[i, :] = array
    return features

def tokenize_custom(review, vocab_to_int):
    new_review = []
    for word in review:
        try:
            new_review.append(vocab_to_int[word])
        except Exception as e:
            pass
    return new_review