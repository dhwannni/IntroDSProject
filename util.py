import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from os import path


NON_ALNUM_RE = re.compile(r'[^a-z0-9\s]+') # keeps the spaces 
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "]+",
    flags = re.UNICODE
)



def preprocess_text(
    text: str,
    remove_stopwords: bool = False, # Ai usually uses more complete sentences while humans speak in fragments, slang
    remove_emojis: bool =  False,
    remove_punctuation: bool = False     #Humans sometimes  use "!!!" or "?!?". AI usually sounds more cleaner 
    ) -> list[str]:
  
  
  
    """
    - Convert text to a list of tokens.
    - Lowercase the text
    - Remove emojis if needed
    - Remove punctuation/non-alphanumeric characters
    - Split into tokens (words)
    """    
    # 1)  lowercase 
    text = text.lower()

    #2) remove emojis 

    if remove_emojis: 
        text = EMOJI_RE.sub('', text)
     
    #3)replace non alphanumeric with spaces 
    if remove_punctuation:
        text = NON_ALNUM_RE.sub(" ", text)

    
    
    # Split text into tokens
    tokens = text.split()
    

    #Removing stopwords 

    if remove_stopwords: 
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    
    
    return tokens

###Loading AI comments 


def load_ai_comments (file_path):
    df = pd.read_csv(file_path)

    comments  = df["comment"].astype(str).tolist()
    labels = ["ai"] * len(comments)

    #processed = [preprocess_text(c) for c in comments]

    return comments, labels



def load_human_comments(file_path):
    df = pd.read_csv(file_path)

    comments = df["CommentText"].astype(str).tolist()
    labels = ["human"] * len(comments)

    return comments, labels


#combining commets from the data frames 
