import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


NON_ALNUM_RE = re.compile(r'[^a-z0-9\s]+') # keeps the spaces 
EMOJI_RE = re.compile(r'[^\w\s,]')


def preprocess_text(
    text: str,
    remove_stopwords: bool = True, 
    remove_emojis: bool = True,
    remove_punctuation: bool = True
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