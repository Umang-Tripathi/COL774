import numpy as np

class NaiveBayes:
    def __init__(self):

        self.vocabulary = dict()
        self.class_prob = []
        self.word_prob = []
        self.smoothening = 0.0
        self.class_count = []
        self.word_count = []
        self.total_word_count = 0
        self.total_class_count = 0
       

        pass
        
    def token_id(self, token):

        # Check if the word is in the vocabulary
        if token in self.vocabulary.keys():
            return self.vocabulary[token]
        else:

            #assign the unknown token to the last index
            self.vocabulary[token] = len(self.vocabulary)
            return len(self.vocabulary)-1
        
        return -1

    def tokenizer(self,text):

        # Tokenize the text
        tokens = text.split()

        # Remove punctuation
        tokens = [word.strip('.,') for word in tokens]

        # Remove special characters
        tokens = [word for word in tokens if word.isalnum()]

        # Lowercase all words
        tokens = [word.lower() for word in tokens]

        # convert to token ids
        tokens = [self.token_id(word) for word in tokens]

        return tokens

    
    def fit(self, df, smoothening, class_col = "Class Index", text_col = "Tokenized Description"):
        """Learn the parameters of the model from the training data.
        Classes are 1-indexed

        Args:
            df (pd.DataFrame): The training data containing columns class_col and text_col.
                each entry of text_col is a list of tokens.
            smoothening (float): The Laplace smoothening parameter.
        """


        
        pass
    
    def predict(self, df, text_col = "Tokenized Description", predicted_col = "Predicted"):
        """
        Predict the class of the input data by filling up column predicted_col in the input dataframe.

        Args:
            df (pd.DataFrame): The testing data containing column text_col.
                each entry of text_col is a list of tokens.
        """
        pass