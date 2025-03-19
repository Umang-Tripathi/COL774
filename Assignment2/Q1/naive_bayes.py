import numpy as np
import pandas as pd
from collections import defaultdict

class NaiveBayes:
    def __init__(self):

        self.phi = {}  # log(P(y=k)) = log (phi_k)
        self.theta = defaultdict(lambda: defaultdict(float))  #  class -> word -> log(prob)
        self.vocab = set() # set of all unique words
        self.class_counts = defaultdict(int) # count of each class ; ( sigma 1{y(i)=k} )
        self.total_words_per_class = defaultdict(int) # total number of words in each class { sigma 1{y(i)=k} * |x(i)| }
        self.alpha = 1.0  # Default Laplace smoothing parameter

    def fit(self, df, smoothening, class_col="Class Index", text_col="Tokenized Description"):
        """
        Learn the parameters of the model from the training data.
        
        Args:
            df (pd.DataFrame): The training data containing class_col and text_col.
            smoothening (float): The Laplace smoothening parameter.
        """

        self.alpha = smoothening # alpha = smoothening parameter
        m = len(df) # m = total number of samples
        class_word_counts = defaultdict(lambda: defaultdict(int))  # word counts per class : class -> word -> count
        
        # class priors P(y=k) = phi_k
        for _, row in df.iterrows():

            label = row[class_col] # class label
            words = row[text_col] # tokenized text
            self.class_counts[label] += 1 # count of each class ; ( sigma 1{y(i)=k} )

            for word in words:

                class_word_counts[label][word] += 1 # count of each word in each class ; ( sigma(i=1 to m) sigma(j=1 to |x(i)| ) 1{y(i)=k} * 1{x(i)(j)=w} )
                self.vocab.add(word) # set of all unique words
                self.total_words_per_class[label] += 1 # total number of words in each class { sigma 1{y(i)=k} * |x(i)| }


        self.phi = {cls: np.log(count / m) for cls, count in self.class_counts.items()} #log(phi_k) = log( count(y=k) / m )
        

        # log P(x(i)(j)=w_l|y(i)=k)= log(theta_l(k)) with Laplace smoothing
        vocab_size = len(self.vocab)

        for cls in self.class_counts: # for each class

            total_words = self.total_words_per_class[cls] # total number of words in each class { sigma 1{y(i)=k} * |x(i)| }

            for word in self.vocab: # for each word in vocab

                word_count = class_word_counts[cls][word] # count of each word in each class ; ( sigma(i=1 to m) sigma(j=1 to |x(i)| ) 1{y(i)=k} * 1{x(i)(j)=w} )
                self.theta[cls][word] = np.log((word_count + self.alpha) / (total_words + vocab_size * self.alpha))
                # log(theta_l(k)) = log( count(w|y=k) + 1 * alpha / total_words(y=k) + alpha * |V| )

    def predict(self, df, text_col="Tokenized Description", predicted_col="Predicted"):
        """
        Predict the class of the input data by filling up column predicted_col in the input dataframe.

        Args:
            df (pd.DataFrame): The testing data containing text_col with tokenized text.
        """
        predictions = []

        for _, row in df.iterrows():

            words = row[text_col]

            class_scores = {cls: self.phi[cls] for cls in self.class_counts} # log(P(y=k)|x) = ( sum(j) log(p(x(j)|y=k)) ) +  log(P(y=k))

            for cls in class_scores:

                for word in words:
                    if word in self.vocab:

                        class_scores[cls] += self.theta[cls][word]  
                        # log(P(x| y)) = log(P( x(1)| y)) + log(P(x(2)| y)) + ... + log(theta_l(k))

                    else :

                        self.theta[cls][word] = np.log(self.alpha / (self.total_words_per_class[cls] + len(self.vocab) * self.alpha))

                        # log(P(x| y)) = log(P( x(1)| y)) + log(P(x(2)| y)) + ... + log(alpha / (total_words(y) + |V| * alpha))
                        
            predictions.append(max(class_scores, key=class_scores.get))  # prediction = argmax_k P(y=k | x)

        df[predicted_col] = predictions
