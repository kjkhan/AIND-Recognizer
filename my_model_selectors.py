import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

NO_SCORE = float('-inf')   # TODO: change to np.nan?

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def score(self, n_component):
        """ calculate BIC score given n_component

        :return: float, BIC score
        """
        # run model
        hmm_model = self.base_model(n_component)
        if not hmm_model:
            return NO_SCORE

        # calculate log scores
        logN = np.log(len(self.X))  # number of data points
        try:
            logL = hmm_model.score(self.X, self.lengths)
        except:
            if self.verbose:
                print("score failure on {} with {} length".format(self.this_word, len(self.lengths)))
            return NO_SCORE

        # calculate number of parameters
        # parameters = probabilities in transition matrix + Gaussian mean + Gaussian variance
        # (thanks yc lu and D. Sheahen for office hours discussion on this)
        n = n_component                     # number of states
        f = len(self.X[0])                  # number of features (e.g., x, y, norm_x, etc.)
        p = (n*(n-1) + n-1) + n*f + n*f     # this can be simplified but is kept like this for readability

        # calculate BIC score (rearraged so higher score is better)
        bic_score = -2*logL + p*logN
        return bic_score

    def select(self):
        """ select the best model for self.this_word based on given
        scoring criteria for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # loop through components to get scores
        scores = [self.score(n) for n in range(self.min_n_components, self.max_n_components+1)]

        # return model with best (lowest) score
        best_num_components = np.argmin(scores) + self.min_n_components
        return self.base_model(best_num_components)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def score(self, n_component):

        # get score for this word (i.e., log P(X(i)))
        hmm_model = self.base_model(n_component)
        if not hmm_model:
            return NO_SCORE
        try:
            logL = hmm_model.score(self.X, self.lengths)
        except:
            return NO_SCORE

        # test other words using this model (i.e., log P(X(all but i)))
        logP_sum = 0
        for w in self.words.keys():
            if w != self.this_word:
                try:
                    X, lengths = self.hwords[w]
                    logP_sum += hmm_model.score(X, lengths)
                except:
                    if self.verbose:
                        print("score failure on {} against {} sequences with {} length".format(self.this_word, w, len(self.lengths)))

        # calculate DIC score
        M = len(self.words)                 # number of classes (words to be trained)
        dic_score = logL - logP_sum/(M-1)
        return dic_score

    def select(self):
        """ select the best model for self.this_word based on given
        scoring criteria for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # loop through components to get scores
        scores = [self.score(n) for n in range(self.min_n_components, self.max_n_components+1)]

        # return model with best (highest) score
        best_num_components = np.argmax(scores) + self.min_n_components
        return self.base_model(best_num_components)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''


    def score(self, n_component):
        """ calculate average logL score for n_component among k-fold subsets

        :return: float, average logL score for n_component
        """
        scores = []

        # initialize split method with 2 folds if there are only 2 samples, otherwise use default 3
        if len(self.sequences) < 2:
            return NO_SCORE
        split_method = KFold(n_splits=min(3, len(self.sequences)))

        # iterate through splits
        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

            self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
            test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)

            # run model on training sequences, move on to next fold if model was not created
            hmm_model = self.base_model(n_component)
            if not hmm_model:
                continue

            # get the test score for this fold
            try:
                hmm_score = hmm_model.score(test_X, test_lengths)
                scores.append(hmm_score)
            except:
                if self.verbose:
                    print("score failure on {} with {} length".format(self.this_word, len(test_lengths)))

        if scores:
            return np.mean(scores)
        else:
            return NO_SCORE

    def select(self):
        """ select the best model for self.this_word based on given
        scoring criteria for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # loop through components to get scores
        scores = [self.score(n) for n in range(self.min_n_components, self.max_n_components+1)]

        # return model with best (highest) score
        best_num_components = np.argmax(scores) + self.min_n_components
        return self.base_model(best_num_components)