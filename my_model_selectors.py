import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


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
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
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
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_n = 0
        lowest_bic = float('inf')
        for n in range(self.min_n_components, self.max_n_components + 1):
            # Calculate logL
            model = self.base_model(n)
            # Use try/except to catch ValueError: rows of transmat_ must sum to 1.0
            # https://discussions.udacity.com/t/hmmlearn-valueerror-rows-of-transmat--must-sum-to-1-0/229995/7
            try:
                logL = model.score(self.X, self.lengths)
            except (ValueError, AttributeError):
                continue

            # Calculate p
            initial_state_params = n - 1
            transition_params = n * (n - 1)
            emission_params = 2 * n * len(model.means_)
            p = initial_state_params + transition_params + emission_params

            # Calculate BIC
            bic = -2 * logL + p * math.log(len(self.lengths))

            if bic < lowest_bic:
                lowest_bic = bic
                best_n = n
        return self.base_model(best_n)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_n = 0
        highest_dic = float('-inf')
        for n in range(self.min_n_components, self.max_n_components + 1):
            # Calculate log(P(X(i)) as logL
            model = self.base_model(n)
            # Use try/except to catch ValueError: rows of transmat_ must sum to 1.0
            # https://discussions.udacity.com/t/hmmlearn-valueerror-rows-of-transmat--must-sum-to-1-0/229995/7
            try:
                logL = model.score(self.X, self.lengths)
            except (ValueError, AttributeError):
                continue

            # Calculate anti-evidence term
            anti_evidence_term = 0
            for word in self.words:
                if word == self.this_word:
                    continue
                else:
                    anti_X, anti_lengths = self.hwords[word]
                    try:
                        anti_evidence_term += model.score(anti_X, anti_lengths)
                    except ValueError:
                        continue

            # Calculate DIC
            dic = logL - (1 / (len(self.words) - 1)) * anti_evidence_term

            if dic > highest_dic:
                highest_dic = dic
                best_n = n
        return self.base_model(best_n)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Only split into folds if len(sequences) >= 3
        # Else, just return n of highest logL
        # https://discussions.udacity.com/t/question-with-model-selection-cross-validation/233047/2?u=aj.jenkins123
        best_n = 0
        highest_logL = float('-inf')
        if len(self.sequences) >= 3:
            for n in range(self.min_n_components, self.max_n_components + 1):
                split_method = KFold()
                count = 0
                cumulative_logL = 0
                # Calculate average logL of all training folds
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                    test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                    training_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                                 random_state=self.random_state,
                                                 verbose=False).fit(train_X, train_lengths)
                    # Use try/except to catch ValueError: rows of transmat_ must sum to 1.0
                    # https://discussions.udacity.com/t/hmmlearn-valueerror-rows-of-transmat--must-sum-to-1-0/229995/7
                    try:
                        cumulative_logL += training_model.score(test_X, test_lengths)
                        count += 1
                    except (ValueError, AttributeError):
                        continue

                # If all training folds fail, skip this n
                if count == 0:
                    continue

                avg_logL = cumulative_logL / count
                if avg_logL > highest_logL:
                    highest_logL = avg_logL
                    best_n = n
        else:
            for n in range(self.min_n_components, self.max_n_components + 1):
                try:
                    logL = self.base_model(n).score(self.X, self.lengths)
                except (ValueError, AttributeError):
                    continue
                if logL > highest_logL:
                    highest_logL = logL
                    best_n = n
        return self.base_model(best_n)
