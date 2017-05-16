import warnings
from asl_data import SinglesData

NO_SCORE = float('-inf')

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []

    # create a list of probabilities
    for i in range(test_set.num_items):

        # get test sample values
        X, length = test_set.get_item_Xlengths(i)

        # iterate over model dictionary to score words in new dictionary scores
        scores = {}
        for word, model in models.items():
            try:
                scores[word] = model.score(X, length)
            except:
                scores[word] = NO_SCORE
        probabilities.append(scores)

    # create a list of best guesses
    guesses = [max(p, key=p.get) for p in probabilities]

    # return probabilities, guesses
    return probabilities, guesses

