import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import senseval
from nltk.corpus.reader.senseval import SensevalInstance
from nltk import Text
from collections import defaultdict


interest = senseval.instances('interest.pos')
sense2 = defaultdict(list)
for ins in interest:
    sense2[ins.senses[0]].extend([word for word, tag in ins.context])

for sense in sense2.keys():
    print('Sense: {}'.format(sense))
    tok = Text(sense2[sense])
    tok.concordance('interest')




# text = []
# for ins in interest:
#     text.extend([word for word, tag in ins.context])
#
# tc = Text(text)
# tc.concordance("interest",width=100, lines=50)



