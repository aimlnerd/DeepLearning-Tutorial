"""
I would suggest that you use the gensim implementation of fastText to train your own word embeddings. 
This should be much easier and faster than your own Keras implementation. 
You can start by loading a pretrained model and then continue training with your own data.
"""

from gensim.models import FastText

model = FastText.load_fasttext_format('cc.en.300.bin')

data = [['Hello', 'world'], ...] # Your own training data, a list of sentences
model.build_vocab(data, update=True)
model.train(sentences=data, 
            total_examples=model.corpus_count,
            epochs=5)

"""
Reference
https://stackoverflow.com/questions/54339299/how-to-train-my-own-custom-word-embedding-on-web-pages/54342250?noredirect=1#comment95502071_54342250
"""