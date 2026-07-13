"""
fastnltk.corpus — Drop-in replacement for nltk.corpus.

Pure Python shim — corpus reading is I/O bound, Rust offers no benefit.
"""

import nltk.corpus as _nltk_corpus

# Re-export all corpus readers
from nltk.corpus import *  # noqa: F401, F403

# Re-export the LazyCorpusLoader instances
from nltk.corpus import (  # noqa: F811
    # Plaintext corpora
    abc,
    brown,
    cess_cat,
    cess_esp,
    floresta,
    genesis,
    gutenberg,
    inaugural,
    mac_morpho,
    movie_reviews,
    nps_chat,
    opinion_lexicon,
    rpm_category,
    state_union,
    stopwords,
    twitter_samples,
    webtext,
    words,
    # Tagged corpora
    alpino,
    biocreative_ppi,
    brown_tagged,
    cess_cat_tagged,
    cess_esp_tagged,
    conll2000,
    conll2002,
    conll2007,
    crubadan,
    floresta_tagged,
    gazetteers,
    indian,
    javascript,
    mac_morpho_tagged,
    nps_chat_tagged,
    paradise,
    ptb,
    qc,
    senseval,
    sinica_treebank_tagged,
    tiger,
    treebank,
    treebank_chunk,
    twitter_samples_tagged,
    udhr,
    universal_treebanks_v20,
    verbnet,
    # Chunked corpora
    conll2000_chunked,
    treebank_chunk,
    # Corpora for classification
    movie_reviews,
    twitter_samples,
    # WordNet
    wordnet,
    wordnet_ic,
    omw,
    omw_1_4,
    # SentiWordNet
    sentiwordnet,
    # Multilingual corpora
    europarl_raw,
    udhr2,
    # Proposition Bank
    propbank,
    nombank,
    # Framenet
    framenet_v15,
    framenet_v17,
    # Other
    ppattach,
    reuters,
    semcor,
    subjectivity,
    switchboard,
    timit,
    toolbox,
    ycoe,
)
