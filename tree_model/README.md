<p align="center">
<img src="https://github.com/Cisco-Talos/fnc-1/blob/master/images/solat-in-the-swen.gif" alt="TALOS IN THE NEWS"/>
</p>

# SOLAT IN THE SWEN - tree\_model

## Overview
This model takes as input a few text-based features derived from the headline and body of an article. Then it feeds the features into Gradient Boosted Trees to predict the relation between the headline and the body (`agree`/`disagree`/`discuss`/`unrelated`)

<p align="center">
<img src="https://github.com/Cisco-Talos/fnc-1/blob/master/images/diagrams_light/tree_model_light.png" alt="Tree Model Diagram"/>
</p>

## Feature Engineering

**1. Preprocessing (`generateFeatures.py`)**

The labels (`agree`, `disagree`, `discuss`, `unrelated`) are encoded into numeric target values as (`0`, `1`, `2`, `3`). The text of headline and body are then tokenized and stemmed (by `preprocess_data()` in `helpers.py`). Finally Uni-grams, bi-grams and tri-grams are created out of the list of tokens. These grams and the original text are used by the following feature extractor modules.

**2. Basic Count Features (`CountFeatureGenerator.py`)**

This module takes the uni-grams, bi-grams and tri-grams and creates various counts and ratios which could potentially signify how a body text is related to a headline. Specifically, it counts how many times a gram appears in the headline, how many unique grams there are in the headline, and the ratio between the two. The same statistics are computed for the body text, too. It then calculates how many grams in the headline also appear in the body text, and a normalized version of this overlapping count by the number of grams in the headline. The results are saved in the pickle file which will be read back in by the classifier.

**3. [TF-IDF](https://en.wikipedia.org/wiki/Tf–idf) Features (`TfidfFeatureGenerator.py`)**

This module constructs sparse vector representations of the headline and body by calculating the Term-Frequency of each gram and normalize it by its Inverse-Document Frequency. First off a `TfidfVectorizer` is fit to the concatenations of headline and body text to obtain the vocabulary. Then using the same vocabulary it separately fits and transforms the headline grams and body grams into sparse vectors. It also calculates the cosine similarity between the headline vector and the body vector.

**4. SVD Features (`SvdFeatureGenerator.py`)**

This module takes the TF-IDF features and applies Singular-Value Decomposition to them to obtain a compact, dense vector representation of the headline and body respectively. This procedure is [well known](https://en.wikipedia.org/wiki/Latent_semantic_analysis) and corresponds to finding the latent `topics` involved in the corpus and represent each headline/body text as a mixture of these `topics`. The cosine similarities between the SVD features of headline and body text are also computed. This similarity metric is very indicative of whether the body is related to the headline or not.

**5. Word2Vec Features (`Word2VecFeatureGenerator.py`)**

This module utilizes the pre-trained [word vectors](https://arxiv.org/abs/1301.3781) from public sources, add them up to build vector representations of the headline and body. The word vectors were trained on a Google News corpus with 100 billion words and a vocabulary size of 3 million. The resulting word vectors can be used to find synonyms, predict the next word given the previous words, or to manipulate semantics. For example, when you calculate `vector(Germany) - Vector(Berlin) + Vector(England)` you will obtain a vector that is very close to `Vector(London)`. For the current problem constructing the vector representation out of word vectors could potentially overcome the ambiguities introduced by the fact that headline and body may use synonyms instead of exact words.

**6. Sentiment Features (`SentimentFeatureGenerator.py`)**

This modules uses the Sentiment Analyzer in the `NLTK` package to assign a sentiment polarity score to the headline and body separately. For example, negative score means the text shows a negative opinion of something. This score can be informative of whether the body is being positive about a subject while the headline is being negative. But it does not indicate whether it's the same subject that appears in the body and headline; however, this piece of information should be preserved in other features.

## Classifier Construction (xgb\_train\_cvBodyId.py)
The classifier used in this model is [Gradient Boosted Trees](https://en.wikipedia.org/wiki/Gradient_boosting). A very efficient implementation of GBDT is [XGBoost](http://xgboost.readthedocs.io/en/latest/). 10-fold cross-validation is used to estimate the performance of this model.

## Library Dependencies
* Python 2.7
* Scipy Stack (`numpy`, `scipy` and `pandas`)
* [scikit-learn](http://scikit-learn.org/stable/)
* [XGBoost](http://xgboost.readthedocs.io/en/latest/)
* [gensim (for word2vec)](https://radimrehurek.com/gensim/)
* [NLTK (python NLP library)](http://www.nltk.org)

## Procedure
**1. Install all the dependencies**

**2.`git clone https://github.com/Cisco-Talos/fnc-1.git`**

**3. Download the `word2vec` [model](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/) trained on Google News corpus. The file `GoogleNews-vectors-negative300.bin` has to be present in both `/deep_learning_model` and `/tree_model`.**

**4. Generate predictions from the `deep_learning_model` by running `/deep_learning_model/clf.py`.  This output is represented by `dosiblOutputFinal.csv`, renamed as `deepoutput.csv` in our directories.**

**5. Run `generateFeatures.py` to produce all the feature files (`train_stances_processed.csv` and `test_stances_processed.csv` are the (encoding-wise) cleaned-up version of the orginal csv files, same as the updated files in the orginal [FNC-1 GitHub](https://github.com/FakeNewsChallenge/fnc-1)). The following files will be generated:**

```
data.pkl
test.basic.pkl
test.body.senti.pkl
test.body.svd.pkl
test.body.tfidf.pkl
test.body.word2vec.pkl
test.headline.senti.pkl
test.headline.svd.pkl
test.headline.tfidf.pkl
test.headline.word2vec.pkl
test.sim.svd.pkl
test.sim.tfidf.pkl
test.sim.word2vec.pkl
train.basic.pkl
train.body.senti.pkl
train.body.svd.pkl
train.body.tfidf.pkl
train.body.word2vec.pkl
train.headline.senti.pkl
train.headline.svd.pkl
train.headline.tfidf.pkl
train.headline.word2vec.pkl
train.sim.svd.pkl
train.sim.tfidf.pkl
train.sim.word2vec.pkl
```

**6. Comment out line 121 in `TfidfFeatureGenerator.py`, then uncomment line 122 in the same file. Raw TF-IDF vectors are needed by `SvdFeatureGenerator.py` during feature generation, but only the similarities are needed for training.**

**7. Run `xgb_train_cvBodyId.py` to train and make predictions on the test set. Output file is `predtest_cor2.csv` (for model averaging)**

**8. Run `average.py` to computed the weighted average of this tree model and the CNN model from Doug Sibley (`deepoutput.csv`). The final output file is `averaged_2models_cor4.csv`, which is the final submission of our team.**

The `run()` function in `average.py` is used to find the optimal weights when combining the models. Uncomment line 164 and 165 to call this function. It needs the probabilities predictions from both the deep learning (`dosiblOutput.csv`) and tree models (`predtest_cor2.csv`) on the holdout set of the official split. This step is not necessary, as we have decided that the weights we were going to use is 0.5 and 0.5

All the output files are also stored under `./results/` and all parameters are hard-coded. 

## Questions?
Contact Yuxi Pan (`yuxpan@cisco.com`) for bugs and questions.

 <!--
   Copyright 2017 Cisco Systems, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
     http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 -->
