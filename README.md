# Project 4: Next-Word Prediction Using Word Embeddings and a Neural Language Model

**Author:** Albert Kabore, PhD Student in Artificial Intelligence, University of the Cumberlands

**GitHub:** [Project repository](https://github.com/albertokabore/Project-4-Next-Word-Prediction-Using-Word-Embeddings-and-a-Neural-Language-Model/tree/main)

## Introduction

Next-word prediction means estimating which word is likely to follow a piece of text. Phone keyboards use this idea when they suggest words as someone types. In this project, the goal is to build a small neural language model and explain how raw text becomes training examples, word vectors and next-word predictions.

The dataset is *The Adventures of Sherlock Holmes*, from Kaggle. The pipeline cleans the text, converts it to lowercase, removes punctuation with an exception for apostrophes inside words, and splits the text into tokens. It then builds a vocabulary and creates examples in which five context words are used to predict the next word.

Word2Vec learns a 100-dimensional vector for each retained word from the training text. These vectors initialize an embedding layer, and an LSTM neural network learns to use them to predict the next word. The vocabulary, Word2Vec model and neural-network weights are learned from the training split. A separate validation split selects the checkpoint, and the final held-out split is used for testing.

The project demonstrates all nine assignment steps with Python code, pseudocode and plain-English explanations. Its reported test scores apply to known-word targets only; the evaluation section explains the excluded words and their effect on accuracy.

The project includes Python code for every required step, the pseudocode below, and a [notebook](Next_Word_Prediction.ipynb) with explanations and recorded outputs.

## Dataset and collection

The project uses Kaggle's [Next Word Prediction dataset](https://www.kaggle.com/datasets/ronikdedhia/next-word-prediction), containing *The Adventures of Sherlock Holmes* by Arthur Conan Doyle in `1661-0.txt`.

The local file is 607,791 bytes and contains 107,602 whitespace-separated words before cleaning.

The collection function downloads the dataset with `kagglehub` and copies the text into `data/raw/`. Later runs use that copy.

## Running the project

Use Python 3.10 or later from this project folder:

```bash
python -m pip install -r requirements.txt
python -m nwp.pipeline
python -m nwp.predict "my dear"
python -m nwp.predict "it was a" --generate 12 --temperature 0.7
python -m pytest -q
```

The full pipeline downloads data if needed, trains, evaluates, and saves files to `artifacts/`. Configuration is in [config.py](nwp/config.py); for example, `python -m nwp.pipeline --epochs 5 --context-size 4` changes two settings.

To check existing results without retraining:

```bash
python -m nwp.verify --output artifacts/verification.json
```

Verification requires the raw text, saved model, vocabulary, configuration, Word2Vec model, and metrics. Data and large model files are ignored by Git, so a fresh clone needs a training run or those files supplied separately.

To use the notebook, install Jupyter separately if needed, then open [Next_Word_Prediction.ipynb](Next_Word_Prediction.ipynb) and run the cells in order from this folder.

## Pseudocode

```text
1. COLLECT A DATASET
   If data/raw/1661-0.txt is missing:
       download Kaggle dataset "ronikdedhia/next-word-prediction"
       copy 1661-0.txt into data/raw/
   raw_text = read the file as UTF-8

2. CLEAN THE TEXT
   Remove text outside the Gutenberg START and END markers, when present
   Normalize curly quotes, dashes and accented characters to ASCII
   Split at sentence-ending punctuation followed by whitespace, and blank lines

3-5. LOWERCASE, REMOVE PUNCTUATION, TOKENIZE
   For each text segment:
       convert to lowercase
       replace punctuation with spaces, except apostrophes inside words
       collapse whitespace and split into word tokens
       discard empty token lists

   Split the segments in reading order:
       first 80% for training, next 10% for validation, remainder for testing

6. BUILD A VOCABULARY
   Count words in the training split only
   Keep words seen at least twice, ordered by frequency, then alphabetically
   Add <pad> with id 0 and <unk> with id 1
   Assign an integer id to each retained word
   Map other words to <unk>

7. GENERATE TRAINING SEQUENCES
   For each split separately:
       stream = five <pad> ids followed by all word ids in that split
       For each target position:
           input = the previous five ids
           target = the id at this position
           keep this example only if its target is not <unk>
   Record how many targets were excluded in each split

8. CONVERT WORDS TO VECTORS WITH WORD2VEC
   Train skip-gram Word2Vec on training segments only:
       100 dimensions, maximum window 5, 10 negative samples, 30 epochs
   Build a matrix with one row per vocabulary entry:
       ordinary word -> its learned vector
       <pad> -> zeros
       <unk> -> mean learned vector

9. TRAIN A NEURAL NETWORK
   Model = embedding lookup initialized with that matrix
           -> two LSTM layers with 256 hidden units each
           -> dropout -> linear layer with one score per vocabulary entry
   Use AdamW with learning rate 0.002 and cross-entropy loss
   For up to 20 epochs:
       keep embeddings frozen for epochs 1-2; then allow them to change
       shuffle training examples into batches of 256
       compute scores and loss; backpropagate
       clip gradient norm to 1.0 and update weights
       measure validation loss and accuracy
       reduce learning rate when validation loss stalls
       retain weights with the lowest validation loss
       stop after three epochs without improvement
   Restore the retained weights

EVALUATE
   Measure test cross-entropy, perplexity, top-1 accuracy and top-5 accuracy
   Compare accuracy with a predictor that always guesses common training words
   Report that these scores cover known-word targets only
```

## How the code works, in plain English

**1. Collect a dataset.**

**2. Clean the text.**

**Observations**

- Cleaning reduced the text from 581,887 to 562,328 characters. The ASCII check returned True, confirming that the cleaned text contains only ASCII characters.
- The splitter produced 6,989 segments. Title-page material still remains inside the book markers, so cleaning did not isolate story text completely.

**3. Convert to lowercase.**

**Observations**

- The samples show ?His? becoming ?his? and ?He? becoming ?he?; word order and punctuation remain unchanged.
- Case variants will share vocabulary entries, reducing duplicate forms. Lowercasing also changes ?I? to ?i?, so capitalization information is lost even though the sequence of words is preserved.

**4. Remove punctuation.**

**Observations**

- Commas and sentence-ending periods disappear from the samples while the words remain separated, preventing punctuation from creating extra word variants.
- The example "it's 'quite' simple, watson!" becomes "it's quite simple watson". Apostrophes inside contractions are retained, while quotation apostrophes are removed; punctuation handling therefore includes an explicit exception.

**5. Tokenize.**

**Observations**

- Tokenization produced 105,428 tokens across 6,989 segments, with 8,023 distinct words. The displayed lists separate words into individual items for vocabulary construction.
- The ordered split contains 5,591 training segments, 698 validation segments and 700 test segments. These percentages count segments, so token totals need not match exactly.

**6. Build a vocabulary.**

**Observations**

- The vocabulary retains 3,898 entries, including padding and unknown tokens, from 7,280 distinct training words. It covers 95.9% of training tokens.
- ?xyzzy? maps to unknown ID 1, while ?my?, ?dear? and ?watson? receive known IDs. The frequency chart is dominated by common words, which provide a useful baseline.

**7. Generate training sequences.**

**Observations**

- The input arrays contain five word IDs per example. Training, validation and testing produce 79,763, 10,656 and 9,935 examples, respectively.
- Sliding windows pair each context with its following word. Unknown targets are skipped, including 895 test tokens; therefore, these sequence counts exclude words outside the training vocabulary.

**8. Convert words to vectors.**

**Observations**

- The embedding matrix has 3,898 rows and 100 dimensions; Word2Vec covers all 3,896 ordinary vocabulary words. Padding and unknown entries receive separate initializations.
- ?holmes? neighbors ?sherlock?, while ?lady? neighbors ?neatly?, showing mixed context relationships. The two-dimensional projection compresses the vectors, so visible distances cannot represent all information.

**9. Train a neural network.**

**Observations**

- The model has 2,284,514 parameters. Training loss falls from 6.279 to 4.038, showing improved fit to training examples.
- Validation loss reaches its minimum, approximately 4.901, at epoch 8, then rises while training loss falls. This pattern suggests overfitting; early stopping ends training at epoch 11 and restores epoch 8.

**Prediction and evaluation.** [predict.py](nwp/predict.py) uses the last five normalized input words, pads shorter inputs, and turns output scores into probabilities. It excludes `<pad>` and `<unk>` from suggestions. Generation adds one word and repeats, either choosing the highest-scoring word or sampling with a temperature.

Evaluation uses the model's full output scores, while the suggestion function masks special tokens and renormalizes probabilities. Perplexity is `exp(mean cross-entropy)`, using natural logarithms. It summarizes predictive uncertainty on the scored targets; it is not a literal count of possible words.

## Results

Training and test results are saved in [metrics.json](artifacts/metrics.json).

| Item | Count |
|---|---:|
| Preprocessed segments / tokens | 6,989 / 105,428 |
| Vocabulary entries, including two special tokens | 3,898 |
| Training / validation / test sequences | 79,763 / 10,656 / 9,935 |
| Model parameters | 2,284,514 |

| Split | Total tokens | Scored targets | Excluded unknown targets |
|---|---:|---:|---:|
| Training | 83,147 | 79,763 | 3,384 |
| Validation | 11,451 | 10,656 | 795 |
| Test | 10,830 | 9,935 | 895 |

Only 91.7% of test tokens are scored. The following accuracies and perplexity apply to that subset:

| Predictor | Top-1 accuracy | Top-5 accuracy | Perplexity |
|---|---:|---:|---:|
| Common-word baseline | 5.3% | 16.8% | Not computed |
| Word2Vec + LSTM | 15.2% | 33.9% | 137.8 |

If the 895 excluded words are counted as incorrect exact-word predictions, the model's accuracies across **all 10,830 test tokens** are **13.9% top-1** and **31.1% top-5**. This denominator change does not provide full-test perplexity: the implementation does not score those original unknown words as output classes.

The saved history contains 11 epochs, with the lowest validation loss at epoch 8.

Example next-word suggestions:

| Input | Five suggestions | Probability of first suggestion |
|---|---|---:|
| my dear | fellow, watson, i, and, holmes | 7.67% |
| what do you | see, mean, do, have, know | 5.41% |
| i shall be happy to | be, do, see, have, find | 8.46% |
| my dear watson | i, and, said, you, he | 12.28% |

These are demonstrations, not separately labeled correctness tests. A phrase can have several reasonable continuations. Low top-word probability alone does not establish poor calibration. The held-out accuracy compares predictions with the actual next word in the book.

## Final Insights and Recommendations

**Final insights**

1. **Context improved prediction by a measurable margin.** On 9,935 known-word test targets, the model achieved 15.2% top-1 and 33.9% top-5 accuracy, compared with 5.3% and 16.8% for the common-word baseline. The gains of 9.9 and 17.1 percentage points show that the complete system learned useful context patterns.
2. **Epoch 8 provided the best validation fit.** Between epochs 8 and 11, training loss fell from 4.552 to 4.038 while validation loss rose from 4.901 to 4.961. This divergence is consistent with overfitting and supports retaining the epoch-8 checkpoint rather than the final weights.
3. **Unknown-word handling materially affects reported accuracy.** The evaluation excludes 895 of 10,830 test tokens, or 8.3%. Counting those tokens as incorrect lowers top-1 accuracy to 13.9% and top-5 accuracy to 31.1%. Perplexity 137.8 describes only the known-target subset, so it cannot summarize performance on every original test word.
4. **Embedding coverage is complete, but the neighbors show mixed relationships.** Word2Vec provides vectors for all 3,896 ordinary vocabulary words. Pairs such as *holmes* with *sherlock* are intuitive, whereas *watson* with *index* is less so. Complete vector coverage alone does not establish embedding quality or Word2Vec's contribution to prediction accuracy.
5. **Next-word accuracy and generation quality reveal different limitations.** For *what do you*, the model suggests *see, mean, do, have,* and *know*. Yet greedy generation repeats *and i have heard* in the displayed sample. The accuracy gains therefore need to be assessed alongside repetition and fluency when judging generated text.
6. **The findings are specific to one book and one run.** The model has 2,284,514 parameters and uses 83,147 training tokens. These counts describe the experiment's scale, but they do not establish the cause of overfitting. Performance across training seeds, other authors and other writing styles remains unmeasured.

**Recommendations**

The following experiments address these findings; their benefits still need to be measured.

1. **Prioritize coverage in the next evaluation.** Keep the original word sequence and split boundaries fixed, and compare vocabulary thresholds of one and two occurrences. Report excluded targets, known-target accuracy and full-token exact-word accuracy together. This directly addresses the 895 test words omitted from the current scores.
2. **Optimize validation performance rather than extending training.** Preserve early stopping and selection by validation loss. Compare the current 256-unit LSTM with smaller hidden sizes and alternative regularization settings. The rise in validation loss after epoch 8 makes these controlled comparisons more informative than simply adding epochs.
3. **Measure Word2Vec's contribution directly.** Compare randomly initialized embeddings, frozen Word2Vec embeddings and fine-tuned Word2Vec embeddings using the same splits, architecture, training budget and seeds. The baseline comparison measures the complete system; this experiment would show whether learned word vectors improve next-word prediction under comparable conditions.
4. **Test data expansion and cleaning separately.** Compare several training-corpus sizes while keeping the held-out evaluation text fixed. In a separate comparison, remove remaining title-page, credit and contents material before retraining. Measuring each change separately would show whether additional text or cleaner input improves validation performance.
5. **Evaluate decoding against the observed repetition.** Generate equal-length continuations for the same prompts using greedy decoding and temperature sampling, including the existing temperature of 0.7. Compare repeated phrase counts and assess grammatical coherence. This would test whether sampling reduces loops while preserving readable continuations.
6. **Measure stability before making broader performance claims.** Repeat training with several fixed seeds and report mean accuracy with its variation. Then evaluate the selected configuration on text from another author without tuning on that test set. This would assess whether the observed gains extend beyond one run and one book.

## Files

| File | Purpose |
|---|---|
| `nwp/data.py`, `nwp/preprocess.py` | Collect and prepare text |
| `nwp/vocab.py`, `nwp/sequences.py` | Build vocabulary and examples |
| `nwp/embeddings.py`, `nwp/model.py`, `nwp/trainer.py` | Learn vectors and train/evaluate the LSTM |
| `nwp/pipeline.py`, `nwp/predict.py` | Run the project and make predictions |
| `nwp/verify.py` | Recompute and check saved results |
| `tests/test_pipeline.py` | Automated checks, including an artificial learning test |
| `Next_Word_Prediction.ipynb` | Assignment walkthrough |

## Conclusion

This project implements a complete next-word prediction pipeline: collecting a dataset, cleaning and lowercasing the text, removing punctuation with a stated apostrophe exception, tokenizing, building a vocabulary, creating training sequences, learning Word2Vec vectors, and training an LSTM neural network. The code and pseudocode show how these steps connect, and the explanations describe their purpose in plain English.

The saved checkpoint achieved 15.2% top-1 accuracy, 33.9% top-5 accuracy and perplexity 137.8 on known-word test targets, exceeding the common-word baseline on that subset. Because 895 unknown targets were excluded, these figures do not describe every test token. Counting the excluded targets as incorrect gives 13.9% top-1 and 31.1% top-5 accuracy across the full test stream.

The results demonstrate that this implementation can learn useful context patterns, while the loss curves and generated samples show limitations in generalization and repetition. Further comparisons are needed to measure Word2Vec's contribution and identify which changes improve performance.
