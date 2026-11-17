# Project 4 review

The project implements all nine assignment steps and supplies code, pseudocode and plain-English explanations. The saved results can be reproduced from the supplied corpus and checkpoint. The README and notebook have been revised to state what those results cover and to remove conclusions the experiment did not establish.

## Assignment coverage

| Requirement | Implementation | Review finding |
|---|---|---|
| Collect a dataset | `nwp/data.py` | Downloads the configured Kaggle dataset and caches the text. |
| Clean the text | `nwp/preprocess.py` | Removes outer Gutenberg boilerplate and normalizes characters. Front matter inside the markers remains. |
| Convert to lowercase | `to_lowercase` | Implemented. |
| Remove punctuation | `remove_punctuation` | Implemented with a documented exception for apostrophes inside words. It does not remove every punctuation character. |
| Tokenize | `tokenize`, `preprocess_corpus` | Splits normalized text into word lists. Sentence segmentation uses simple rules. |
| Build a vocabulary | `nwp/vocab.py` | Counts training words only, retains words seen twice, and includes padding/unknown tokens. |
| Generate training sequences | `nwp/sequences.py` | Uses five-word sliding windows, separately within each split. Unknown targets are excluded. |
| Use Word2Vec | `nwp/embeddings.py` | Learns skip-gram vectors from training text and initializes the embedding layer with them. |
| Train a neural network | `nwp/model.py`, `nwp/trainer.py` | Trains an LSTM using cross-entropy and AdamW, with validation and early stopping. |
| Explain the code | `README.md`, notebook | Provides explanations and pseudocode for every required step. |
| Evaluate the model | `evaluate`, `unigram_baseline` | Reports held-out metrics and compares with a common-word baseline, on known targets only. |

The apostrophe exception should remain visible in the submission. If the rubric requires removal of **all** punctuation without exceptions, this preprocessing choice would need changing and the model would need retraining; the present results must not be reused after changing tokenization.

## Results checked against the files

Verification rebuilt the text segments, vocabulary and sequences, then evaluated the saved checkpoint on validation and test data. It checked the saved Word2Vec vocabulary counts against training text, reproduced its recorded neighbors, and reproduced every demonstration in `metrics.json`.

| Quantity | Verified result |
|---|---:|
| Raw file size | 607,791 bytes |
| Preprocessed segments / tokens | 6,989 / 105,428 |
| Vocabulary entries | 3,898, including two special tokens |
| Parameters | 2,284,514 |
| Scored test targets | 9,935 of 10,830 tokens |
| Excluded unknown test targets | 895, or 8.3% |
| Test top-1 / top-5 accuracy, scored subset | 15.1887% / 33.9406% |
| Test perplexity, scored subset | 137.77325 |
| Baseline top-1 / top-5, same subset | 5.2743% / 16.8395% |
| All-token top-1 / top-5, excluded words counted incorrect | 13.9335% / 31.1357% |

The original test loss was `4.9256091843`; re-evaluation produced `4.9256091966`. The difference is approximately `1.23e-8`. Both accuracies match exactly. The saved history selects epoch 8, and the checkpoint reproduces that epoch's validation metrics within floating-point tolerance.

Run `python -m nwp.verify --output artifacts/verification.json` to repeat the checks. The [verification report](artifacts/verification.json) records fingerprints and the environment used for verification. These are verification-time versions, not an authenticated record of the original training environment.

Validation completed: 18 automated tests passed; all 13 checks on the original artifacts passed; the notebook passed format validation and syntax checks. Its revised evaluation and save cells were also executed against the supplied checkpoint in a temporary directory, where all 15 checks passed, including the new fingerprint and target-coverage fields. Deliberately changing the temporary test accuracy was correctly detected. The original corpus, model and metrics retained their recorded fingerprints.

## Corrections made

- Made unknown-target exclusions explicit wherever results are explained. The reported perplexity does not cover the full original test vocabulary.
- Replaced claims that dataset size was the proven main bottleneck with the supported observation: the saved loss curves are consistent with overfitting.
- Removed claims that Word2Vec was proven better than random initialization. No such comparison was performed.
- Stopped labeling an unlabeled prompt continuation as automatically wrong or using low top-word probability as evidence of poor calibration.
- Corrected the explanation of gradient clipping: it limits gradient norm, not each AdamW parameter update.
- Qualified authentication, runtime, reproducibility and sentence-segmentation claims.
- Fixed typed Unicode prompts so curly apostrophes and accented characters receive the same normalization as corpus text.
- Fixed temperature sampling to preserve zero probability for special tokens; invalid suggestion counts now raise a clear error.
- Updated notebook saving to write current metrics and a training plot with the current model. Previously, rerunning the notebook could leave old metrics beside newly saved weights.
- Added raw-file fingerprints and excluded-target counts to future pipeline/notebook reports. The original `metrics.json` and trained model were preserved.
- Cleared outputs for revised notebook code cells instead of presenting historical outputs as new executions.

## What this review does not establish

The supplied file contains recognizable book text and identifies its title and source in its header. The publisher's Kaggle page exists, but consistency with a local checkpoint does not independently authenticate the file's download history or establish that no historical manipulation occurred. No entire training run was repeated. Historical runtime and the full sequence of training updates remain unverified.

The artificial sentences in the automated learning test are explicitly test fixtures; they are separate from the book corpus and reported project results. The author and affiliation line was retained from the original submission and was not independently verified.

This is one model run on one book. It does not establish superiority over other architectures, the isolated value of Word2Vec, performance on other domains, or variation across seeds. The technical explanation was checked against primary references linked in the [README](README.md), including the Word2Vec paper, Gensim, KaggleHub and PyTorch documentation.
