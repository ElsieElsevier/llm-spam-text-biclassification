# SMS Spam Classification: TF-IDF vs. BERT Embeddings

A binary classifier that labels SMS messages as spam or ham (legitimate), built as a team course project. The notebook compares TF-IDF baselines with classifiers trained on fixed BERT embeddings.

The goal was to catch spam while rarely flagging legitimate messages. The targets were precision above 95%, specificity above 97% and F1 above 90%.

## Results

Test set of 774 unique messages, 96 of them spam, using a 0.5 decision threshold:

| Model | Accuracy | Precision | Recall | F1 | Specificity | False positives | False negatives |
|---|---|---|---|---|---|---|---|
| TF-IDF + Naive Bayes | 93.93% | 100.00% | 51.04% | 67.59% | 100.00% | 0 | 47 |
| TF-IDF + logistic regression | 97.80% | 91.58% | 90.62% | 91.10% | 98.82% | 8 | 9 |
| BERT + neural network (selected) | 98.45% | 93.75% | 93.75% | 93.75% | 99.12% | 6 | 6 |
| BERT + logistic regression | 98.71% | 95.74% | 93.75% | 94.74% | 99.41% | 4 | 6 |

The BERT + neural network model was selected because it had the best validation F1 (91.80%, against 91.58% for TF-IDF + logistic regression). It meets the specificity and F1 targets but misses the precision target.

BERT + logistic regression scored higher on the test set, but the test set isn't used to choose a model. Its validation F1 was 90.61%.

### Limitations

- **Small differences:** the gaps between the top three models amount to a few messages on a single split, so they don't establish a clear winner.
- **Training cap:** the neural network ran to its 300-epoch limit without early stopping, so it may not have finished converging.
- **Weak on some new messages:** in a spot check of six hand-written messages, the selected model labeled an obvious "WINNER! ... free prize" message as ham (probability 0.46). TF-IDF + logistic regression caught that message but missed a fake delivery notice.
- **Dated data:** the dataset is old and doesn't reflect current scams. The models haven't been tested on recent real-world spam.

## How it works

1. **Clean the split:** the 5,572 rows contain 415 repeated messages. These are removed before a stratified 70/15/15 split, so no test message also appears in training.
2. **TF-IDF baselines:** word and two-word-phrase features, fitted on the training set only. Logistic regression's regularization is chosen on the validation set.
3. **BERT embeddings:** a frozen BERT model from TensorFlow Hub turns each message into a 768-dimension vector. BERT gets the raw text, with no stop-word removal or stemming.
4. **Classifiers:** a small neural network (64 and 32 units, with dropout and L2 regularization) and a logistic regression are trained on those vectors.
5. **Selection and saving:** the best validation F1 selects the model, and all fitted models are saved for reuse.

Word2Vec and stop-word removal were explored early in the project but aren't part of this pipeline.

## Usage

Tested on Linux (WSL) with Python 3.12 on CPU, where a full run takes about 11 minutes. `tensorflow-text` has no Windows build, so use Linux, macOS or WSL on Windows. Colab hasn't been tested with these pinned versions.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyter
jupyter notebook spam_classification.ipynb
```

Seeds are fixed, and repeated CPU runs produced identical results. Models are saved to `artifacts/`, which isn't tracked by git.

After running the notebook, classify new messages in a new cell:

```python
loaded_models, metadata = load_models()

# Selected model (BERT + neural network)
predict(["Are we still meeting for lunch?"], loaded_models, metadata, embedder=bert)

# TF-IDF model, which doesn't need BERT
predict(["Are we still meeting for lunch?"], loaded_models, metadata, "tfidf_logreg")
```

## Data

[Spam Text Message Classification](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification/data) from Kaggle: 4,825 ham and 747 spam messages before deduplication, and 4,516 ham and 641 spam after.

## Contributors

Jin Bai, Tim Burke, Jeremiah Fowler and Navya Rani Kanaka. The TF-IDF baselines are adapted from Jeremiah Fowler's preprocessing work.

## License

This project has no license.

## Contact

jin.ken.bai@gmail.com
