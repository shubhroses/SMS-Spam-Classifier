# SMS Spam Classifier

A Jupyter notebook that classifies SMS messages as spam or ham (not spam). It turns each message into a sentence embedding with a pretrained RoBERTa model, trains four scikit-learn classifiers on those embeddings, and reports accuracy, precision, recall and F1 for each.

The notebook is `sms_spam_classifier.ipynb`. It was run in Google Colab in 2022, and the outputs saved in it are from that run. It was first committed as `5170_Final.ipynb`.

## Approach

1. **Embeddings.** The notebook loads the `roberta-base` tokenizer and model from Hugging Face `transformers`, the model with `output_hidden_states=True`. Each message is tokenized without RoBERTa's start and end tokens and run through the model in evaluation mode with gradients disabled. The notebook takes the second-to-last hidden layer and averages it over the tokens, which gives one vector of 768 values per message. RoBERTa is used only as a fixed feature extractor; it is not fine-tuned.
2. **Data.** The `ucirvine/sms_spam` dataset is loaded with Hugging Face `datasets`: 5574 messages labeled 1 for spam and 0 for ham. The first 70% of the messages, in their original order, are the training set and the rest are the test set.
3. **Classifiers.** Gaussian Naive Bayes, a random forest, logistic regression and a multi-layer perceptron (`MLPClassifier`) are fitted on the training embeddings, each with scikit-learn's default settings.
4. **Evaluation.** Each model is scored on the test set for accuracy, and for precision, recall and F1 on the spam class.

## Results

These numbers are from the original run in 2022. They are copied from the outputs saved in the notebook, written as the notebook printed them (it rounds to three decimal places).

| Classifier | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
| --- | --- | --- | --- | --- |
| Gaussian Naive Bayes | 0.987 | 0.99 | 0.912 | 0.95 |
| Random forest | 0.983 | 1.0 | 0.877 | 0.935 |
| Logistic regression | 0.993 | 0.995 | 0.956 | 0.975 |
| MLP | 0.995 | 1.0 | 0.965 | 0.982 |

The notebook also scores the four models on a list of 15 messages written into one of its cells. The accuracies it printed are 1.0 for logistic regression and for the MLP, 0.8666666666666667 for Gaussian Naive Bayes and 0.7333333333333333 for the random forest. Most of those messages are copies or near copies of dataset rows that fall in the training set, so this is a sanity check rather than an independent test.

The last cell is interactive: choose one of the four classifiers, type a message, and it prints `Spam` or `Ham`.

Three runs on current library versions printed numbers close to these. They are under [Checked on current versions](#checked-on-current-versions).

## Running the notebook

The outputs saved in the notebook are from the original run in 2022, in Google Colab on Python 3.7 with `transformers` 4.18.0 and `datasets` 2.2.1. They have not been regenerated. Two source lines have changed since that run, because the originals fail on current versions of IPython and `datasets`:

- `% matplotlib inline` in the first cell is now `%matplotlib inline`.
- `datasets.load_dataset('sms_spam')` is now `datasets.load_dataset('ucirvine/sms_spam')`. The output saved under that cell still shows the old name.

With those two changes the notebook runs on current versions of its libraries, and `requirements.txt` pins the versions it was checked with. To run it in Jupyter, with Python 3.12 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt notebook ipywidgets
jupyter notebook sms_spam_classifier.ipynb
```

Then run the cells from top to bottom.

- `requirements.txt` lists the libraries the notebook imports, pinned to the versions it was last run with. Those versions need Python 3.12 or newer; the run used 3.13.
- `notebook` is the Jupyter front end. `ipywidgets` is for the download progress bars; without it the first cell prints a `TqdmWarning`.
- The notebook also runs `pip install transformers` and `pip install datasets` itself, without pinning versions. In the environment above both lines find the packages already installed and change nothing.
- In Google Colab, where the saved outputs were produced, `requirements.txt` is not used. The two `pip install` lines take care of `transformers` and `datasets`, and the notebook relies on Colab for the other libraries. Colab has not been tried again since then.
- The first run downloads the `roberta-base` weights (about 500 MB) and the dataset, so it needs internet access. No Hugging Face account is needed. Without one, `huggingface_hub` prints a warning about unauthenticated requests.
- The model runs on the CPU, one message at a time, with no batching. On an Apple M4 the two embedding cells together took under two minutes.

### Checked on current versions

On 9 October 2026 the notebook was run from top to bottom, three times, in a new virtual environment made with the commands above. The environment was Python 3.13.7 on macOS 26.6.2 with an Apple M4, IPython 9.17.1, ipykernel 7.4.0 and the versions in `requirements.txt`: `torch` 2.14.1, `transformers` 5.19.0, `datasets` 5.1.0, `scikit-learn` 1.9.1, `pandas` 3.0.6, `numpy` 2.5.3, `matplotlib` 3.11.2 and `tqdm` 4.70.1. The packages those depend on are not pinned; `pip` chose `huggingface_hub` 2.2.0, `tokenizers` 0.23.3 and `pyarrow` 26.0.0 among them. A script sent each code cell to a Jupyter kernel in order and typed the answers to the two prompts of the last cell. The Jupyter page in the browser was not used: `jupyter notebook sms_spam_classifier.ipynb` was only checked to start and to serve the notebook.

- All 33 code cells finished without an error in each run. The last cell printed `Spam` for a spam message with logistic regression and with Gaussian Naive Bayes, and `Ham` for a ham message with the MLP.
- With the two original lines put back, the first cell stops at `% matplotlib inline` with the error "Line magic function `%` not found", and the dataset cell raises `HfUriError`: "Repository id must be 'namespace/name', got 'sms_spam'".
- `ucirvine/sms_spam` is the id the Hub's API redirects `sms_spam` to. It loads 5574 messages with the fields `sms` and `label`: 4827 ham and 747 spam. They are the messages the 2022 run loaded, in the same order and with the same labels. The cache path in the saved output contains the hash of the `sms_spam.py` loader script from `datasets` 2.2.1, which builds one row per line of the file `SMSSpamCollection` in a zip from the UCI archive. That zip still has the SHA-256 that `datasets` 2.2.1 recorded for it, and the rows rebuilt from it are equal to the rows the Hub returns.

The first run printed these test-set numbers:

| Classifier | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
| --- | --- | --- | --- | --- |
| Gaussian Naive Bayes | 0.987 | 0.99 | 0.912 | 0.95 |
| Random forest | 0.986 | 1.0 | 0.895 | 0.944 |
| Logistic regression | 0.994 | 0.995 | 0.961 | 0.978 |
| MLP | 0.996 | 0.996 | 0.974 | 0.984 |

Gaussian Naive Bayes printed the same numbers as in 2022, in all three runs. Logistic regression printed the same numbers in all three runs too: it got 1662 of the 1672 test messages right, one more than in 2022, and gave no convergence warning. The random forest and the MLP have no fixed seed, and their numbers changed between runs. The random forest's accuracy was 0.986, 0.983 and 0.984, and the MLP's was 0.996, 0.995 and 0.995.

On the 15 messages, Gaussian Naive Bayes, logistic regression and the MLP printed the saved accuracies in every run. The random forest printed 0.7333333333333333, 0.8 and 0.8666666666666667.

Not checked: Google Colab, a GPU, Linux or Windows, and Python versions other than 3.13.

## Limitations

- The train/test split is a single cut at 70% of the dataset in its original order, with no shuffling, stratification or cross-validation.
- All four classifiers use default hyperparameters. The saved output shows logistic regression stopping at scikit-learn's default iteration limit with a convergence warning. In the check above, on scikit-learn 1.9.1, it converged in 70 iterations, under the limit of 100.
- The random forest and the MLP are created without a `random_state`, so their numbers differ from one run to the next, as they did in the check above.
- `sen_2_embed` passes the model a second tensor filled with the message's position in its list (1, 2, 3 and so on), where the tutorial it follows passes all ones. `RobertaModel` takes its second positional argument as the attention mask. In `transformers` 4.18.0 that adds a constant, larger for each later position, to every attention score. The constant cancels in the softmax in exact arithmetic, but in single-precision floating point it rounds the scores more coarsely the later a message comes in its list. `transformers` 5.19.0 converts the mask to booleans instead, so there the position has no effect: in the check above, one message embedded as position 1 and as position 3902 gave the same vector. The size of the effect on the saved results was not measured. The runs on 5.19.0 differ from the 2022 run in more than this, so their numbers do not measure it either.

## Credits

- The embedding code follows the [BERT Word Embeddings Tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/) by Chris McCormick and Nick Ryan (2019, May 14). Most of the lines of `sen_2_embed` and its variable names come from that tutorial, as does the `from_pretrained` call, with `BertModel` replaced by `RobertaModel`.
- The data is the SMS Spam Collection, described in "Contributions to the study of SMS Spam Filtering: New Collection and Results" by T.A. Almeida, J.M. Gomez Hidalgo and A. Yamakami. The notebook loads it from the Hugging Face Hub as [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam), which the Hub used to list as `sms_spam`.
- The pretrained model is `roberta-base`, which the Hub now lists as [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base).
