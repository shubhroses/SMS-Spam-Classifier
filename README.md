# SMS Spam Classifier

A Jupyter notebook that classifies SMS messages as spam or ham (not spam). It turns each message into a sentence embedding with a pretrained RoBERTa model, trains four scikit-learn classifiers on those embeddings, and reports accuracy, precision, recall and F1 for each.

The notebook is `sms_spam_classifier.ipynb`. It was run in Google Colab, and the outputs saved in it are from that run. It was first committed as `5170_Final.ipynb`.

## Approach

1. **Embeddings.** The notebook loads the `roberta-base` tokenizer and model from Hugging Face `transformers`, the model with `output_hidden_states=True`. Each message is tokenized without RoBERTa's start and end tokens and run through the model in evaluation mode with gradients disabled. The notebook takes the second-to-last hidden layer and averages it over the tokens, which gives one vector of 768 values per message. RoBERTa is used only as a fixed feature extractor; it is not fine-tuned.
2. **Data.** The `ucirvine/sms_spam` dataset is loaded with Hugging Face `datasets`: 5574 messages labeled 1 for spam and 0 for ham. The first 70% of the messages, in their original order, are the training set and the rest are the test set.
3. **Classifiers.** Gaussian Naive Bayes, a random forest, logistic regression and a multi-layer perceptron (`MLPClassifier`) are fitted on the training embeddings, each with scikit-learn's default settings.
4. **Evaluation.** Each model is scored on the test set for accuracy, and for precision, recall and F1 on the spam class.

## Results

These numbers are copied from the outputs saved in the notebook, written as the notebook printed them (it rounds to three decimal places).

| Classifier | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
| --- | --- | --- | --- | --- |
| Gaussian Naive Bayes | 0.987 | 0.99 | 0.912 | 0.95 |
| Random forest | 0.983 | 1.0 | 0.877 | 0.935 |
| Logistic regression | 0.993 | 0.995 | 0.956 | 0.975 |
| MLP | 0.995 | 1.0 | 0.965 | 0.982 |

The notebook also scores the four models on a list of 15 messages written into one of its cells. The accuracies it printed are 1.0 for logistic regression and for the MLP, 0.8666666666666667 for Gaussian Naive Bayes and 0.7333333333333333 for the random forest. Most of those messages are copies or near copies of dataset rows that fall in the training set, so this is a sanity check rather than an independent test.

The last cell is interactive: choose one of the four classifiers, type a message, and it prints `Spam` or `Ham`.

## Running the notebook

The saved outputs come from Python 3.7 with `transformers` 4.18.0 and `datasets` 2.2.1. The notebook has since been run on current versions of these libraries, and `requirements.txt` pins the versions used.

To run it in Jupyter, with Python 3.12 or newer:

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
- The first run downloads the `roberta-base` weights (478M in the saved output) and the dataset, so it needs internet access.
- The model runs on the CPU, one message at a time, with no batching.

## Limitations

- The train/test split is a single cut at 70% of the dataset in its original order, with no shuffling, stratification or cross-validation.
- All four classifiers use default hyperparameters. The saved output shows logistic regression stopping at scikit-learn's default iteration limit with a convergence warning.
- The random forest and the MLP are created without a `random_state`, so their numbers can differ from one run to the next.
- `sen_2_embed` passes the model a second tensor filled with the message's position in its list (1, 2, 3 and so on), where the tutorial it follows passes all ones. `RobertaModel` takes its second positional argument as the attention mask. In `transformers` 4.18.0 that adds a constant, larger for each later position, to every attention score. The constant cancels in the softmax in exact arithmetic, but in single-precision floating point it rounds the scores more coarsely the later a message comes in its list. `transformers` 5.19.0 converts the mask to booleans instead, so there the position has no effect. The size of the effect on the saved results was not measured, and a run on a current version should not be expected to reproduce them exactly.

## Credits

- The embedding code follows the [BERT Word Embeddings Tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/) by Chris McCormick and Nick Ryan (2019, May 14). Most of the lines of `sen_2_embed` and its variable names come from that tutorial, as does the `from_pretrained` call, with `BertModel` replaced by `RobertaModel`.
- The data is the SMS Spam Collection, described in "Contributions to the study of SMS Spam Filtering: New Collection and Results" by T.A. Almeida, J.M. Gomez Hidalgo and A. Yamakami. The notebook loads it from the Hugging Face Hub as [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam), which the Hub used to list as `sms_spam`.
- The pretrained model is `roberta-base`, which the Hub now lists as [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base).
