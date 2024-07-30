# BART Fine-Tuned for Question Answering Model

## Introduction

Welcome to my project on developing a state-of-the-art question-answering model using the [Quora Question Answer Dataset](https://huggingface.co/datasets/toughdata/quora-question-answer-dataset). The goal of this project is to create an AI system that can understand and generate accurate responses to a wide range of user queries, effectively mimicking human-like interactions.

To achieve this, I fine-tuned the BART model on the Quora dataset, leveraging its advanced capabilities in natural language understanding and generation. This approach allows the model to handle diverse questions and provide relevant answers with high accuracy.

## Dataset

The model was trained on the [Quora Question Answer Dataset](https://huggingface.co/datasets/toughdata/quora-question-answer-dataset) available on Hugging Face. This dataset comprises a diverse set of questions and answers, containing  56,402 such pairs scraped from Quora.

## Details

### Pre-trained Model
[bart-base](https://huggingface.co/facebook/bart-base)


### Evaluation Metric
ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

## Training
- **Training Framework**: Hugging Face's Transformers library
- **Hyperparameters**:
  - Learning rate: 3e-4
  - Batch size: 8 (training), 4 (evaluation)
  - Weight decay: 0.01
  - Number of epochs: 2
  - Gradient accumulation steps: 4
  - Mixed precision training (fp16)

## Tech Stack
- **Programming Languages**: Python
- **Frameworks and Libraries**:
  - Transformers: `transformers`, `tokenizers`, `datasets`, `evaluate`
  - Data Processing: `pandas`, `numpy`
  - Visualization: `seaborn`, `matplotlib`
  - Others: `nltk`, `rouge_score`, `huggingface_hub`


## Installation
To set up the environment, you can use the `requirements.txt` file provided. Here are the steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/sag-27/BART_fine-tuned_H2H2024.git
    ```
    ```sh
    cd BART_fine-tuned_H2H2024/
    ```

3. **Install the required packages**:
   
     To install the dependencies,

    1. Create a Python virtual environment with Python version 3.10.13 and activate it
    2. Install dependency libraries using the following command in the terminal.
    ```sh
    pip install -r requirements.txt
    ```

## Report

The link to the detailed report can be found
