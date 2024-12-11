The work in this project was done as part of the MNLP course at EPFL by: Stefan Krsteski, Mikhail Terekhov, Said Gürbüz, Matea Tashkovska 

## Project Description 

- In this project, we developed large language models (LLMs) capable of communicating with humans and assisting with various tasks. Exploring various frontier methods to apply LLMs in real-world applications, including Direct Preference Optimization (DPO), Retrieval-Augmented Generation (RAG), and model quantization.

## Abstract
The increase in educational attainment worldwide comes with demands for novel tools to be used by students and teachers. Language models provide a great opportunity in this respect, but the technology must be handled carefully. To facilitate the adoption of language models in this key domain, we design multiple extensions of a model from the  Phi-3 family, originally developed by Microsoft. We call the collection of our extensions $\varphi$-3PO. We fine-tune Phi-3 using DPO on a carefully curated list of datasets, including  data from the students taking the MNLP course. We apply two techniques to further fine-tune the model to improve its performance on multiple-choice questions, including Chain-of-Thought prompting and Supervised Fine-Tuning. We compress the model using the GPTQ quantization technique. The 8-bit version of the  model retains the original performance while halving the size. We also design a RAG system by gathering a collection of STEM-related factual data and employing a state-of-the-art embedding model. We implement two systems for embedding lookup, using naive search and a FAISS index. RAG is shown to be effective on knowledge domains which were included in the pre-selected data. Overall, our models consistently perform well on the tasks they were designed for.

All the models can be found here: https://huggingface.co/StefanKrsteski

## Codebase File Structure

```txt
model
├── checkpoints
│   ├── your_checkpoint_name
│   │   ├── config.json
│   │   |── model.safetensor
│   │   └── ...
├── datasets
│   │   ├── your_dataset_name
│   │   │   └── ...
│   │   └── ...
├── documents (For RAG only)
├── models
│       ├── model_base.py
│       └── model_dpo.py
├── utils.py
├── evaluator.py
├── main_config.yaml
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup

### Setup via Conda Virtual Environment

```bash
# Replace <my-env> with the name of your environment.
conda create --name <my-env> python=3.10.11
conda activate <my-env>

# Install dependencies from a `requirements.txt`
pip install -r requirements.txt
# If you intend to use flash-attention for more efficient training and inference
pip install flash-attn --no-build-isolation
```

### Setup via Docker Container

```bash
# Replace <my-docker> with the name of your docker image.
docker build -f Dockerfile . -t <my-docker>
docker run <my-docker>
docker exec -it <my-docker> bash
# Continue any bash operations ...

# Replace <num-gpu> with the number of GPUs you have
sudo docker run --gpus <num-gpu> -it -d  \
    --name $NAME \
    --rm --shm-size=128gb \
    --network host \
    -v /pure-mlo-scratch:/pure-mlo-scratch \
    -v /home:/home meditron \
    -- /bin/bash -c 'bash'
```

## Codebase Introduction (For Testing Only)

- `model_base.py`: In this file, you will find a wrapper model class `PreTrainedModelWrapper` around a (`transformers.PreTrainedModel`) to be compatible with the (`~transformers.PreTrained`) class in order to keep some attributes and methods of the (`~transformers.PreTrainedModel`) class. You can save a checkpoint through `PreTrainedModelWrapper.save_pretrained(model_name_or_path)` and load a checkpoint locally or from HuggingFace hub through the method `PreTrainedModelWrapper.from_pretrained(model_name_or_path)`.
- `model_dpo.py`: In this file, you the DPO model is implemented. If you are working with a sequence-to-sequence language model like T5 or Bart, use the `AutoDPOModelForSeq2SeqLM` class. The functions that are required for all to implement including `forward`, `prediction_step_reward`, and `prediction_step_mcqa`.
- In addition to a transformer model, you can add custom modules to the `AutoDPOModel` classes. Below is an example custom module. 

### Basic Model functionalities

For `AutoDPOModelForCausalLM` and `AutoDPOModelForSeq2SeqLM`, which both inherit `PreTrainedModelWrapper`, you have the following basic operations:

**Load from a pre-trained model listed in [Huggingface Hub](https://huggingface.co/models)**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.model_dpo import AutoDPOModelForCausalLM

# Download the pre-trained model and tokenizer from the Hub
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize your model class and import the pre-trained model into your class
# Note that if you have a custom module in your class
# You should initialize the weights of this module in the `__init__` function
model_wrapper = AutoDPOModelForCausalLM(pretrained_model=model)
```

**Save your model as a Huggingface transformers compatible checkpoint**

```python
# Save your model and tokenizer to the checkpoint directory `models/dpo_gpt2`
checkpoint_dir = "models/dpo_gpt2"
model_wrapper.save_pretrained(checkpoint_dir)
tokenizer.save_pretrained(checkpoint_dir)
```

**Load from your local checkpoint**

```python
checkpoint_dir = "models/dpo_gpt2"
model_wrapper = AutoDPOModelForCausalLM.from_pretrained(checkpoint_dir)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
```

### Custom Module Example

```python
class CustomModule(nn.Module):
    """
    This is only a dummy example of a custom module. You can replace this with your own custom module.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        self.summary = nn.Linear(hidden_size, 1)
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        output = self.summary(output)
        return output
```

- `evaluator.py` is the main evaluation script. To execute this script, you first need to specify details in the `main_confi.yaml` configuration file. The details in this config file will be used by the evaluation script to execute the grading properly. Make sure you fill all the important information in the config file.

### Main Configuration Arguments

```yaml
"team_name": "Team 1" # Your team name
"eval_method": ["mcqa", "rag"] # Tells the evaluator which evaluations need to be executed. choices = [mcqa, reward, rag, quantiz]
"task_type": "causal_lm" # Identifies which model class you use. choices = [causal_lm, seq2seq]
"policy_model_path": "./checkpoints/best_model/" # Your path to the final checkpoint
"reference_model_path": "microsoft/phi-2" # The repo id of your pretrained DPO reference model
"quantized_policy_model_path": "./checkpoints/best_model_quantized/" # Your path to the final quantized checkpoint
"rag_policy_model_path": "./checkpoints/best_model_rag/" # Your path to the final RAG checkpoints
"test_data_path": "./data/test.json" # Your path to the test data. (We will replace it with the official test sets when grading)
"dpo_model_args": null # Any required arguments to load your dpo model using "from_pretrained"
"rag_model_args": # Any required arguments to load your rag model using "from_pretrained" For example:
    "encoder_model_path": "facebook/bart-large"
    "retriever_model_path": "./checkpoints/rag_retriever"
    "document_dir": "./data/documents"
"quantized_model_args": null # Any required arguments to load your quantized model using "from_pretrained"
```

- Note: `eval_method`'s value must be a list object.

- Note: `reward` and `mcqa` cannot co-exist in the `eval_method` list at the same time.

Please review the evaluation script code for detailed evaluation methods and the input and output of each evaluation function.
