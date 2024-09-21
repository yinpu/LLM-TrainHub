This provides a method to generate text embeddings using an LLM. The embeddings are trained with specific datasets and can be fine-tuned or augmented with an additional fully connected layer.


## Data Format

The training data is organized as JSON lines. Each line contains a query, positive samples (`pos`), and negative samples (`neg`) for contrastive learning. Here is an example of the data structure:

```json
{
  "query": "A girl with a blue tank top sitting watching three dogs.",
  "pos": ["A girl is wearing blue."],
  "neg": [
    "A girl is with three cats.",
    "The people are watching a funeral procession.",
    "The child is wearing black.",
    "Financing is an issue for us in public schools.",
    "Kids at a pool.",
    "It is calming to be assaulted.",
    "I face a serious problem at eighteen years old."
  ]
}
```

## Usage
To begin training the model with the provided data, execute the following command:

```bash
bash run_train.sh
```

Once the training is completed, you can perform inference using:

```bash
bash run_infer.sh
```

## Training Methods
There are two main methods for training the LLM:

1. Fine-tuning the LLM parameters:
In this approach, all the LLM’s parameters are updated during training to generate better text embeddings.
2. Freezing the LLM parameters and adding a fully connected layer:
Here, the LLM’s parameters remain fixed, and a new fully connected layer is trained on top of the embeddings produced by the LLM.

Choose the method that best suits your use case, depending on whether you want to adjust the underlying LLM or keep it static and focus on higher-level learning.