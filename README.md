# 👩‍💻 Embeddings Techniques in NLP and the LLM ecosystem

This repository will contain several notebooks and python scripts to work with embeddings, I will collect them periodically and uploading to this repo.

## Content

- Transforming LLMs into High-Quality Text Embeddings with LLM2Vec: `Qwen2-05B-LLM2Vec.ipynb`

## Training and Evaluation with LLM2Vec

LLM2Vec is a simple recipe to convert decoder-only LLMs into text encoders. It consists of 3 simple steps: 1) enabling bidirectional attention, 2) training with masked next token prediction, and 3) unsupervised contrastive learning. The model can be further fine-tuned to achieve state-of-the-art performance. With LLM2Vec, we can extract an inaccurate embedding model directly from the LLM. Then, we can improve this model with a two-stage training including masked next-token prediction and contrastive learning.

Read the great article ["Llama 3.2 Embeddings: Training and Evaluation with LLM2Vec"](https://kaitchup.substack.com/p/llama-32-embeddings-training) by Benjamin Marie on his "The Kaitchup – AI on a Budget" Substack, for a full detailed analysis and description of this technique.

### Problem description

In the [notebook](./Qwen05B-LLM2Vec.ipynb), we will see how to make text embeddings from Qwen2 0.5 B. We will see in detail all the steps: masked next-token prediction training, contrastive learning, and then how to evaluate the resulting embeddings.
You can find the base model on Huggingface, [Qwen2 0.5B Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)

To train and evaluate the embedding model, I used an RTX 3090 from Vast.ai.

### Example of usage

```py
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

model_id = "edumunozsala/phi3-mini-4k-qlora-python-code-20k"
device_map="cuda"

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct", padding_side='left' 
)
config = AutoConfig.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)

# Loading MNTP (Masked Next Token Prediction) model.
model = PeftModel.from_pretrained(
    model,
    "edumunozsala/Qwen2-0.5B-mntp-simcse",
)

# Wrapper for encoding and pooling operations
l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
# Encoding queries using instructions
instruction = (
    "Given a web search query, retrieve relevant passages that answer the query:"
)
queries = [
    [instruction, "how much protein should a female eat"],
    [instruction, "summit define"],
]
q_reps = l2v.encode(queries)

# Encoding documents. Instruction are not required for documents
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
]
d_reps = l2v.encode(documents)

# Compute cosine similarity
q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

print(cos_sim)

```

## Contributing
If you find some bug or typo, please let me know or fixit and push it to be analyzed. 

## License

Copyright 2023 Eduardo Muñoz

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
