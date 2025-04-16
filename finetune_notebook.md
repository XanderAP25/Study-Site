---
layout: default
title: Notebook
---

# [Fine-Tuning a Language Model for Story Generation in a Notebook Environment](https://colab.research.google.com/drive/1goVTnNt6FauofB_BAQ2Db4h8uWFvfY1d?usp=sharing)

Note: To load a model from Hugging Face, you will need to make a free account and set up an API key.

More information on getting your key into Colab can be found here: https://drlee.io/how-to-use-secrets-in-google-colab-for-api-key-protection-a-guide-for-openai-huggingface-and-c1ec9e1277e0

## Story Generation

### Baseline Pretrained Model Story Generation


```python
# Load dependencies
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import pandas as pd
```

#### Load the Model and Tokenizer and Set Up Generation Pipeline



```python
# Load Model & Tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16  # Efficient on A100/L4
)

# Set up generation pipeline
storyteller = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
```

#### Generate a List of Stories


```python
# Prompt (Baseline prompt)
# Since we are using an instruct model, we must give it some type of instruction
prompt = (
    "Tell an original traditional German folktale that is at least 1000 words long. "
    "The story must end with 'The End' and should not include any additional content, "
    "analysis, references, or explanations. The story must be at least 1000 words long."
)

# Set up params
stories_to_generate = 63 # Define the number of stories that you want the model to generate
stories = [] # Create a list to hold the generated responses
model_type = "baseline" # Designate the output as either baseline, ICL, or finetuned

# Generate a number of short stories, and write them to a list
for i in range(stories_to_generate):
    # Generate a story
    story = storyteller(
        (prompt),
        max_new_tokens=1800,
        min_length=900,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=2  # Ensures the output stops at a proper endpoint
    )
    story[0]['generated_text'] = story[0]['generated_text'].replace(prompt, "").strip()
    stories.append(story[0]['generated_text'])  # Append the text of the generated story
    print(f"Story {i+1} generated.")

print("***Generation Complete***")
```

#### Save the Stories to CSV File


```python
# Save stories to a csv file for analyis
def save_stories_to_csv(story_number, model_type, prompt, stories):
        d = {
            "stories_number": (i + 1 for i in range(stories_to_generate)),
            "type": ([model_type] * story_number),
            "prompt": prompt,
            "output": stories,
        }
        df = pd.DataFrame(d)

        # Set the filename depending on the changes to the prompt/model
        file_name = f"stories_{model_type}.csv"

        # Save the generated outputs to a csv file
        df.to_csv(file_name, index=False, encoding="utf-8")
        # return df

save_stories_to_csv((i + 1 for i in range(stories_to_generate)),
                    model_type, prompt, stories)
print("Stories saved to csv")
```

### Story Generation on a Pretrained Model Using ICL


```python
# Load dependencies
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import pandas as pd
import kagglehub
import os
from datasets import load_dataset
```

Here the LLM will be given some stories from Grimm's Fairy Tales as examples so that it will "learn" what a proper German fairy tale looks like. There are limits to how much context we can give the LLM before it begins to "forget", so we will keep the set of example stories to below 2900 words.

#### Load and Prepare Grimm's Dataset


```python
path = kagglehub.dataset_download("tschomacker/grimms-fairy-tales")

# List files in the directory
files = os.listdir(path)
grimms_df = pd.read_csv(os.path.join(path, files[0]))
grimms_df = grimms_df.drop(22, axis=0).reset_index(drop=True) # This row appears to be incomplete or made in error, so we will drop it

# Add word count to the dataframe as a column
grimms_df['Word_Count'] = grimms_df['Text'].apply(lambda x: len(str(x).split()))
```


```python
# Function for selecting a random set of stories for ICL
def select_stories_for_ICL(df):
    # Set parameters for selecting stories
    MAX_WORDS = 2900 # Max words for set of stories
    stories_used = [] # List of stories that are used for the prompt
    total_words = 0 # Total words for the chosen set of stories

    # Start with a random story
    first_story = df.sample(n=1).iloc[0]
    total_words = first_story["Word_Count"]
    stories_prompt = first_story["Text"] + "\n"
    stories_used.append(first_story["Title"])

    # Shuffle the stories
    shuffled_stories = df.sample(frac=1)

    # Add more stories without exceeding the max word limit
    for _, story in shuffled_stories.iterrows():
        if total_words + story["Word_Count"] <= MAX_WORDS:
            total_words += story["Word_Count"]
            stories_used.append(story["Title"])
            stories_prompt += story["Text"] + "\n===\n"
        else:
            break

    # Construct final prompt
    beg_prompt = (
        "Tell an original traditional German folktale that is at least 1000 words long. "
        "The story must end with 'The End' and should not include any additional content, "
        "analysis, references, or explanations."
    )

    end_prompt = (
        "Now, tell a single original traditional German folktale based on the structure and style above. "
        "Do not generate multiple stories. Only one story should be created, and it should follow the same "
        "narrative structure as the examples above. Conclude the story with 'The End.'"
    )

    final_prompt = f"{beg_prompt}\n\n===\n{stories_prompt}\n{end_prompt}"

    return final_prompt, stories_used
```

#### Load the Model and Tokenizer and Set Up Generation Pipeline



```python
# Load Model & Tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16  # Efficient on A100/L4
)

# Set up generation pipeline
storyteller = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
```

#### Generate a List of Stories Using ICL


```python
# Set up params
stories_to_generate = 63 # Define the number of stories that you want the model to generate
stories = [] # Create a list to hold the generated responses
model_type = "ICL" # Designate the output as either baseline, ICL, or finetuned
stories_used_composite = [] # List of stories that are used for the prompt

# Generate a number of short stories, and write them to a list
for i in range(stories_to_generate):
    ICL_prompt, stories_used = select_stories_for_ICL(grimms_df)
    # Generate a story
    story = storyteller(
        ICL_prompt,
        max_new_tokens=1800,
        min_length=900,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=2  # Ensures the output stops at a proper endpoint
    )
    story[0]['generated_text'] = story[0]['generated_text'].replace(prompt, "").strip()
    stories.append(story[0]['generated_text'])  # Append the text of the generated story
    stories_used_composite.append(stories_used)
    print(f"Story {i+1} generated.")

d = {"stories_used": stories_used_composite}
print("***Generation Complete***")
```

#### Save the Stories to CSV File


```python
# Save stories to a csv file for analyis
def save_stories_to_csv(story_number, model_type, prompt, stories_used, stories):
        d = {
            "stories_number": (i + 1 for i in range(stories_to_generate)),
            "type": ([model_type] * story_number),
            "prompt": prompt,
            "stories_used": stories_used,
            "output": stories,
        }
        df = pd.DataFrame(d)

        # Set the filename depending on the changes to the prompt/model
        file_name = f"stories_{model_type}.csv"

        # Save the generated outputs to a csv file
        df.to_csv(file_name, index=False, encoding="utf-8")
        # return df

save_stories_to_csv((i + 1 for i in range(stories_to_generate)),
                    model_type, prompt, stories_used_composite, stories)
print("Stories saved to csv")
```

### Story Generation Using a Saved Fine-Tuned Model


```python
# Load dependencies
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
import bitsandbytes
import pandas as pd
```

#### Load Finetuned Model, Tokenizer, and Set Up Pipeline


```python
# Set model path, this is how it should be if you are in Colab
model_path = "/content/Mistral Finetune"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

storyteller = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
```

#### Generate a List of Stories Using a Finetuned Model


```python
# Prompt (same as baseline prompt)
# Since we are using an instruct model, we must give it some type of instruction
prompt = (
    "Tell an original traditional German folktale that is at least 1000 words long. "
    "The story must end with 'The End' and should not include any additional content, "
    "analysis, references, or explanations. The story must be at least 1000 words long."
)

# Set up params
stories_to_generate = 63 # Define the number of stories that you want the model to generate
stories = [] # Create a list to hold the generated responses
model_type = "finetune" # Designate the output as either baseline, ICL, or finetuned

# Generate a number of short stories, and write them to a list
for i in range(stories_to_generate):
    # Generate a story
    story = storyteller(
        prompt,
        max_new_tokens=1800,
        min_length=900,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=2  # Ensures the output stops at a proper endpoint
    )
    story[0]['generated_text'] = story[0]['generated_text'].replace(prompt, "").strip()
    stories.append(story[0]['generated_text'])  # Append the text of the generated story
    print(f"Story {i+1} generated.")

print("***Generation Complete***")
```

#### Save Finetuned Stories to CSV


```python
# Save stories to a csv file for analyis
def save_stories_to_csv(story_number, model_type, prompt, stories):
        d = {
            "stories_number": (i + 1 for i in range(stories_to_generate)),
            "type": ([model_type] * story_number),
            "prompt": prompt,
            "output": stories,
        }
        df = pd.DataFrame(d)

        # Set the filename depending on the changes to the prompt/model
        file_name = f"stories_{model_type}.csv"

        # Save the generated outputs to a csv file
        df.to_csv(file_name, index=False, encoding="utf-8")
        # return df

save_stories_to_csv((i + 1 for i in range(stories_to_generate)),
                    model_type, prompt, stories)
print("Stories saved to csv")
```

## Finetuning a Model Using Unsloth


```python
# Install dependencies
import os

if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
    !pip install --no-deps unsloth
```


```python
# Load dependencies
from unsloth import FastLanguageModel
from transformers import TrainingArguments, TextStreamer, AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from datasets import load_dataset
import kagglehub
import pandas as pd
import os
import torch
import json
```

#### Load Pretrained Model for Finetuning


```python
# Load your model of choice
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Initialize model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length=4096,  # Make sure this matches your input length requirements
    dtype=torch.float16,  # Use float16 for efficiency
    load_in_4bit=True,  # Ensure LoRA is being used
)

EOS_TOKEN = tokenizer.eos_token  # Ensure you have the tokenizer before initilization

# Set up LoRA with PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Rank for LoRA (increase for more parameters)
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.05,  # Regularization
    bias="none",  # No bias training
    use_gradient_checkpointing=True,  # Memory efficiency
    random_state=42,
    use_rslora=False,  # Standard LoRA
)
```

#### Load Training Data


```python
path = kagglehub.dataset_download("tschomacker/grimms-fairy-tales")

# List files in the directory
files = os.listdir(path)
df = pd.read_csv(os.path.join(path, files[0]))
df = df.drop(22, axis=0).reset_index(drop=True)

# Convert to JSONL format for fine-tuning
jsonl_data = [
    {"instruction": "Tell me a German fairy tale.", "input": "", "output": row["Text"]}
    for _, row in df.iterrows()
]

# Save the JSON file
jsonl_path = "fairy_tales.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Dataset saved to {jsonl_path}")

# Assuming you already loaded your dataset as a JSON object:
dataset = load_dataset("json", data_files="fairy_tales.jsonl", split="train[:2500]")
```

#### Finetune the Pretrained Model


```python
def formatting_prompts_func(examples):
    formatted_texts = []

    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        prompt = f"{instruction} {input_text}".strip()
        completion = f"{output}{EOS_TOKEN}"  # Ensure the output has an EOS token

        formatted_texts.append(f"{prompt}\n{completion}")  # Instruction + Input + Output

    return {"text": formatted_texts}

# Apply the formatting function to your dataset
dataset = dataset.map(formatting_prompts_func, batched=True)

# Now proceed with the SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",  # Adjust field name based on your data
    max_seq_length=4096,
    dataset_num_proc=2,
    packing=False,  # Adjust based on sequence lengths
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=7,
        learning_rate=2e-4,
        fp16=is_bfloat16_supported(),
        bf16=not is_bfloat16_supported(),
        logging_steps=816,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="/content",
    ),
)

trainer.train()
```

#### Generate Output With the New Model


```python
streamer = TextStreamer(tokenizer)

prompt = (
    "Tell an original traditional German folktale. "
    "The story must end with 'The End' and should not include any additional content, "
    "analysis, references, or explanations."
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

model.generate(
    input_ids=inputs["input_ids"],
    max_length=500,
    streamer=streamer,
    temperature=0.7,
    top_k=50,
    top_p=0.9,         # Nucleus sampling (0.9 keeps only 90% most probable words)
    repetition_penalty=1.2,
    do_sample=True,
)
```

#### Save the Finetuned Model and Tokenizer


```python
# Save fine-tuned model
model.save_pretrained("mistral_finetuned")
tokenizer.save_pretrained("mistral_finetuned")
```
