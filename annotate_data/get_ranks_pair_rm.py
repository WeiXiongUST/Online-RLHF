import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, AutoModelForCausalLM
from accelerate import Accelerator

tqdm.pandas()

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    dataset_name_or_path: Optional[str] = field(
        default="uf_split0_responses_K8.json",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="data_with_ranks.json",
        metadata={"help": "the location of the output file"},
    )
    # No need to set record_dir
    record_dir: Optional[str] = field(
        default=None, 
        metadata={"help": "the location of the recording file"},
    )
    reward_name_or_path: Optional[str] = field(
        default="RLHFlow/pair-preference-model-LLaMA3-8B",
        metadata={"help": "the name of the reward model"},
    )
    input_output_delimiter: Optional[str] = field(
        default="",
        metadata={"help": "the delimiter between input and output"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of responses per prompt"},
    )


accelerator = Accelerator()

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

device = accelerator.device

model = AutoModelForCausalLM.from_pretrained(script_args.reward_name_or_path,
                                             torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(script_args.reward_name_or_path, use_fast=True)
tokenizer_plain = AutoTokenizer.from_pretrained(script_args.reward_name_or_path, use_fast=True)
tokenizer_plain.chat_template = "\n{% for message in messages %}{% if loop.index0 % 2 == 0 %}\n\n<turn> user\n {{ message['content'] }}{% else %}\n\n<turn> assistant\n {{ message['content'] }}{% endif %}{% endfor %}\n\n\n"


prompt_template = "[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"
token_id_A = tokenizer.encode("A", add_special_tokens=False)
token_id_B = tokenizer.encode("B", add_special_tokens=False)
assert len(token_id_A) == 1 and len(token_id_B) == 1
token_id_A = token_id_A[0]
token_id_B = token_id_B[0]

ds_dir = script_args.dataset_name_or_path
world_size = int(os.getenv("WORLD_SIZE", "1"))

####### The dataset is supposed to contain {"prompt": prompt, "responses": [response1, response2, ..., responsek]}
# No additional format! Just raw prompt and response.
ds = load_dataset("json", data_files=ds_dir, split="train", field="instances")

local_rank = Accelerator().local_process_index

data_size = len(ds["prompt"])

share = int(data_size / world_size) + 1
ds = ds.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(ds))))
temperature=1.0


def comp(context, responses):
    """
    This function takes a context and two responses as the input. 
    The two responses will be ranked by the pairwise preference model.
    Return 0 if the first one is better, 1 if the second one is better, and 0.5 if tie.
    """
    # probability of chossing first sentence
    probs_chosen = []
    for chosen_position in [0, 1]:
        # swap order to mitigate position bias
        response_A = responses[chosen_position]
        response_B = responses[1 - chosen_position]
        prompt = prompt_template.format(context=context, response_A=response_A, response_B=response_B)
        message = [
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model(input_ids)
        logit_A = output.logits[0, -1, token_id_A].item()
        logit_B = output.logits[0, -1, token_id_B].item()
        # take softmax to get the probability; using numpy
        Z = np.exp(logit_A / temperature) + np.exp(logit_B / temperature)
        logit_chosen = [logit_A, logit_B][chosen_position]
        prob_chosen = np.exp(logit_chosen / temperature) / Z
        probs_chosen.append(prob_chosen)
    avg_prob_chosen = np.mean(probs_chosen)
    
    if avg_prob_chosen > 0.5:
        return 0
    elif avg_prob_chosen == 0.5:
        return 0.5
    else:
        return 1
    

def full_pairwise_ranking(prom, test_texts):
    # Initialize score dictionary
    scores = {txt: 0 for txt in test_texts}
    
    # Compare every pair of items so can be slow...
    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            result = comp(prom, [test_texts[i], test_texts[j]])
            if result == 0: # the first response is better
                scores[test_texts[i]] += 1
            elif result == 0.5:
                scores[test_texts[i]] += 0.5
                scores[test_texts[j]] += 0.5
            else:
                scores[test_texts[j]] += 1

    return scores


data = []


# tqdm is used to show the progress bar
with torch.no_grad():
    for sample in tqdm(ds):
        # The VLLM may not generate responses for some prompts because it is too long, we skip them
        if len(sample["responses"]) < script_args.K:
            continue
        test_prompt = tokenizer_plain.apply_chat_template(sample['prompt'], tokenize=False)
        test_texts = sample['responses']

        scores = full_pairwise_ranking(test_prompt, test_texts)
        rewards = [scores[txt] for txt in test_texts]
        print(test_prompt)
        data.append({"prompt": sample['prompt'], "responses": sample['responses'], "rewards": rewards})


# Send the data to other GPUs
world_size = int(os.getenv("WORLD_SIZE", "1"))
all_process_list = [{}] * world_size

data_to_send = {
    "data": [[data[i]] for i in range(len(data))],
}

import torch.distributed as dist

dist.all_gather_object(all_process_list, data_to_send)
gathered_data = []


for i in range(world_size):
    tmp_data = [tmp[0] for tmp in all_process_list[i]["data"]]
    gathered_data.extend(tmp_data)


if local_rank == 0:
    output_eval_dataset = {}
    output_eval_dataset["type"] = "text_only"
    output_eval_dataset["instances"] = gathered_data
    with open(script_args.output_dir, "w", encoding="utf8") as f:
        json.dump(output_eval_dataset, f, ensure_ascii=False)

