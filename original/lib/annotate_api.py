import anthropic 
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from openai import OpenAI
from pathlib import Path
import pandas as pd 
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os
from itertools import batched
import time
import json

from annotate_prompts import dataset_labels, system_prompts, make_user_prompt

def save_prompt(annotation_dir, prompt, index):
    prompt_path = annotation_dir / Path("prompts")
    os.makedirs(prompt_path, exist_ok = True)
    prompt_file = prompt_path / Path(f"prompt_{index:05}.txt")
    with open(prompt_file, "w", encoding="utf-8") as file:
        file.write(prompt)

def save_response(annotation_dir, response, index):
    response_path = annotation_dir / Path("responses")
    os.makedirs(response_path, exist_ok = True)
    response_file = response_path / Path(f"response_{index:05}.txt")
    with open(response_file, "w", encoding="utf-8") as file:
        file.write(response)

def save_error(annotation_dir, error, index):
    error_path = annotation_dir / Path("errors")
    os.makedirs(error_path, exist_ok = True)
    response_file = error_path / Path(f"error_{index:05}.txt")
    with open(response_file, "w", encoding="utf-8") as file:
        file.write(error)

##########
# Claude #
##########

def annotate_claude(system_prompt, user_prompts, annotation_dir):

    with open(".api_key_anthropic", "r") as file:
        api_key = file.read().strip()

    client = anthropic.Anthropic(
        api_key = api_key
    )

    def make_request(custom_id, prompt):
        return Request(
            custom_id = str(custom_id),
            params = MessageCreateParamsNonStreaming(
                model = "claude-3-7-sonnet-20250219",
                max_tokens = 100,
                system = system_prompt,
                messages = [{"role": "user", "content": prompt, }],
            )
        )

    requests = [make_request(k, v) for k, v in user_prompts.items()]

    print(f"\n\nCreating a batch request with {len(requests)} messages")
    batch = client.messages.batches.create(requests = requests)
    print(f"Batch ID: {batch.id}")

    with open(annotation_dir / Path("batch_id"), "w") as file:
        file.write(batch.id)

    print("Updates:")
    while True:
        message_batch = client.messages.batches.retrieve(batch.id)
        if message_batch.processing_status == "ended":
            break
        else:
            print(f" ... {message_batch.processing_status}")
            time.sleep(60)

    print(f"\nSaving the results to {annotation_dir}")
    for result in client.messages.batches.results(batch.id):
        index = int(result.custom_id)
        if result.result.type == "succeeded":
            save_prompt(annotation_dir, user_prompts[index], index)
            response = result.result.message.content[0].text
            save_response(annotation_dir, response, index)
        elif result.result.type == "errored":
            save_error(annotation_dir, str(result.result.error), index)
        else:
            print("Response was cancelled or ran out of time")

############
# DeepSeek #
############

def call_deepseek(system_prompt, user_prompt):

    with open(".api_key_deepseek", "r") as file:
        api_key = file.read().strip()

    client = OpenAI(
        api_key = api_key,
        base_url = "https://api.deepseek.com",
    )

    api_response = client.chat.completions.create(
        model = "deepseek-chat",
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt },
        ],
        stream = False
    )
    response = api_response.choices[0].message.content

    return user_prompt, response

def annotate_deepseek(system_prompt, user_prompts, annotation_dir):

    responses = {}

    with ThreadPoolExecutor(max_workers = len(user_prompts)) as executor:

        futures = {}

        for index, user_prompt in user_prompts.items():
            future = executor.submit(call_deepseek, system_prompt, user_prompt)
            futures[future] = index

        for future in as_completed(futures):
            index = futures[future]
            try:
                user_prompt, response = future.result()
                responses[index] = response
                save_prompt(annotation_dir, user_prompt, index)
                save_response(annotation_dir, response, index)
            except Exception as exception:
                save_error(annotation_dir, str(exception), index)
                print(f"Prompt {index:05} failed with exception: {exception}") 

    return responses

##########
# OpenAI #
##########

def annotate_openai(system_prompt, user_prompts, annotation_dir):

    with open(".api_key_openai", "r") as file:
        api_key = file.read().strip()

    client = OpenAI(api_key = api_key)

    def make_request(custom_id, prompt):
        return {
            "custom_id": str(custom_id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": system_prompt },
                    {"role": "user", "content": prompt },
                ],
            }
        }

    requests = [make_request(k, v) for k, v in user_prompts.items()]

    input_file = annotation_dir / Path("input_file.jsonl")
    with open(input_file, "w") as file:
        for request in requests:
            file.write(json.dumps(request) + "\n")

    batch_file = client.files.create(
        file = open(input_file, "rb"),
        purpose = "batch",
    )

    batch = client.batches.create(
        input_file_id = batch_file.id,
        endpoint = "/v1/chat/completions",
        completion_window = "24h",
    )
    print(f"Batch ID: {batch.id}")

    with open(annotation_dir / Path("batch_id"), "w") as file:
        file.write(batch.id)

    print("Updates:")
    while True:
        batch = client.batches.retrieve(batch.id)
        if batch.status == "completed":
            break
        else:
            print(f" ... {batch.status}")
            time.sleep(60)

    print(f"\nSaving the results to {annotation_dir}")
    output_file = annotation_dir / Path("output_file.jsonl")
    with open(output_file, "w") as file:
        responses = client.files.content(batch.output_file_id).text
        file.write(responses)

########
# Main #
########

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('api', choices = ["claude", "deepseek", "openai"])
    parser.add_argument('dataset', choices = system_prompts)
    parser.add_argument('parsed_path', type = Path)
    parser.add_argument('annotation_dir', type = Path)
    parser.add_argument('--num', type = int, default = 5)
    parser.add_argument('--start', type = int, default = 0)
    parser.add_argument('--num_examples', type = int, default = 0)
    args = parser.parse_args()

    data = pd.read_json(args.parsed_path)
    data = data[args.start:args.start + args.num]
    data = data.reset_index(drop = True)

    examples = data.sample(min(args.num, args.num_examples))
    data = data.loc[~data.index.isin(examples.index)]
    labels = examples["y"].apply(lambda i: dataset_labels[args.dataset][i])
    examples = list(zip(examples["text"], labels))

    data["prompt"] = data["text"].apply(lambda x: make_user_prompt(args.dataset, x, examples))

    print(data)

    system_prompt = system_prompts[args.dataset]
    user_prompts = {i: data["prompt"][i] for i in data.index}

    if args.api == "claude":
        annotate_claude(system_prompt, user_prompts, args.annotation_dir)

    elif args.api == "deepseek":
        start = time.time()
        for batch in batched(data.index, n = 200):
            batch = list(batch)
            print(f"Sending batch {batch[0]:05} - {batch[-1]:05}")
            user_prompts = {k: v for k, v in user_prompts.items() if k in batch}
            annotate_deepseek(system_prompt, user_prompts, args.annotation_dir)

    elif args.api == "openai":
        annotate_openai(system_prompt, user_prompts, args.annotation_dir)
