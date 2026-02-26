from pathlib import Path
from openai import OpenAI
import time

annotation_dir = Path("results/annotations/amazon/openai")

with open(".api_key_openai", "r") as file:
    api_key = file.read().strip()

with open(annotation_dir / Path("batch_id"), "r") as file:
    batch_id = file.read().strip()
print(f"Batch ID: {batch_id}")

client = OpenAI(api_key = api_key)

print("Updates:")
while True:
    batch = client.batches.retrieve(batch_id)
    if batch.status == "completed":
        break
    else:
        print(f" ... {batch.status}")
        time.sleep(5)

print(f"\nSaving the results to {annotation_dir}")
output_file = annotation_dir / Path("output_file.jsonl")
with open(output_file, "w") as file:
    responses = client.files.content(batch.output_file_id).text
    file.write(responses)

