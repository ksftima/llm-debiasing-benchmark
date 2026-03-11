from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

batch = client.batches.retrieve("batch_69aea08889148190be120bd088e9bad5")
print(f"Status: {batch.status}")
print(f"Counts: {batch.request_counts}")
print(f"output_file_id: {batch.output_file_id}")
print(f"error_file_id:  {batch.error_file_id}")

if batch.error_file_id:
    error_jsonl = client.files.content(batch.error_file_id).text
    print("\nFirst error:")
    import json
    first = json.loads(error_jsonl.strip().splitlines()[0])
    print(json.dumps(first, indent=2))
