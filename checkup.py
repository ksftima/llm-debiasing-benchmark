  
from openai import OpenAI
import os                                                                                                                               
from dotenv import load_dotenv                                                                                                        
load_dotenv()
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
batch_id = open('thesis/annotations/pubmedqa/openai/batch_id.txt').read().strip()
batch = client.batches.retrieve(batch_id)
print(batch.status)
print(batch.request_counts)