'''
https://modal.com/docs/examples/vllm_inference

https://github.com/vllm-project/vllm/blob/1f24755bf802a2061bd46f3dd1191b7898f13f45/vllm/entrypoints/llm.py#L13

'''


import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
os.environ["TRANSFORMERS_CACHE"] = "/data/hugging_face_model/"
os.environ["TORCH_HOME"] = "/data/hugging_face_model/"
os.environ["HUGGINGFACE_TOKEN"] = "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"
os.environ["HF_HOME"] = "/data/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/.cache/huggingface/hub"

from vllm import LLM, SamplingParams

'''
mkdir /data/.cache/
mkdir /data/.cache/huggingface/
mkdir /data/.cache/huggingface/hub/
chmod 777 /data/.cache/huggingface/hub

ls -l /data/.cache/huggingface/hub

rm -rf /root/.cache/huggingface
'''


llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1",)

'''
#llm = LLM(model="facebook/opt-125m")

max_seq_len = 1024,
trust_remote_code = True,
quantization = "awq"

>>> llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1")
INFO 11-06 07:18:53 llm_engine.py:72] Initializing an LLM engine with config: model='mistralai/Mistral-7B-Instruct-v0.1', tokenizer='mistralai/Mistral-7B-Instruct-v0.1', tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=auto, tensor_parallel_size=1, quantization=None, seed=0)

/data/anaconda3_1/envs/jim_python_3_8/lib/python3.8/site-packages/huggingface_hub/file_download.py:979: UserWarning: Not enough free disk space to download the file. The expected file size is: 9943.03 MB. The target location /root/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/blobs only has 4824.85 MB free disk space
'''

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.

questions = [
# Coding questions
"Implement a Python function to compute the Fibonacci numbers.",
"Write a Rust function that performs binary exponentiation.",
"How do I allocate memory in C?",
"What are the differences between Javascript and Python?",
"How do I find invalid indices in Postgres?",
"How can you implement a LRU (Least Recently Used) cache in Python?",
"What approach would you use to detect and prevent race conditions in a multithreaded application?",
"Can you explain how a decision tree algorithm works in machine learning?",
"How would you design a simple key-value store database from scratch?",
"How do you handle deadlock situations in concurrent programming?",
"What is the logic behind the A* search algorithm, and where is it used?",
"How can you design an efficient autocomplete system?",
"What approach would you take to design a secure session management system in a web application?",
"How would you handle collision in a hash table?",
"How can you implement a load balancer for a distributed system?",
# Literature
"What is the fable involving a fox and grapes?",
"Write a story in the style of James Joyce about a trip to the Australian outback in 2083, to see robots in the beautiful desert.",
"Who does Harry turn into a balloon?",
"Write a tale about a time-traveling historian who's determined to witness the most significant events in human history.",
"Describe a day in the life of a secret agent who's also a full-time parent.",
"Create a story about a detective who can communicate with animals.",
"What is the most unusual thing about living in a city floating in the clouds?",
"In a world where dreams are shared, what happens when a nightmare invades a peaceful dream?",
"Describe the adventure of a lifetime for a group of friends who found a map leading to a parallel universe.",
"Tell a story about a musician who discovers that their music has magical powers.",
"In a world where people age backwards, describe the life of a 5-year-old man.",
"Create a tale about a painter whose artwork comes to life every night.",
"What happens when a poet's verses start to predict future events?",
"Imagine a world where books can talk. How does a librarian handle them?",
"Tell a story about an astronaut who discovered a planet populated by plants.",
"Describe the journey of a letter traveling through the most sophisticated postal service ever.",
"Write a tale about a chef whose food can evoke memories from the eater's past.",
# History
"What were the major contributing factors to the fall of the Roman Empire?",
"How did the invention of the printing press revolutionize European society?",
"What are the effects of quantitative easing?",
"How did the Greek philosophers influence economic thought in the ancient world?",
"What were the economic and philosophical factors that led to the fall of the Soviet Union?",
"How did decolonization in the 20th century change the geopolitical map?",
"What was the influence of the Khmer Empire on Southeast Asia's history and culture?",
# Thoughtfulness
"Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
"In a dystopian future where water is the most valuable commodity, how would society function?",
"If a scientist discovers immortality, how could this impact society, economy, and the environment?",
"What could be the potential implications of contact with an advanced alien civilization?",
# Math
"What is the product of 9 and 8?",
"If a train travels 120 kilometers in 2 hours, what is its average speed?",
"Think through this step by step. If the sequence a_n is defined by a_1 = 3, a_2 = 5, and a_n = a_(n-1) + a_(n-2) for n > 2, find a_6.",
"Think through this step by step. Calculate the sum of an arithmetic series with first term 3, last term 35, and total terms 11.",
"Think through this step by step. What is the area of a triangle with vertices at the points (1,2), (3,-4), and (-2,5)?",
"Think through this step by step. Solve the following system of linear equations: 3x + 2y = 14, 5x - y = 15.",
# Facts
"Who was Emperor Norton I, and what was his significance in San Francisco's history?",
"What is the Voynich manuscript, and why has it perplexed scholars for centuries?",
"What was Project A119 and what were its objectives?",
"What is the 'Dyatlov Pass incident' and why does it remain a mystery?",
"What is the 'Emu War' that took place in Australia in the 1930s?",
"What is the 'Phantom Time Hypothesis' proposed by Heribert Illig?",
"Who was the 'Green Children of Woolpit' as per 12th-century English legend?",
"What are 'zombie stars' in the context of astronomy?",
"Who were the 'Dog-Headed Saint' and the 'Lion-Faced Saint' in medieval Christian traditions?",
"What is the story of the 'Globsters', unidentified organic masses washed up on the shores?",
]

# Create a sampling params object.
sampling_params = SamplingParams(
	temperature=0.75,
	top_p=1,
	max_tokens=1024,
	presence_penalty=1.15,
	)


start_time = time.time()
outputs = llm.generate(
	questions, 
	sampling_params,
	)
end_time = time.time()

print(f"runing time: {end_time- start_time:0.4f} s")

# runing time: 5.6295 s

# Print the outputs.
for output in outputs:
	prompt = output.prompt
	generated_text = output.outputs[0].text
	print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")




import pandas as pd


'''
scp -P 50030 *.parquet jim.wang@37.224.68.132:/data/home/jim.wang/llama_2_7b_batch
'''


cia_paragraphs = pd.read_parquet('cia_paragraphs.parquet')
data = cia_paragraphs.to_dict('records')


system_prompt = f"""
Please generate question-answer pairs from the following document. Both the questions and answers should contain sufficient context information, and be self-contained, short, and abstract. The generated text is in the format of 

Q 1:
A 1:

Q 2:
A 2:

Q 3:
A 3:

...
"""

prompts = []

for r in data:
	paragraph = r['paragraphs']
	prompt = f"""
[INST] <<SYS>> {system_prompt.strip()} <<\SYS>>\n
Document: {paragraph}\n
[\INST] Sure. Here are the question and answer pairs: 
""".strip()
	prompts.append(prompt)


###

start_time = time.time()
responses = llm.generate(
	prompts, 
	sampling_params,
	)
end_time = time.time()

print(f"runing time: {end_time- start_time:0.4f} s")

#runing time:  17.0441 s


# Print the outputs.
for output in responses:
	prompt = output.prompt
	generated_text = output.outputs[0].text
	print(f"Prompt: {prompt!r},\n\n\nGenerated text: {generated_text!r}\n\n\n")

