##############server_path.py##############

import re
import os
import time

print('loading the model')

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TORCH_HOME"] = "/data/hugging_face_model/"
os.environ["TRANSFORMERS_CACHE"] = "/data/hugging_face_model/"
os.environ["HUGGINGFACE_TOKEN"] = "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"
os.environ["HF_HOME"] = "/data/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/.cache/huggingface/hub"

from huggingface_hub import login
login(token = os.environ["HUGGINGFACE_TOKEN"])

from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf",)

def prompts_to_responses(
	prompts,
	max_tokens = 1024,
	):
	sampling_params = SamplingParams(
		temperature=0.75,
		top_p=1,
		max_tokens=max_tokens,
		presence_penalty=1.15,
		)
	responses = llm.generate(
		prompts, 
		sampling_params,
		use_tqdm = False,
		)
	outputs = []
	for output in responses:
		prompt = output.prompt
		generated_text = output.outputs[0].text
		outputs.append(generated_text)
	return outputs



##############

import re
import os
import time
import logging
import argsparser
from flask_restx import *
from flask import *

ns = Namespace(
	'llm', 
	description='LLM',
	)

args = argsparser.prepare_args()

#############

llama2_qa_parser = ns.parser()
llama2_qa_parser.add_argument('prompts', type=list, location='json')
llama2_qa_parser.add_argument('max_tokens', type=int, location='json')

llama2_qa_inputs = ns.model(
	'prompt_batch', 
		{
			'prompts': fields.List(
				fields.String(),
				example = [
				"Implement a Python function to compute the Fibonacci numbers.",
				"Write a Rust function that performs binary exponentiation.",
				"How do I allocate memory in C?",
				]),
			'max_tokens': fields.Integer(example = 512),
		}
	)

@ns.route('/llama_2_7b_batch')
class llama2_qa_api(Resource):
	def __init__(self, *args, **kwargs):
		super(llama2_qa_api, self).__init__(*args, **kwargs)
	@ns.expect(llama2_qa_inputs)
	def post(self):		
		start = time.time()
		try:			
			args = llama2_qa_parser.parse_args()	

			output = {}
			output['responses'] = prompts_to_responses(
				args['prompts'],
				max_tokens = args['max_tokens'],
				)
			output['status'] = 'success'
			output['running_time'] = float(time.time()- start)
			return output, 200
		except Exception as e:
			output = {}
			output['status'] = str(e)
			output['running_time'] = float(time.time()- start)
			return output

##############server_path.py##############