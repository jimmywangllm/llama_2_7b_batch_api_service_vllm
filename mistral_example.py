import os
from transformers import *
from huggingface_hub import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
os.environ["TRANSFORMERS_CACHE"] = "/data/hugging_face_model/"
os.environ["TORCH_HOME"] = "/data/hugging_face_model/"
os.environ["HUGGINGFACE_TOKEN"] = "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"
os.environ["HF_HOME"] = "/data/.cache/huggingface"
 os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/.cache/huggingface/hub"
 

'''
mkdir /data/.cache/
mkdir /data/.cache/huggingface/
mkdir /data/.cache/huggingface/hub/
chmod 777 /data/.cache/huggingface/hub
'''

MODEL_DIR = os.path.join(
	os.environ["TRANSFORMERS_CACHE"],
	"mistral_7b_instruct"
	)
BASE_MODEL = "meta-llama/Llama-2-7b-hf"

os.makedirs(MODEL_DIR, exist_ok=True)

snapshot_download(
BASE_MODEL,
local_dir=MODEL_DIR,
token=os.environ["HUGGINGFACE_TOKEN"],
)


