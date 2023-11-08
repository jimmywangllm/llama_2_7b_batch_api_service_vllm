# llama_2_7b_batch_api_service_vllm
batch processing of Llama-2 7B

## create the virtual enviroment 

```
conda create --name=jim_python_3_8 python=3.8 -y
conda activate jim_python_3_8
conda deactivate
```

## install the packages

```
export TRANSFORMERS_CACHE=/data/hugging_face_model/
export TORCH_HOME=/data/hugging_face_model/

pip3 install vllm==0.2.1
pip3 install huggingface_hub

pip3 install Flask==2.2.2
pip3 install flasgger==0.9.5
pip3 install Werkzeug==2.2.2
pip3 install flask-restx==1.0.3
```
## start the service 

```
conda activate jim_python_3_8
cd /data/home/jim.wang/llama_2_7b_batch
python3 app_path.py >> log.txt &
```
