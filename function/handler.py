# python3 detection.py --modeldir=Sample_TFLite_model --graph=detect.tflite --labels=labelmap.txt --threshold=0.0 --image=test1.jpg --edgetpu --count=10
#response = requests.post('http://10.0.0.95:5000/', headers={'Use-Local-Image': 'image1.jpg'})
#curl -X POST -i -F image_file=@./test.jpg  http://localhost:5000/
#curl -X POST -H "Image-URL:http://10.0.0.95:5005/" http://localhost:5000/
#curl -X GET -i -H "Use-Local-Image: image1.jpg"  http://localhost:5000/
#curl -X GET -i -H "Image-URL: http://localhostmachine:5500/pioss/api/read/w3-ssd/pic_41.jpg"  http://10.0.0.90:31112/function/w5-ssd/
#curl -X GET -i -H "Image-URL: http://10.0.0.96:9000/mybucket/pic_41.jpg" -H "Internal-as:aaa" http://10.0.0.95:5001/

# Ref: from git clone https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi.git
# Files: from git clone https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi.git
# mv TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi tflite1
# cd tflite1
# Files needed: get_requirements.sh, this python file, Sample_TFLite_model/detect.tflite, Sample_TFLite_model/edgetpu.tflite, Sample_TFLite_model/labelmap.txt
# Note the model is SSDLite-MobileNet-v2 and can find 80 objects, download and unzip from wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
#Note: The python code (i.e., function) at the moment works only on model V1 and throws an error when using model V2 due to the difference image preprocessing requirements.
#By default, only CPU model is preloaded and inferences are run on CPU.

import os
from time import time
import urllib.request
import requests
import numpy as np
from flask import json, make_response
import datetime
import socket
import json
import argparse
import sys
import glob
import importlib.util
from pycoral.adapters import common
from PIL import Image
from . import load_inference_model
from . import inference
import threading
import multiprocessing

import names

from minio import Minio
from minio.commonconfig import Tags

#If you want to test the app localy on your host, set the env variable EXEC_ENV to 'local'
EXEC_ENV=os.getenv("EXEC_ENV", "container")

lock = threading.Lock()
# lock = request.environ['HTTP_FLASK_LOCK']

#initiate the config (Note: environment variables overwrite the default values if set)

import configparser
config = configparser.ConfigParser()
config.add_section('Default')
#CONFIG_FULL_PATH is meant to remain unchanged always since others read/write on this path.
# CONFIG_FULL_PATH = config['Default']['full_path'] = os.getenv("CONFIG_FULL_PATH", "/home/ubuntu/aiFaaS/config.ini")
# CONFIG_FULL_PATH = config['Default']['full_path'] = os.getenv("CONFIG_FULL_PATH", "/home/app/config.ini")
CONFIG_FULL_PATH = config['Default']['full_path'] = os.getenv("CONFIG_FULL_PATH", f"{'/home/ubuntu/aiFaaS/' if EXEC_ENV == 'local' else '/home/app/'}config.ini")


#multithreading
WAITRESS_THREADS = config['Default']['waitress_threads'] = os.getenv("WAITRESS_THREADS", "4")
WAITRESS_THREADS = int(WAITRESS_THREADS)
config.add_section('Model')
MODEL_PRE_LOAD = config['Model']['pre_load'] = os.getenv("MODEL_PRE_LOAD", 'cpu-only') #'no', 'cpu-only' or 'yes'

#[Supported Resources/Accelerator Detection]
#if hardware is on the device? (Note: env variables overwrite the supported device)
#[CPU]
MODEL_SUPPORTED_RESOURCES_CPU = config['Model']['supported_resources_cpu'] = os.getenv("MODEL_SUPPORTED_RESOURCES_CPU", "yes")
#search for TPU device
#hardware: A USB with attached Coral Edge TPU must be found. Its name may be 'Global Unichip Corp.' before use/after reboot or be 'Google Inc.' after first use.
usb_devices = ""

#[TPU]
print('Search for TPU...', flush=True)
import subprocess
process = subprocess.Popen(['lsusb'],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
output, error_cmd = process.communicate(timeout=30)
if process.returncode != 0: 
   usb_devices = 'Failed Command'
   captured_error = error_cmd.decode('utf-8') if error_cmd != None else 'No captured_error'
   print('Search for TPU by CMD = lsusb failed \n' + captured_error, flush=True)
else:
    usb_devices = output.decode('utf-8')
    print(f"USB devices:\n{usb_devices}", flush=True)

# usb_devices = subprocess.check_output('lsusb', shell=True, stderr=subprocess.STDOUT, timeout=30).decode("utf-8") 
is_tpu_available = 'yes' if 'Google Inc.' in usb_devices or 'Global Unichip Corp.' in usb_devices else 'no'
#software: Either of tflite_runtime or tensorflow must be installed and pycoral also must be installed.
pkg_tflite_tuntime = importlib.util.find_spec('tflite_runtime')
pkg_tensorflow = importlib.util.find_spec('tensorflow')
pkg_pycoral = importlib.util.find_spec('pycoral')

print(f"TPU packages (pkg_tflite_tuntime or pkg_tensorflow) is found? {'yes' if (pkg_tflite_tuntime or pkg_tensorflow) else 'no'}", flush=True)
print(f"TPU packages pkg_pycoral is found? {'yes' if pkg_pycoral else 'no'}", flush=True)

is_tpu_available = 'yes' if is_tpu_available=='yes' and (pkg_tflite_tuntime or pkg_tensorflow) and pkg_pycoral else 'no'
if is_tpu_available == 'yes':
    import pycoral.utils.edgetpu 
    print("edgetpu run-time version is " + str(pycoral.utils.edgetpu.get_runtime_version()))
if is_tpu_available == 'no':
    print("TPU not found (hardware search: lsusb=Google Inc. or Global Unichip Corp. and software search: tflite/tensorflow and pycoral"
    + "\nIf this is not an expected behavior,"
    + "\nMake sure USB and privileged permissions are given to container."
    + "\nMake sure USB devices are enabled by echo '2-1' |sudo tee /sys/bus/usb/drivers/usb/bind)", flush=True)
MODEL_SUPPORTED_RESOURCES_TPU = config['Model']['supported_resources_tpu'] = os.getenv("MODEL_SUPPORTED_RESOURCES_TPU", is_tpu_available)
if os.getenv("MODEL_SUPPORTED_RESOURCES_TPU"):
    if os.getenv("MODEL_SUPPORTED_RESOURCES_TPU") == 'yes' and is_tpu_available == 'no':
        print('MODEL_SUPPORTED_RESOURCES_TPU is set to no because TPU is not detected.', flush=True)
print('MODEL_SUPPORTED_RESOURCES_TPU=' + MODEL_SUPPORTED_RESOURCES_TPU, flush=True)

#[GPU]
print('Search for GPU...', flush=True)
#hardware: if device is Jetson Nano, Xavier or TX2. Ref: https://forums.developer.nvidia.com/t/best-way-to-check-which-tegra-board/111603/7
process = subprocess.Popen(['cat', '/proc/device-tree/model'],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
output, error_cmd = process.communicate(timeout=30)
if process.returncode != 0: 
   gpu_devices = 'Failed Command'
   captured_error = error_cmd.decode('utf-8') if error_cmd != None else ''
   print('Search for GPU hardware by CMD = cat /proc/device-tree/model failed (maybe --previliged is not given to the container)\n' + captured_error, flush=True)
else:
    gpu_devices = output.decode('utf-8')

# gpu_devices = subprocess.check_output('cat /proc/device-tree/model', shell=True, stderr=subprocess.STDOUT, timeout=30).decode("utf-8")
is_gpu_available = 'yes' if 'NVIDIA Jetson Nano' in gpu_devices or 'Jetson-AGX' in gpu_devices or 'quill' in gpu_devices else 'no'
if is_gpu_available == 'yes': print('GPU hardware is detected. Lets search for its software...', flush=True)
if gpu_devices != "Failed Command" and is_gpu_available != 'yes': print('GPU hardware search by /proc/device-tree/model did not fail, but it did not find the device as a GPU supported one, including NVIDIA Jetson Nano, Jetson-AGX, or quill', flush=True)

#software: if jetson package is installed
pkg_jetson = importlib.util.find_spec('jetson')
if is_gpu_available == 'yes' and not pkg_jetson:
    print('GPU software jetson not found. If this is not an expected behavior, check the base image if it is like dustynv/jetson-inference:r32.7.1', flush=True)
is_gpu_available = 'yes' if is_gpu_available=='yes' and pkg_jetson else 'no'
if is_gpu_available == 'no':
    print('GPU not found (hardware search: cat /proc/device-tree/model and software search: import jetson)', flush=True)
MODEL_SUPPORTED_RESOURCES_GPU = config['Model']['supported_resources_gpu'] = os.getenv("MODEL_SUPPORTED_RESOURCES_GPU", is_gpu_available)
if os.getenv("MODEL_SUPPORTED_RESOURCES_GPU"):
    if os.getenv("MODEL_SUPPORTED_RESOURCES_GPU") == 'yes' and is_gpu_available == 'no':
        print('MODEL_SUPPORTED_RESOURCES_GPU is set to no because GPU is not detected.', flush=True)
print('MODEL_SUPPORTED_RESOURCES_GPU=' + MODEL_SUPPORTED_RESOURCES_GPU, flush=True)

# MODEL_DIR = config['Model']['dir'] = os.getenv("MODEL_DIR", '/home/ubuntu/aiFaaS/networks/tensorflow-lite/SSD-MobileNet-V1-300-300-TF1-90obj/')
# MODEL_DIR = config['Model']['dir'] = os.getenv("MODEL_DIR", '/home/app/networks/tensorflow-lite/SSD-MobileNet-V1-300-300-TF1-90obj/')
MODEL_DIR = config['Model']['dir'] = os.getenv("MODEL_DIR", f"{'/home/ubuntu/aiFaaS/' if EXEC_ENV == 'local' else '/home/app/'}networks/tensorflow-lite/SSD-MobileNet-V1-300-300-TF1-90obj/")


MODEL_CPU_FILE = config['Model']['cpu_file'] = os.getenv("MODEL_CPU_FILE", "model.cpu.tflite")
MODEL_TPU_FILE = config['Model']['tpu_file'] = os.getenv("MODEL_TPU_FILE", "model.edgetpu.tflite")
MODEL_LABEL_FILE = config['Model']['label_file'] = os.getenv("MODEL_LABEL_FILE", "labelmap.txt")
MODEL_IMAGE_GET = config['Model']['image_get'] = os.getenv("MODEL_IMAGE_GET", 'single') #or batch that will feed from image_dir
# MODEL_IMAGE_DIR = config['Model']['image_dir'] = os.getenv("MODEL_IMAGE_DIR", "/home/ubuntu/aiFaaS/images/")
# MODEL_IMAGE_DIR = config['Model']['image_dir'] = os.getenv("MODEL_IMAGE_DIR", "/home/app/images/")
MODEL_IMAGE_DIR = config['Model']['image_dir'] = os.getenv("MODEL_IMAGE_DIR", f"{'/home/ubuntu/aiFaaS/' if EXEC_ENV == 'local' else '/home/app/'}images/")


# MODEL_IMAGE_SAMPLE1 = config['Model']['image_sample1'] = os.getenv("MODEL_IMAGE_SAMPLE1", "/home/ubuntu/aiFaaS/images/image1.jpg")
MODEL_IMAGE_SAMPLE1 = config['Model']['image_sample1'] = os.getenv("MODEL_IMAGE_SAMPLE1", f"{'/home/ubuntu/aiFaaS/' if EXEC_ENV == 'local' else '/home/app/'}images/image1.jpg")


MODEL_MIN_CONFIDENCE_THRESHOLD = config['Model']['min_confidence_threshold'] = os.getenv("MODEL_MIN_CONFIDENCE_THRESHOLD", '0.5')
MODEL_INFERENCE_REPEAT = config['Model']['inference_repeat'] = os.getenv("MODEL_INFERENCE_REPEAT", '1')
#number of thread workers the tensorflow will spawn to do the object detection per task. Default is 1.
MODEL_CPU_TPU_INTERPRETER_THREADS = config['Model']['interpreter_cpu_tpu_threads'] = os.getenv("MODEL_CPU_TPU_INTERPRETER_THREADS", '1')
MODEL_CPU_TPU_INTERPRETER_THREADS = int(MODEL_CPU_TPU_INTERPRETER_THREADS)

MODEL_RUN_ON = config['Model']['run_on'] = os.getenv("MODEL_RUN_ON", 'cpu') #cpu, tpu or gpu

#CPU
MODEL_CPU_WORKERS = config['Model']['cpu_workers'] = os.getenv("MODEL_CPU_WORKERS", "1")
MODEL_CPU_WORKERS = int(MODEL_CPU_WORKERS)

#GPU
# MODEL_DIR_GPU = config['Model']['dir_gpu'] = os.getenv("MODEL_DIR_GPU", '/home/ubuntu/aiFaaS/networks/SSD-Mobilenet-v1/')
# MODEL_DIR_GPU = config['Model']['dir_gpu'] = os.getenv("MODEL_DIR_GPU", '/home/app/networks/SSD-Mobilenet-v1/')
MODEL_DIR_GPU = config['Model']['dir_gpu'] = os.getenv("MODEL_DIR_GPU", f"{'/home/ubuntu/aiFaaS/' if EXEC_ENV == 'local' else '/home/app/'}networks/SSD-Mobilenet-v1/")


MODEL_GPU_FILE = config['Model']['gpu_file'] = os.getenv("MODEL_GPU_FILE", "ssd_mobilenet_v1_coco.uff")
MODEL_LABEL_FILE_GPU = config['Model']['label_file_gpu'] = os.getenv("MODEL_LABEL_FILE_GPU", "ssd_coco_labels.txt")
MODEL_GPU_BUILTIN_NETWORK = config['Model']['gpu_builtin_network'] = os.getenv("MODEL_GPU_BUILTIN_NETWORK", "ssd-mobilenet-v1")

#check misconfiguration of threads and models
if MODEL_RUN_ON == 'cpu':
    if MODEL_CPU_WORKERS != WAITRESS_THREADS:
        print('Misconfiguration: MODEL_RUN_ON=' + str(MODEL_RUN_ON) + ', but MODEL_CPU_WORKERS (' + str(MODEL_CPU_WORKERS) 
                    + ') != WAITRESS_THREADS (' + str(WAITRESS_THREADS) + ') while they better be equal and X times cpu cores.', flush=True)
elif MODEL_RUN_ON == 'tpu':
    if WAITRESS_THREADS > 1:
        print('Misconfiguration: MODEL_RUN_ON=' + str(MODEL_RUN_ON) + ', but WAITRESS_THREADS (' + str(WAITRESS_THREADS) + ') !=1', flush=True)
elif MODEL_RUN_ON == 'gpu':
    if WAITRESS_THREADS == 1:
        print('Misconfiguration: MODEL_RUN_ON=' + str(MODEL_RUN_ON) + ', but WAITRESS_THREADS (' + str(WAITRESS_THREADS) 
                + ') ==1 while it can be X times CPU cores.', flush=True)

#persist the config
with open(CONFIG_FULL_PATH, 'w') as configfile:
    config.write(configfile)


#overwrite config by arguments
#...

#minio
minio_client = None
if os.getenv('MINIO_ENABLED'):
    #MINIO_ENABLED=endpoint=10.0.0.96:9000,access_key=minioadmin,secret_key=minioadmin,secure=False
    try:
        minio_server_info = str(os.getenv('MINIO_ENABLED'))
        minio_endpoint = minio_server_info.split(',')[0].split('=')[1]
        minio_access_key = minio_server_info.split(',')[1].split('=')[1]
        minio_secret_key = minio_server_info.split(',')[2].split('=')[1]
        minio_secure = minio_server_info.split(',')[3].split('=')[1]
        if minio_secure.lower() == 'false' or minio_secure.lower() == 'no' or minio_secure.lower() == '0':
            minio_secure = False
        elif minio_secure.lower() == 'true' or minio_secure.lower() == 'yes' or minio_secure.lower() == '1':
            minio_secure = True
        else:
            pass
        minio_client = Minio(endpoint = minio_endpoint, access_key = minio_access_key, secret_key = minio_secret_key, secure = minio_secure)
    except Exception as e:
        print('minio_server_info=' + minio_server_info)
        print(str(e))

#session
internal_session = requests.session()
internal_session.mount(
    'http://',
    requests.adapters.HTTPAdapter(pool_maxsize= 10,
                                    max_retries=3,
                                    pool_block=True))

#update config: in case config.ini is updated, apply the changes in the assocciated variables.
def get_latest_config(lock):
    global CONFIG_FULL_PATH, WAITRESS_THREADS, MODEL_DIR, MODEL_DIR_GPU, MODEL_CPU_FILE, MODEL_TPU_FILE, MODEL_GPU_FILE, MODEL_GPU_BUILTIN_NETWORK, MODEL_LABEL_FILE, MODEL_LABEL_FILE_GPU, MODEL_IMAGE_GET, MODEL_IMAGE_DIR, MODEL_IMAGE_SAMPLE1, MODEL_MIN_CONFIDENCE_THRESHOLD, MODEL_INFERENCE_REPEAT, MODEL_CPU_TPU_INTERPRETER_THREADS, MODEL_RUN_ON, MODEL_SUPPORTED_RESOURCES_CPU, MODEL_SUPPORTED_RESOURCES_TPU, MODEL_SUPPORTED_RESOURCES_GPU, MODEL_CPU_WORKERS
    with lock:
        config = configparser.ConfigParser()
        config.read(CONFIG_FULL_PATH)
        CONFIG_FULL_PATH = config['Default']['full_path'] 
        WAITRESS_THREADS = int(config['Default']['waitress_threads'])
        MODEL_PRE_LOAD = config['Model']['pre_load']

        MODEL_SUPPORTED_RESOURCES_CPU = config['Model']['supported_resources_cpu'] 
        MODEL_SUPPORTED_RESOURCES_TPU = config['Model']['supported_resources_tpu'] 
        MODEL_SUPPORTED_RESOURCES_GPU = config['Model']['supported_resources_gpu'] 

        MODEL_DIR = config['Model']['dir'] 
        MODEL_DIR_GPU = config['Model']['dir_gpu']
        MODEL_CPU_FILE = config['Model']['cpu_file'] 
        MODEL_TPU_FILE = config['Model']['tpu_file'] 
        MODEL_GPU_FILE = config['Model']['gpu_file']
        MODEL_GPU_BUILTIN_NETWORK = config['Model']['gpu_builtin_network']
        MODEL_LABEL_FILE = config['Model']['label_file'] 
        MODEL_LABEL_FILE_GPU = config['Model']['label_file_gpu']
        MODEL_IMAGE_GET = config['Model']['image_get'] 
        MODEL_IMAGE_DIR = config['Model']['image_dir'] 
        MODEL_IMAGE_SAMPLE1 = config['Model']['image_sample1']  
        MODEL_MIN_CONFIDENCE_THRESHOLD = config['Model']['min_confidence_threshold'] 
        MODEL_INFERENCE_REPEAT = config['Model']['inference_repeat'] 
        MODEL_CPU_TPU_INTERPRETER_THREADS = config['Model']['interpreter_cpu_tpu_threads']
        MODEL_CPU_TPU_INTERPRETER_THREADS = int(MODEL_CPU_TPU_INTERPRETER_THREADS)

        MODEL_RUN_ON = config['Model']['run_on'] 
        MODEL_CPU_WORKERS = int(config['Model']['cpu_workers'])

        #check misconfiguration of threads and models
        if MODEL_RUN_ON == 'cpu':
            if MODEL_CPU_WORKERS != WAITRESS_THREADS:
                print('Misconfiguration: MODEL_RUN_ON=' + str(MODEL_RUN_ON) + ', but MODEL_CPU_WORKERS (' + str(MODEL_CPU_WORKERS) 
                            + ') != WAITRESS_THREADS (' + str(WAITRESS_THREADS) + ') while they better be equal and X times cpu cores.', flush=True)
        elif MODEL_RUN_ON == 'tpu':
            if WAITRESS_THREADS > 1:
                print('Misconfiguration: MODEL_RUN_ON=' + str(MODEL_RUN_ON) + ', but WAITRESS_THREADS (' + str(WAITRESS_THREADS) + ') !=1', flush=True)
        elif MODEL_RUN_ON == 'gpu':
            if WAITRESS_THREADS == 1:
                print('Misconfiguration: MODEL_RUN_ON=' + str(MODEL_RUN_ON) + ', but WAITRESS_THREADS (' + str(WAITRESS_THREADS) 
                        + ') ==1 while it can be X times CPU cores.', flush=True)
        

    return CONFIG_FULL_PATH, WAITRESS_THREADS, MODEL_DIR, MODEL_DIR_GPU, MODEL_CPU_FILE, MODEL_TPU_FILE, MODEL_GPU_FILE, MODEL_GPU_BUILTIN_NETWORK, MODEL_LABEL_FILE, MODEL_LABEL_FILE_GPU, MODEL_IMAGE_GET, MODEL_IMAGE_DIR, MODEL_IMAGE_SAMPLE1, MODEL_MIN_CONFIDENCE_THRESHOLD, MODEL_INFERENCE_REPEAT,MODEL_CPU_TPU_INTERPRETER_THREADS, MODEL_RUN_ON, MODEL_SUPPORTED_RESOURCES_CPU, MODEL_SUPPORTED_RESOURCES_TPU, MODEL_SUPPORTED_RESOURCES_GPU, MODEL_CPU_WORKERS



#keep this resource in mind (cpu or tpu or gpu?)
CURRENT_MODEL_RUN_ON = MODEL_RUN_ON

#inference workers
#NOTE: cpu interpreter is an array because "flask uses multiple threads. Tensorflow models loaded in one thread, must be used in that same thread." Ref: https://stackoverflow.com/questions/49400440/using-keras-model-in-flask-app-with-threading
interpreter_cpu, interpreter_tpu, interpreter_gpu = [None]*MODEL_CPU_WORKERS, None, None
floating_model_cpu, input_mean_cpu, input_std_cpu, input_details_cpu, output_details_cpu, boxes_idx_cpu, classes_idx_cpu,scores_idx_cpu, labels_cpu, error_tmp = [None]*MODEL_CPU_WORKERS,[None]*MODEL_CPU_WORKERS,[None]*MODEL_CPU_WORKERS,[None]*MODEL_CPU_WORKERS,[None]*MODEL_CPU_WORKERS,[None]*MODEL_CPU_WORKERS,[None]*MODEL_CPU_WORKERS,[None]*MODEL_CPU_WORKERS,[None]*MODEL_CPU_WORKERS,[None]*MODEL_CPU_WORKERS

#should it pre load models for supported resources?
if MODEL_PRE_LOAD == 'yes':
    print('Preload models...', flush=True)
    #bring all models up if the resource is supported
    #cpu
    if MODEL_SUPPORTED_RESOURCES_CPU == 'yes':
        print('Load cpu model ' + str(MODEL_CPU_WORKERS) + ' times', flush=True)
        MODEL_RUN_ON = 'cpu'
        for i in range(MODEL_CPU_WORKERS):
            
            interpreter_cpu[i], floating_model_cpu[i], input_mean_cpu[i], input_std_cpu[i], input_details_cpu[i], output_details_cpu[i], boxes_idx_cpu[i], classes_idx_cpu[i],scores_idx_cpu[i], labels_cpu[i], error_tmp[i] = load_inference_model.load(MODEL_RUN_ON, MODEL_DIR, MODEL_DIR_GPU, MODEL_CPU_FILE, MODEL_TPU_FILE, MODEL_GPU_FILE, MODEL_LABEL_FILE, MODEL_LABEL_FILE_GPU, MODEL_GPU_BUILTIN_NETWORK, MODEL_MIN_CONFIDENCE_THRESHOLD, MODEL_IMAGE_SAMPLE1, MODEL_INFERENCE_REPEAT, MODEL_CPU_TPU_INTERPRETER_THREADS)

            if error_tmp[i]:
                print('loading a CPU model failed???????????:\n' + str(error_tmp[i]), flush=True)
                print(str(len(error_tmp)))
    #tpu
    if MODEL_SUPPORTED_RESOURCES_TPU == 'yes':
        print('Load tpu model... ', flush=True)
        MODEL_RUN_ON = 'tpu'
        interpreter_tpu, floating_model_tpu, input_mean_tpu, input_std_tpu, input_details_tpu, output_details_tpu, boxes_idx_tpu, classes_idx_tpu,scores_idx_tpu, labels_tpu, error_tmp[0] = load_inference_model.load(MODEL_RUN_ON, MODEL_DIR, MODEL_DIR_GPU, MODEL_CPU_FILE, MODEL_TPU_FILE, MODEL_GPU_FILE, MODEL_LABEL_FILE, MODEL_LABEL_FILE_GPU, MODEL_GPU_BUILTIN_NETWORK, MODEL_MIN_CONFIDENCE_THRESHOLD, MODEL_IMAGE_SAMPLE1, MODEL_INFERENCE_REPEAT, MODEL_CPU_TPU_INTERPRETER_THREADS)
        if error_tmp[0]:
            print('loading a TPU model failed???????????????:\n' + str(error_tmp[0]), flush=True)
    #gpu
    if MODEL_SUPPORTED_RESOURCES_GPU == 'yes':
        print('Load gpu model... ', flush=True)
        MODEL_RUN_ON = 'gpu'
        interpreter_gpu, labels_gpu, error_tmp[0] = load_inference_model.load(MODEL_RUN_ON, MODEL_DIR, MODEL_DIR_GPU, MODEL_CPU_FILE, MODEL_TPU_FILE, MODEL_GPU_FILE, MODEL_LABEL_FILE, MODEL_LABEL_FILE_GPU, MODEL_GPU_BUILTIN_NETWORK, MODEL_MIN_CONFIDENCE_THRESHOLD, MODEL_IMAGE_SAMPLE1, MODEL_INFERENCE_REPEAT, MODEL_CPU_TPU_INTERPRETER_THREADS)
        if error_tmp[0]:
            print('loading a GPU model failed?????????????:\n' + str(error_tmp[0]), flush=True)

    #fix MODEL_RUN_ON
    MODEL_RUN_ON = CURRENT_MODEL_RUN_ON

    print('All requested models are preloaded', flush=True)


elif MODEL_PRE_LOAD == 'cpu-only':
    #cpu
    if MODEL_SUPPORTED_RESOURCES_CPU == 'yes':
        MODEL_RUN_ON = 'cpu'

        for i in range(MODEL_CPU_WORKERS):
            print('Load cpu model ' + str(MODEL_CPU_WORKERS) + ' times', flush=True)
        
            interpreter_cpu_tmp, floating_model_cpu[i], input_mean_cpu[i], input_std_cpu[i], input_details_cpu[i], output_details_cpu[i], boxes_idx_cpu[i], classes_idx_cpu[i],scores_idx_cpu[i], labels_cpu[i], error_tmp[i] = load_inference_model.load(MODEL_RUN_ON, MODEL_DIR, MODEL_DIR_GPU, MODEL_CPU_FILE, MODEL_TPU_FILE, MODEL_GPU_FILE, MODEL_LABEL_FILE, MODEL_LABEL_FILE_GPU, MODEL_GPU_BUILTIN_NETWORK, MODEL_MIN_CONFIDENCE_THRESHOLD, MODEL_IMAGE_SAMPLE1, MODEL_INFERENCE_REPEAT, MODEL_CPU_TPU_INTERPRETER_THREADS)
            interpreter_cpu[i] = interpreter_cpu_tmp
            # import copy
            # interpreter_cpu[i] = copy.deepcopy(interpreter_cpu_tmp)
            if str(error_tmp[0]):
                print('loading a model failed??????????:\n' + str(error_tmp[0]), flush=True)
            
elif MODEL_PRE_LOAD == 'no':
    print('No model is loaded', flush=True)
else:
    print('ERROR: MODEL_PRE_LOAD=' + MODEL_PRE_LOAD + ', but must be yes, no, or cpu-only.', flush=True)



#restrict access to the main function by MODEL_CPU_WORKERS
semaphore = threading.Semaphore(MODEL_CPU_WORKERS)

#handle
def handle(request, counter):
    """handle a request to the function
    Args:
        request (object): Flask request object
    """
    print('counter= ' + str(counter))

    global CONFIG_FULL_PATH, WAITRESS_THREADS, MODEL_DIR, MODEL_CPU_FILE, MODEL_TPU_FILE, MODEL_LABEL_FILE, MODEL_IMAGE_GET, MODEL_IMAGE_DIR, MODEL_IMAGE_SAMPLE1, MODEL_MIN_CONFIDENCE_THRESHOLD, MODEL_INFERENCE_REPEAT, MODEL_RUN_ON, MODEL_CPU_WORKERS, MODEL_CPU_TPU_INTERPRETER_THREADS
    global CURRENT_MODEL_RUN_ON

    global lock

    global interpreter_cpu, floating_model_cpu, input_mean_cpu, input_std_cpu, input_details_cpu, output_details_cpu, boxes_idx_cpu, classes_idx_cpu,scores_idx_cpu, labels_cpu
    global interpreter_tpu, floating_model_tpu, input_mean_tpu, input_std_tpu, input_details_tpu, output_details_tpu, boxes_idx_tpu, classes_idx_tpu,scores_idx_tpu, labels_tpu
    global interpreter_gpu, labels_gpu


    global semaphore
    
    worker_index = counter % MODEL_CPU_WORKERS

    error = ""

    start_main = datetime.datetime.now(datetime.timezone.utc).astimezone().timestamp()

    #Get latest config
    start = datetime.datetime.now(datetime.timezone.utc).astimezone().timestamp()
    try:
        CONFIG_FULL_PATH, WAITRESS_THREADS, MODEL_DIR, MODEL_DIR_GPU, MODEL_CPU_FILE, MODEL_TPU_FILE, MODEL_GPU_FILE, MODEL_GPU_BUILTIN_NETWORK, MODEL_LABEL_FILE, MODEL_LABEL_FILE_GPU, MODEL_IMAGE_GET, MODEL_IMAGE_DIR, MODEL_IMAGE_SAMPLE1, MODEL_MIN_CONFIDENCE_THRESHOLD, MODEL_INFERENCE_REPEAT, MODEL_CPU_TPU_INTERPRETER_THREADS, MODEL_RUN_ON, MODEL_SUPPORTED_RESOURCES_CPU, MODEL_SUPPORTED_RESOURCES_TPU, MODEL_SUPPORTED_RESOURCES_GPU,MODEL_CPU_WORKERS = get_latest_config(lock)
    except Exception as e:
        error = 'Get latest config: ' + str(e)
        return None, None, error

    elapsed_config = datetime.datetime.now(datetime.timezone.utc).astimezone().timestamp() - start

    #lock for cpu workers max
    if MODEL_RUN_ON == 'cpu':
        semaphore.acquire()

    #Get image
    start = datetime.datetime.now(datetime.timezone.utc).astimezone().timestamp()
    url = None
    raw_image = None

    # Check if image URL specified in request query parameter
    if request.args.get('image_url'):
        url = str(request.args.get('image_url'))

    # Check if image URL specified in request header
    elif request.headers.get('Image-URL'):
        url = str(request.headers.get('Image-URL'))
        #replcae localhostmachine with function host IP. Applies to when object storage is local to function place.
        if 'localhostmachine' in url:
            if os.getenv('POD_HOST_IP'):
                url = url.replace('localhostmachine', str(os.getenv('POD_HOST_IP')), 1)
            else:
                error += 'localhostmachine in Image-URL=' + url + ', but function failed to obtain POD_HOST_IP env.\n' 
                url = None

    # Check if image URL specified in request body as raw text
    else:
        try:
            body_url = request.get_data().decode('UTF-8')
            if body_url:
                url = body_url
        except:
            pass

    #fetch image
    if url is not None:
        try:
            #headers
            #like headers["Connection"] = "keep-alive" and/or headers["Keep-Alive"] = "timeout=5, max=100"
            internal_request_header = {}
            session_is_requested = False
            for k,v in dict(request.headers).items():
                if 'Internal-' in k:
                    internal_request_header[k.replace('Internal-', '')] = v
                if 'Internal-Session' in k:
                    session_is_requested = True
            #url_req = urllib.request.urlopen(url)
            #image_array = np.asarray(bytearray(url_req.read()), dtype=np.uint8)
            #image = cv2.imdecode(image_array, -1)
            #session.get
            if session_is_requested:
                with internal_session as session:
                    raw_image = Image.open(session.get(url, headers= internal_request_header, stream=True).raw)
            #requests.get
            else:
                raw_image = Image.open(requests.get(url, headers= internal_request_header, stream=True).raw)
            #Ref: https://stackoverflow.com/questions/7391945/how-do-i-read-image-data-from-a-url-in-python
            #Or
        except Exception as e:
            error += '\nFetch file: Fetching file failed, URL: ' + str(url) + '\n' + str(e)
            print("Fetch file failed, URL: " + str(url) + '\n' + str(e), flush=True)
    else:
    #read image
        try:
            #use local images
            if request.headers.get('Use-Local-Image'):
                print('using sample image= ' + str(request.headers.get('Use-Local-Image')), flush=True)

                #by given image name, using '.' delimeter
                if len(request.headers.get('Use-Local-Image').split('.')) > 1:
                    raw_image = Image.open(MODEL_IMAGE_DIR + request.headers.get('Use-Local-Image'))

                #by random within a range according to value range and counter
                elif len(request.headers.get('Use-Local-Image').split('-')) > 1:
                    #get range
                    pics_num = int(request.headers.get('Use-Local-Image').split('-')[1]) - int(request.headers.get('Use-Local-Image').split('-')[0])
                    #calculate file_name
                    file_name = 'pic_' + str(counter % pics_num if counter % pics_num != 0  else 1) + '.jpg'
                    print('file_name= ' + file_name, flush=True)

                    raw_image = Image.open(MODEL_IMAGE_DIR + file_name)    

            #attached
            else:
                print('read image from request', flush=True)
                file = request.files['image_file']
                #image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
                raw_image = Image.open(file)
                #req = urllib.request.Request(image_url)
                #response = urllib.request.urlopen(req)
                #image_data = response.read()
        except Exception as e:
            error += '\nRead image: Reading file from image_file failed\n' + str(e)
            print("Read image: Reading file from image_file failed\n" + str(e), flush=True)


    end = datetime.datetime.now(datetime.timezone.utc).astimezone().timestamp()
    elapsed_fetch_image = end - start

    if raw_image is None:
        error += "\nImage must be specified either as file, or URL. For file, it needs to be attached in a multipart/form-data request via the 'image_file' key. For URL, it must be provided via query parameter 'image_url', via request header 'Image-URL', or via raw plaintext in the request body."
        return None, None, error


    #[Get model]
    start = datetime.datetime.now(datetime.timezone.utc).astimezone().timestamp()
    
    #check for loading a model and setting its interpreter
        
    #if a model other than the current model in use is requested by MODEL_RUN_ON
    if CURRENT_MODEL_RUN_ON != MODEL_RUN_ON:
        #if the requested model is cpu
        if MODEL_RUN_ON == 'cpu':
            #if it is not already loaded
            # if interpreter_cpu == None:
            if interpreter_cpu == [None]*MODEL_CPU_WORKERS:
                #if the resource is supported
                if MODEL_SUPPORTED_RESOURCES_CPU == 'yes':
                    #load it and set its interpreter
                    for i in range(MODEL_CPU_WORKERS):
                        print('Load cpu model ' + str(MODEL_CPU_WORKERS) + ' times', flush=True)
                        interpreter_cpu_tmp, floating_model_cpu[i], input_mean_cpu[i], input_std_cpu[i], input_details_cpu[i], output_details_cpu[i], boxes_idx_cpu[i], classes_idx_cpu[i],scores_idx_cpu[i], labels_cpu[i], error[i] = load_inference_model.load(MODEL_RUN_ON, MODEL_DIR, MODEL_DIR_GPU, MODEL_CPU_FILE, MODEL_TPU_FILE, MODEL_GPU_FILE, MODEL_LABEL_FILE, MODEL_LABEL_FILE_GPU, MODEL_GPU_BUILTIN_NETWORK, MODEL_MIN_CONFIDENCE_THRESHOLD, MODEL_IMAGE_SAMPLE1, MODEL_INFERENCE_REPEAT, MODEL_CPU_TPU_INTERPRETER_THREADS)   
                        interpreter_cpu[i] = interpreter_cpu_tmp
                        # import copy
                        # interpreter_cpu[i] = copy.deepcopy(interpreter_cpu_tmp)
                        if len(error)==0:
                            print('MODEL_RUN_ON = ' + MODEL_RUN_ON + ', so a model is loaded and its interpreter is set to be used from now on', flush=True)    
                        else:
                            error += '\nHandler failed to load a model'
                            return None, None, error
                else:
                    error += '\nMODEL_RUN_ON = ' + MODEL_RUN_ON + ', but MODEL_SUPPORTED_RESOURCES_CPU = ' + MODEL_SUPPORTED_RESOURCES_CPU
                    return None, None, error
            else:
                print('MODEL_RUN_ON switched from ' + CURRENT_MODEL_RUN_ON + ' to ' + MODEL_RUN_ON + ' that is already loaded and has an interpreter', flush=True)
        
        #if the requested model is tpu
        elif MODEL_RUN_ON == 'tpu':
            #if it is not already loaded
            if interpreter_tpu == None:
                #if the resource is supported
                if MODEL_SUPPORTED_RESOURCES_TPU == 'yes':
                    #load it and set its interpreter
    
                    interpreter_tpu, floating_model_tpu, input_mean_tpu, input_std_tpu, input_details_tpu, output_details_tpu, boxes_idx_tpu, classes_idx_tpu,scores_idx_tpu, labels_tpu, error = load_inference_model.load(MODEL_RUN_ON, MODEL_DIR, MODEL_DIR_GPU, MODEL_CPU_FILE, MODEL_TPU_FILE, MODEL_GPU_FILE, MODEL_LABEL_FILE, MODEL_LABEL_FILE_GPU, MODEL_GPU_BUILTIN_NETWORK, MODEL_MIN_CONFIDENCE_THRESHOLD, MODEL_IMAGE_SAMPLE1, MODEL_INFERENCE_REPEAT, MODEL_CPU_TPU_INTERPRETER_THREADS)

                    if len(error)==0:
                        print('MODEL_RUN_ON = ' + MODEL_RUN_ON + ', so a model is loaded and its interpreter is set to be used from now on', flush=True)    
                    else:
                        error += '\nHandler failed to load a model'
                        return None, None, error
                else:
                    error += '\nMODEL_RUN_ON = ' + MODEL_RUN_ON + ', but MODEL_SUPPORTED_RESOURCES_TPU = ' + MODEL_SUPPORTED_RESOURCES_TPU
                    return None, None, error
            else:
                print('MODEL_RUN_ON switched from ' + CURRENT_MODEL_RUN_ON + ' to ' + MODEL_RUN_ON + ' that is already loaded and has an interpreter', flush=True)

        #if the requested model is gpu
        elif MODEL_RUN_ON == 'gpu':
            #if it is not already loaded
            if interpreter_gpu == None:
                #if the resource is supported
                if MODEL_SUPPORTED_RESOURCES_GPU == 'yes':
                    #load it and set its interpreter
                    
                    interpreter_gpu, labels_gpu, error = load_inference_model.load(MODEL_RUN_ON, MODEL_DIR, MODEL_DIR_GPU, MODEL_CPU_FILE, MODEL_TPU_FILE, MODEL_GPU_FILE, MODEL_LABEL_FILE, MODEL_LABEL_FILE_GPU, MODEL_GPU_BUILTIN_NETWORK, MODEL_MIN_CONFIDENCE_THRESHOLD, MODEL_IMAGE_SAMPLE1, MODEL_INFERENCE_REPEAT, MODEL_CPU_TPU_INTERPRETER_THREADS)
                    
                    if len(error)==0:
                        print('MODEL_RUN_ON = ' + MODEL_RUN_ON + ', so a model is loaded and its interpreter is set to be used from now on', flush=True)    
                    else:
                        error += '\nHandler failed to load a model'
                        return None, None, error

                else:
                    error += '\nMODEL_RUN_ON = ' + MODEL_RUN_ON + ', but MODEL_SUPPORTED_RESOURCES_GPU = ' + MODEL_SUPPORTED_RESOURCES_GPU
                    return None, None, error
            else:
                print('MODEL_RUN_ON switched from ' + CURRENT_MODEL_RUN_ON + ' to ' + MODEL_RUN_ON + ' that is already loaded and has an interpreter', flush=True)

        else:
            error += '\nLoading a model for MODEL_RUN_ON = ' + MODEL_RUN_ON + ' is not implemented'
            return None, None, error

        #keep current model in mind
        CURRENT_MODEL_RUN_ON = MODEL_RUN_ON
    else:
        pass
        #no action, use previously in use interpreter


    #pick an already loaded interpreter
    print('mode run on ' + MODEL_RUN_ON)
    if MODEL_RUN_ON == 'cpu':
        print('interpreter_cpu', flush=True)

        interpreter_worker = interpreter_cpu
    elif MODEL_RUN_ON == 'tpu':
        if MODEL_SUPPORTED_RESOURCES_TPU == 'yes':
            interpreter_worker = interpreter_tpu
            print('interpreter_tpu', flush=True)
        else:
            error +=  '\nMODEL_RUN_ON == tpu and MODEL_SUPPORTED_RESOURCES_TPU != yes, so first enable MODEL_SUPPORTED_RESOURCES_TPU and load its model'
            return None, None, error
    elif MODEL_RUN_ON == 'gpu':
        if MODEL_SUPPORTED_RESOURCES_GPU == 'yes':
            print('interpreter_gpu')
            interpreter_worker = interpreter_gpu
        else:
            error += '\nMODEL_RUN_ON == gpu and MODEL_SUPPORTED_RESOURCES_GPU != yes, so first enable MODEL_SUPPORTED_RESOURCES_GPU and load its model'
            return None, None, error
    else:
        error += '\nMODEL_RUN_ON not found, value = ' + MODEL_RUN_ON
        return None, None, error

    elapsed_load_model = datetime.datetime.now(datetime.timezone.utc).astimezone().timestamp() - start


    #[Image Preprocessing]
    start = datetime.datetime.now(datetime.timezone.utc).astimezone().timestamp()

    #for GPU use
    if MODEL_RUN_ON == 'gpu':
        #jetson.inference loads images by jetson.utils.loadImage from the host in a given address and does not need any extra work, but if the image is from Pillow or Numpy, etc. objects, the image  needs converting. Hence, to avoid conversions, I just save the Pillow file locally, load it and delete it.
        #raw_image is an image received by request as read by Image.open()
        if hasattr(raw_image, 'filename') and len(raw_image.filename)> 0:
            #if has a name, pick the actual name from the path (if that is a path)
            image_name = raw_image.filename.split('/')[-1]
            
        else:
            #generate a random name
            image_name = names.get_first_name(gender='female') + '.jpg'
        #save on the current directory
        raw_image.save(image_name)
        raw_image.close()

        #give the image to jetson-inference
        import jetson.utils
        img_cuda = jetson.utils.loadImage(image_name)
        #delete it from host
        os.remove(image_name)


    #for CPU or TPU
    else:
        # Model must be uint8 quantized
        #cpu
        if MODEL_RUN_ON == 'cpu':
            if common.input_details(interpreter_worker[worker_index], 'dtype') != np.uint8:
                raise ValueError('Only support uint8 input type.')

            size = common.input_size(interpreter_worker[worker_index])
        #tpu
        else:
            if common.input_details(interpreter_worker, 'dtype') != np.uint8:
                raise ValueError('Only support uint8 input type.')

            size = common.input_size(interpreter_worker)

        #The Image.ANTIALIAS in EdjeElectronics example gives a warning that  DeprecationWarning: ANTIALIAS is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.LANCZOS instead.
        #image = Image.open(args.image).convert('RGB').resize(size, Image.ANTIALIAS)
        #image = Image.open('/home/ubuntu/object-detection/ssd-cpu-tpu/function/image2.jpg').convert('RGB').resize(size, Image.ANTIALIAS)
        image = raw_image.convert('RGB').resize(size, Image.LANCZOS)

        # Image data must go through two transforms before running inference:
        # 1. normalization: f = (input - mean) / std
        # 2. quantization: q = f / scale + zero_point
        # The following code combines the two steps as such:
        # q = (input - mean) / (std * scale) + zero_point
        # However, if std * scale equals 1, and mean - zero_point equals 0, the input
        # does not need any preprocessing (but in practice, even if the results are
        # very close to 1 and 0, it is probably okay to skip preprocessing for better
        # efficiency; we use 1e-5 below instead of absolute zero).
        #cpu
        if MODEL_RUN_ON == 'cpu':
            params = common.input_details(interpreter_worker[worker_index], 'quantization_parameters')
        #tpu
        else:
            params = common.input_details(interpreter_worker, 'quantization_parameters')
        scale = params['scales']
        zero_point = params['zero_points']
        mean = 128.0
        std = 128.0
        if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
            # Input data does not require preprocessing.
            #cpu
            if MODEL_RUN_ON == 'cpu':
                common.set_input(interpreter_worker[worker_index], image)
            #tpu
            else:
                common.set_input(interpreter_worker, image)
            input_data = image
        else:
            # Input data requires preprocessing
            normalized_input = (np.asarray(image) - mean) / (std * scale) + zero_point
            np.clip(normalized_input, 0, 255, out=normalized_input)
            #cpu
            if MODEL_RUN_ON == 'cpu':
                common.set_input(interpreter_worker[worker_index], normalized_input.astype(np.uint8))
            #tpu
            else:
                common.set_input(interpreter_worker, normalized_input.astype(np.uint8))
            input_data = normalized_input
            

        # Load image and resize to expected shape [1xHxWx3]
        # image = cv2.imread(image_path)
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # imH, imW, _ = image.shape 
        # image_resized = cv2.resize(image_rgb, (width, height))
        image_resized = input_data
        input_data = np.expand_dims(image_resized, axis=0)
        
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        #cpu
        if MODEL_RUN_ON == 'cpu':
            if floating_model_cpu[worker_index]:
                input_data = (np.float32(input_data) - input_mean_cpu[worker_index]) / input_std_cpu[worker_index]
        #tpu
        else:
            if floating_model_tpu:
                input_data = (np.float32(input_data) - input_mean_tpu) / input_std_tpu
  
    #end image preprocessing
    elapsed_image_preprocessing = datetime.datetime.now(datetime.timezone.utc).astimezone().timestamp() - start


    who_executed = ''

    #[Detection]
    start = datetime.datetime.now(datetime.timezone.utc).astimezone().timestamp()
    #gpu adds new detection in the new repeat to the previous detectted objects while cpu and tpu do independent detections. That is GPU can find new objects in new iterations.
    detected_objects = []
    if MODEL_RUN_ON == 'cpu':
        img_cuda, input_details_tpu,labels_gpu,output_details_tpu,boxes_idx_tpu,classes_idx_tpu,scores_idx_tpu,labels_tpu= None,None,None,None,None,None,None,None
        detected_objects, who_executed,inference_dur_first,inference_dur_second_to_last = inference.run_inference(MODEL_INFERENCE_REPEAT,MODEL_RUN_ON,interpreter_worker,img_cuda,worker_index,input_details_cpu,input_data,input_details_tpu,labels_gpu,MODEL_MIN_CONFIDENCE_THRESHOLD,output_details_cpu,boxes_idx_cpu,classes_idx_cpu,scores_idx_cpu,output_details_tpu,boxes_idx_tpu,classes_idx_tpu,scores_idx_tpu,labels_cpu,labels_tpu)
    elif MODEL_RUN_ON == 'tpu':
        img_cuda,input_details_cpu,labels_gpu,output_details_cpu,boxes_idx_cpu,classes_idx_cpu,scores_idx_cpu,labels_cpu=None,None,None,None,None,None,None,None,
        detected_objects, who_executed,inference_dur_first,inference_dur_second_to_last = inference.run_inference(MODEL_INFERENCE_REPEAT,MODEL_RUN_ON,interpreter_worker,img_cuda,worker_index,input_details_cpu,input_data,input_details_tpu,labels_gpu,MODEL_MIN_CONFIDENCE_THRESHOLD,output_details_cpu,boxes_idx_cpu,classes_idx_cpu,scores_idx_cpu,output_details_tpu,boxes_idx_tpu,classes_idx_tpu,scores_idx_tpu,labels_cpu,labels_tpu)
    elif MODEL_RUN_ON == 'gpu':
        worker_index,input_details_cpu,input_data,input_details_tpu,output_details_cpu,boxes_idx_cpu,classes_idx_cpu,scores_idx_cpu,output_details_tpu,boxes_idx_tpu,classes_idx_tpu,scores_idx_tpu,labels_cpu,labels_tpu = None,None,None,None,None,None,None,None,None,None,None,None,None,None,
        detected_objects, who_executed,inference_dur_first,inference_dur_second_to_last = inference.run_inference(MODEL_INFERENCE_REPEAT,MODEL_RUN_ON,interpreter_worker,img_cuda,worker_index,input_details_cpu,input_data,input_details_tpu,labels_gpu,MODEL_MIN_CONFIDENCE_THRESHOLD,output_details_cpu,boxes_idx_cpu,classes_idx_cpu,scores_idx_cpu,output_details_tpu,boxes_idx_tpu,classes_idx_tpu,scores_idx_tpu,labels_cpu,labels_tpu)
    else:
        print(f'ERROR MODEL_RUN_ON={MODEL_RUN_ON}')
    

    inference_dur_second_to_last_avg = inference_dur_second_to_last/(int(MODEL_INFERENCE_REPEAT)-1) if int(MODEL_INFERENCE_REPEAT)>1 else inference_dur_first
    #inference total duration
    elapsed_inference = datetime.datetime.now(datetime.timezone.utc).astimezone().timestamp() - start
    # print('%.1fms total inference' % (elapsed_inference * 1000), file=sys.stdout)
    elapsed_total = datetime.datetime.now(datetime.timezone.utc).astimezone().timestamp() - start_main

    #get kubernetes service ip and port
    #if replica is created before service object, KUBERNETES_SERVICE_HOST and KUBERNETES_SERVICE_PORT are set
    #otherwise, <DEPLOYMENT_NAME>_SERVICE_PORT and <DEPLOYMENT_NAME>_SERVICE_HOST are set where <DEPLOYMENT_NAME> must be obtained already.
    KUBERNETES_SERVICE_IP = os.getenv("KUBERNETES_SERVICE_HOST", os.getenv(os.getenv("DEPLOYMENT_NAME") + "_SERVICE_HOST" if os.getenv("DEPLOYMENT_NAME") else "", None))
    KUBERNETES_SERVICE_PORT = os.getenv("KUBERNETES_SERVICE_PORT", os.getenv(os.getenv("DEPLOYMENT_NAME") + "_SERVICE_PORT" if os.getenv("DEPLOYMENT_NAME") else "", None))

    #HTTP headers are case insensitive
    # OpenFaaS already creates a X-Duration-Seconds header which means the round trip
    headers = {
        "X-Counter": str(counter),
        "Sensor-ID": str(request.headers.get('Sensor-ID')),
        "X-Elapsed-Time": str(elapsed_total),
        "Start-Time": str(start_main),
        "X-Start-Time": str(start_main),
        "X-Image-Config-Fetch-Time": str(elapsed_config),
        "X-Image-Fetch-Time": str(elapsed_fetch_image),
        "X-Processing-Time": str(elapsed_inference),
        "X-Processing-First-Inference-Time": str(inference_dur_first),
        "X-Processing-Second-To-Last-Inference-Avg-Time": str(inference_dur_second_to_last_avg),
        "X-Image-Preprocessing-Time": str(elapsed_image_preprocessing),
        "X-Load-Model-Time": str(elapsed_load_model),
        "X-Worker-Name": socket.gethostname(),
        "X-Worker-Ip": socket.gethostbyname(socket.gethostname()),
        "X-MODEL_DIR": str(MODEL_DIR),
        "X-MODEL_CPU_FILE": str(MODEL_CPU_FILE),
        "X-MODEL_TPU_FILE": str(MODEL_TPU_FILE),
        "X-MODEL_LABEL_FILE": str(MODEL_LABEL_FILE),
        "X-MODEL_IMAGE_GET": str(MODEL_IMAGE_GET),
        "X-MODEL_IMAGE_DIR": str(MODEL_IMAGE_DIR),
        "X-MODEL_IMAGE_SAMPLE1": str(MODEL_IMAGE_SAMPLE1),
        "X-MODEL_MIN_CONFIDENCE_THRESHOLD": str(MODEL_MIN_CONFIDENCE_THRESHOLD),
        "X-MODEL_INFERENCE_REPEAT": str(MODEL_INFERENCE_REPEAT),
        "X-MODEL_RUN_ON": str(MODEL_RUN_ON),
        "X-WHO_EXECUTED": str(who_executed),
        "X-KUBERNETES_DEPLOYMENT_NAME": os.getenv("DEPLOYMENT_NAME", None),
        "X-KUBERNETES_SERVICE_IP": KUBERNETES_SERVICE_IP,
        "X-KUBERNETES_SERVICE_PORT": KUBERNETES_SERVICE_PORT,
        "X-KUBERNETES_DEPLOYMENT_NAME": os.getenv("DEPLOYMENT_NAME", None),
        "X-NODE-NAME": os.getenv("NODE_NAME", None),
        "X-POD-NAME": os.getenv("POD_NAME", None),
        "X-POD-NAMESPACE": os.getenv("POD_NAMESPACE", None),
        "X-POD-IP": os.getenv("POD_IP", None),
        "X-POD-IPS": os.getenv("POD_IPS", None),
        "X-POD-HOST-IP": os.getenv("POD_HOST_IP", None),
        "X-POD-UID": os.getenv("POD_UID", None),
    } 
    
    output={}
    output['detected_objects'] = detected_objects
    if request.headers.get('Header-Output'):
        output['Header-Output'] = headers

    response = make_response(json.dumps(output), 200, headers)
    response.mimetype = "application/json"
    if MODEL_RUN_ON == 'cpu':
        semaphore.release()
     
    #In case of async request, the response will be sent to the callback url given as header by queue-worker of OpenFaaS.
    return response, detected_objects, error


def help():
    msg = ''' App Help:
curl -X POST -i -F image_file=@./images/image1.jpg  http://localhost:5001/
MODEL_RUN_ON=cpu 
[{'object': 'dog', 'confidence': 75}, {'object': 'dog', 'confidence': 70}]
MODEL_RUN_ON=tpu 
[{'object': 'dog', 'confidence': 73}, {'object': 'dog', 'confidence': 70}]
MODEL_RUN_ON=gpu 
[{'object': 'dog', 'confidence': 87}, {'object': 'dog', 'confidence': 91}]
    '''
    return msg

print(help(), flush=True)
