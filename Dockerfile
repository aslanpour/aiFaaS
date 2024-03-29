#[Guide]
#Build command:
#docker build --build-arg CACHEBUST=$(date +%s) --tag image_name:image_tag .

#Sample (build for GPU execution on ARM platforms)
#docker build --build-arg CACHEBUST=$(date +%s) --build-arg BASEIMAGE=dustynv/jetson-inference:r32.7.1 --build-arg TARGETPLATFORM=linux/arm64   --tag image_name:image_tag .

#Notes: 
#After 'CACHEBUST' everything will be reloaded to avoid caching.

#Run command:
#docker run -d -t <docker_image_name>

#Notes:
#Add '--user root' option to use TPU or GPU for root privileged permission (insecure)
#In kubernetes, equivalent of --root is "runAsUser: 0" under securityContext.
#Add '--runtime nvidia' option to enable nvidia runtime on host level or container level if not enabled on host level already.
#Add '--privileged' option for tpu access (and for first use of TPU on a node, use "-v /dev/bus/usb:/dev/bus/usb"). 
#In kuberneets equivalent of --privileged is privilged: true under securityContext.
#Inkubernetes equivalent of -v /dev/bus/usb:/dev/bus/usb is to dfine volumes.
#Add '--env <VARIABLE>' to configure the app, like '--env MODEL_PRE_LOAD=yes --env MODEL_RUN_ON=gpu' to preload models and run on GPU

#Sample run: 
#docker run -d -t -p 5000:5000 --privileged --user root -v /dev/bus/usb:/dev/bus/usb --rm --name container_name <docker_image_name>


#--------------SET ARGUMENTS------------------------------------------
#[Global ARGs] - available in all FROMs but need recall with just ARG ARG_NAME (no default value) inside the scope (after the corresponding FROM) to be reusable in multi-stage builds.
#Ref. https://stackoverflow.com/a/61258832/14167325
#BASEIMAGE defaults to the cpu base image: python:3.7-slim-buster that also works for TPU the image. 
#Base image for GPU is dustynv/jetson-inference:r32.7.1
#Note: GPU image tag is assocciated to the Jetson Nano L4T version, obtain yours by cat /etc/nv_tegra_release and get relevant image from https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-docker.md#running-the-docker-container
ARG BASEIMAGE='python:3.7-slim-buster'
#valid values='linux/amd64' and 'linux/arm64'
#Note: if both --platform and --build-arg TARGETPLATFORM are set, the latter takes precedence over the former.
ARG TARGETPLATFORM='linux/amd64'
ARG APP_PORT='5000'
ARG ADDITIONAL_PACKAGE

#--------------BASE 1 as watchdog: OpenFaaS Agent------------------------------------------

#[Base Image]
#Set OpenFaaS watchdog base image
FROM --platform=${TARGETPLATFORM} ghcr.io/openfaas/of-watchdog:0.9.12 as watchdog

#-----------------BASE 2 as base: System tools, Python tools, and App data---------------------------------------

#Set the base image
FROM --platform=${TARGETPLATFORM} ${BASEIMAGE} as base

ARG BASEIMAGE

#[User]
# Add non root user
# RUN addgroup -S app && adduser app -S -G app
RUN addgroup --system app && adduser app --system --ingroup app
RUN chown app /home/app

USER app
ENV PATH=$PATH:/home/app/.local/bin


#------------------System TOOLS--------------------
ARG ADDITIONAL_PACKAGE

USER root

#Note: usbutils is for lsusb command that gets USB info, but this does not show Product info like Google Inc. in the container that is the name of Google TPU Coral, although it does in the host, so udevadm is installed by udev package to update usb info which is a known issue in some cases: Ref: https://www.suse.com/support/kb/doc/?id=000017623)

# Fixing GPG error NO_PUPKEY for dustynv/jetson-inference:r32.7.1
RUN if [ "$BASEIMAGE" = "dustynv/jetson-inference:r32.7.1" ]; then \
    # Command to execute if BASEIMAGE matches
    echo "Fixing GPG error NO_PUBKEY for dustynv/jetson-inference:r32.7.1"; \
    #Ref. https://askubuntu.com/questions/13065/how-do-i-fix-the-gpg-error-no-pubkey
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 42D5A192B819C5DA; \
else \
    # Command to execute if BASEIMAGE does not match
    echo "BASEIMAGE is not dustynv/jetson-inference:r32.7.1. Skipping GPG fix command."; \
fi

RUN apt-get -qy update && apt-get install -y git curl wget nano gnupg2 ca-certificates unzip tar usbutils udev openssl nginx net-tools tree ${ADDITIONAL_PACKAGE}
#NOTE: TPU connection still fails with this error: lsusb: cannot open "/var/lib/usbutils/usb.ids", No such file or directory
RUN udevadm trigger --subsystem-match=usb
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

#install standard TPU (or with maximum frequency 'libedgetpu1-max')
RUN apt-get -qy update && apt-get install -y libedgetpu1-std
RUN  apt-get update -y && apt-get install -y python3-pycoral

#----------APP TOOLS----------------

COPY requirements.txt   .

RUN apt-get update -y
#Note:Tensorflow lite examples require protobuf>=3.18.0,<4, but not sure if not practising that will cause an issue. Ref: #Ref: https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/requirements.txt
RUN python3 -m pip install --upgrade pip && python3 -m pip install --user -r requirements.txt
RUN python3 -m pip install --user --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
#Note1 for pycoral: Although Pycoral package is installed, it might not be recognized, so it is reinstalled by the above command (https://coral.ai/software/#pycoral-api) 
#according to the discussion in here: https://github.com/google-coral/pycoral/issues/24. If this also did not work, build the wheel, 
#example: https://blogs.sap.com/2020/02/11/containerizing-a-tensorflow-lite-edge-tpu-ml-application-with-hardware-access-on-raspbian/
#Note2: by installing pycoral package, tflite-runtime and setuptools are also installed.
#Note3: Tensorflow Lite examples require tflite-support==0.4.0 to be installed but their code also works with tflite-runtime. Ref: #Ref: https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/requirements.txt
#Note4: connect TPU to USB 3; otherwise,  inferencing times actually increase by a factor of ×2 if mounted to USB 2. Ref https://www.hackster.io/news/benchmarking-the-intel-neural-compute-stick-on-the-new-raspberry-pi-4-model-b-e419393f2f97
#Note5: Original USB cable of Coral TPU wont probably work on USB 2 and has limitations in speed (5Gbps); to use USB 3 or better speed, (at least mount TPU to USB 3) use 10Gbps cables. Ref:https://github.com/tensorflow/tensorflow/issues/32743 and https://www.hackster.io/news/benchmarking-the-intel-neural-compute-stick-on-the-new-raspberry-pi-4-model-b-e419393f2f97
#Note6: For bare metal use, if ValueError: Failed to load delegate from libedgetpu.so.1, reboot the host. Ref: https://github.com/tensorflow/tensorflow/issues/32743
#Note7: You may need to add your linux user to plugdev group. Ref: https://github.com/tensorflow/tensorflow/issues/32743. As follows: sudo usermod -aG plugdev [your username]

#expected installations: python3 -m pip list
#Installed versions
#numpy          1.21.6
#Pillow         9.2.0
#pip            22.2.1
#protobuf       4.21.4
#pycoral        2.0.0
#setuptools     57.5.0
#tflite-runtime 2.5.0.post1
#wheel          0.37.1

#----------APP DATA----------

USER app

#Download models

WORKDIR /home/app/
RUN mkdir networks
RUN mkdir networks/tensorflow
RUN mkdir networks/tensorflow-lite
RUN mkdir networks/tensorrt

#Tensorflow Lite models. ref: https://coral.ai/models/all/#detection
#Note: The python code (i.e., function) at the moment works only on model V1 and throws an error when using model V2 due to the difference in image preprocessing requirements.
#Model 1: SSD-MobileNet-V1-300-300-TF1-90obj.
WORKDIR /home/app/networks/tensorflow-lite/
RUN mkdir SSD-MobileNet-V1-300-300-TF1-90obj
WORKDIR /home/app/networks/tensorflow-lite/SSD-MobileNet-V1-300-300-TF1-90obj/
#CPU model
RUN wget --content-disposition https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v1_coco_quant_postprocess.tflite -O model.cpu.tflite
#TPU model
RUN wget --content-disposition https://raw.githubusercontent.com/google-coral/test_data/master/ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite -O model.edgetpu.tflite
#Labels
RUN wget --content-disposition https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt -O labelmap.txt
#GPU model
#TensorRT models. ref: #links in jetson-inference/tools/downlaod-models.sh or this link https://github.com/dusty-nv/jetson-inference/blob/master/tools/download-models.sh
#Since TensorRT (jetson-inference) only recognizes ./networks/SSD-Mobilenet-v1/.. for model file location, we need to put GPU model files under /home/app/function/networks dir. 
#But, since handler.py is called from index.py which is in /home/app/ dir, then tensorrt models must locate in /home/app/networks/ dir.
#If one needs a different location, you may edit jetson-inference and rebuild it as suggested here https://forums.developer.nvidia.com/t/how-to-load-models-from-a-custom-directory/223016

WORKDIR /home/app/networks/
#Model 1: SSD-Mobilenet-v1
RUN wget https://nvidia.box.com/shared/static/0pg3xi9opwio65df14rdgrtw40ivbk1o.gz -O SSD-Mobilenet-v1.tar.gz
#Note: in some cases, downloading from box.com is blocked. In that case, use the below mirror link
RUN wget https://github.com/dusty-nv/jetson-inference/releases/download/model-mirror-190618/SSD-Mobilenet-v1.tar.gz -O SSD-Mobilenet-v1.tar.gz
RUN tar -xvzf SSD-Mobilenet-v1.tar.gz
RUN rm -rf SSD-Mobilenet-v1.tar.gz
#This results in a new directory as SSD-Mobilenet-v1 that contains ssd_mobilenet_v1_coco.uff and ssd_coco_labels.txt
#upon the first inference use, TensorRT creates an engine that optimizes the inferences. 
#This may take a while for the first run. To avoid this delay, we copy here a prebuilt engine for this model.
#Copy the engine prebuilt model to avoid first run delay. If you do not have the engine file, run test_gpu_detection.py on the host to create one.
COPY ./networks/SSD-Mobilenet-v1/ssd_mobilenet_v1_coco.uff.1.1.8201.GPU.FP16.engine ./SSD-Mobilenet-v1/
#More models for Tensorflow Lite: (the models may require images to undergo a particular preprocessing)
#Model of EdjeElectronics (CPU/TPU) work on the python code already. Ref: from git clone https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi.git
#Google Models Lite CPU&TPU  https://coral.ai/models/all/
#More models for Tensorflow:
#From NVIDIA https://github.com/dusty-nv/jetson-inference/#object-detection where a label and .uff file is downloaded and in first execution TensorRT will create a GPU.FP16.engine file for itself.
#More models for Tensorflow and Tensorflow Lite:
#From Tensorflow Hub https://tfhub.dev/tensorflow/collections/object_detection/1

#Download test images
RUN mkdir -p /home/app/images
WORKDIR /home/app/images
RUN wget --content-disposition https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/image1.jpg -O image1.jpg
RUN wget --content-disposition https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/image2.jpg -O image2.jpg


#---------------------------------BUILD IMAGE--------------------------------

FROM --platform=${TARGETPLATFORM} ${BASEIMAGE} as builder

#Reuse args inside the scope
ARG BASEIMAGE
ARG TARGETPLATFORM
ARG APP_PORT

# Add non root user
# RUN addgroup -S app && adduser app -S -G app
RUN addgroup --system app && adduser app --system --ingroup app
RUN chown app /home/app

USER app
ENV PATH=$PATH:/home/app/.local/bin

#---------COPY APP DATA------------
WORKDIR /home/app/
COPY --from=base /home/app .
# RUN touch __init__.py

USER root
RUN chown -R app:app ../
#---------COPY PYTHON TOOLS-------------
USER root

COPY --from=base /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

#-------COPY SYSTEM TOOLS---------------
COPY --from=base /bin /bin 
COPY --from=base /usr/bin /usr/bin
COPY --from=base /lib /lib
# COPY --from=base /lib64 /lib64
COPY --from=base /usr/lib /usr/lib

#--------COPY OPENFAAS AGENT------------
COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog

USER root
RUN chown -R app:app *

#-----------------------Run Proxy Server--------------
# RUN service nginx start 
# RUN service nginx enable
# COPY --from=base /etc/nginx /etc/nginx
RUN apt-get update -y && apt-get install -y nginx 
RUN service nginx start
COPY nginx_proxy.conf /etc/nginx/sites-available/nginx_proxy.conf
RUN rm /etc/nginx/sites-enabled/default /etc/nginx/sites-available/default
#Create a symbolic link to enable the site
RUN ln -s /etc/nginx/sites-available/nginx_proxy.conf /etc/nginx/sites-enabled/
#Test Nginx
RUN nginx -t
# RUN service nginx restart 2>&1
COPY start.sh .
RUN chmod +x start.sh
#------------APP Code-----------------
#This needs a different value each time you build the image so it wont cache the application files and copies updated ones.
ARG CACHEBUST=1 

USER app

WORKDIR /home/app/

COPY index.py           .

#function files
RUN mkdir -p /home/app/function
COPY function/ /home/app/function/
# RUN touch __init__.py

USER root
RUN chown -R app:app ../
#--------------Wrap up----------------------
WORKDIR /home/app/

#Allow tpu delegate for the user or run container as root; otherwise, it receives this error ValueError: Failed to load delegate from libedgetpu.so.1.0. Ref: https://github.com/tensorflow/tensorflow/issues/32743#issuecomment-543806766
USER root
#RUN usermod -aG plugdev app

#[TPU]
#To use TPU delegate, add app to root group 
RUN usermod -G root app
#and grant it root privilege
RUN usermod -u 0 -g 0 -o app
USER app

#configure WSGI server and healthcheck

ENV fprocess="python index.py"

ENV cgi_headers="true"
ENV mode="http"
ENV upstream_url="http://127.0.0.1:${APP_PORT}"

HEALTHCHECK --interval=5s CMD [ -e /tmp/.lock ] || exit 1

LABEL org.opencontainers.image.source=https://github.com/aslanpour/aiFaaS
LABEL org.opencontainers.image.description="A Machine Learning Benchmark Tool Based on Flask for CPU, TPU, and GPU Runtimes, on Both X86 and ARM Platforms."

ENV BASEIMAGE=${BASEIMAGE}
ENV TARGETPLATFORM=${TARGETPLATFORM}
ENV APP_PORT=${APP_PORT}

#Inspection
RUN echo "BASEIMAGE=${BASEIMAGE}, TARGETPLATFORM=${TARGETPLATFORM}, APP_PORT=${APP_PORT}" >> info.txt
RUN cat info.txt
RUN tree .


# CMD ["fwatchdog"]
CMD ["/home/app/start.sh"]
# CMD ["sh", "-c", "nginx -g 'daemon off;' && fwatchdog "]

#others
#Code for CPU/TPU based on EdjeElectronics
#(ref. used for this code) EdjeElectronics: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md
#note that EdjeElectroincs instruction is for ARMv7 and exactly following that may face error on ARM64, so I follow my own instruction.
#Alternative ref1: Google https://coral.ai/docs/accelerator/get-started/#3-run-a-model-on-the-edge-tpu
#Alternative ref2: Tensorflow Lite official: https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi