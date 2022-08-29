import requests
supported_resources_gpu
res = requests.post('http://localhost:5000/config/write', json={"config":{"Model": {"supported_resources_gpu": "gpu"}}})
res = requests.post('http://localhost:5000/config/write', json={"config":{"Model": {"run_on": "gpu"}}})
res = requests.post('http://localhost:5000/config/write', json={"config":{"Model": {"inference_repeat": "10"}}})

res = requests.post('http://localhost:5000/config/read', json={})

if res.ok:
    print(res.json())

'''
#read
curl -X GET -i  http://localhost:5000/config/read

#runs by default on cpu --- otherwise preload TPU and/or GPU models and set their interpreter by setting env variables for the container (e.g., e.g., MODEL_SUPPORTED_RESOURCES_TPU=yes) and/or setting them as the model in use (e.g., MODEL_RUN_ON=tpu)
curl -X POST -i -F  image_file=@./images/image1.jpg  http://localhost:5000/

#on Jetson Nano (test for GPU use)
#this node has GPU (but just merely this setting, nothing will happen -- this is a prerequisitic for using another resource)
curl -X POST -i http://localhost:5000/config/write  -H 'Content-Type: application/json' -H 'Accept: application/json' -d '{"config":{"Model": {"supported_resources_gpu": "yes"}}}'
#now on, run on GPU
curl -X POST -i http://localhost:5000/config/write  -H 'Content-Type: application/json' -H 'Accept: application/json' -d '{"config":{"Model": {"run_on": "gpu"}}}'
#same for function on tpu
curl -X POST -i 10.0.0.90:31112/function/w5-ssd/config/write  -H 'Content-Type: application/json' -H 'Accept: application/json' -d '{"config":{"Model": {"run_on": "tpu"}}}'
#test GPU
curl -X POST -i -F  image_file=@./images/image1.jpg  http://localhost:5000/
#Note: convert curl commands in python: https://curlconverter.com/


#on Kuberentes (test for TPU access)
#create a Deployment as deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ssd-tpu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ssd-app
  template:
    metadata:
      labels:
        app: ssd-app
    spec:
      nodeName: w5
      containers:
      - name: ssd-tpu
        image: aslanpour/ssd:cpu-tpu
        #IfNotPresent or Never
        imagePullPolicy: Always
        securityContext:
          privileged: true
        volumeMounts:
        - mountPath: /dev/bus/usb
          name: usb-devices
      volumes:
      - name: usb-devices
        hostPath:
          path: /dev/bus/usb

kubectl apply -f deployment.yaml

#patch, solution 1
#Grant --privileged
kubectl patch deployment ssd-tpu --patch '{"spec": {"template": {"spec": {"containers": [{"name": "ssd-tpu","image": "aslanpour/ssd:cpu-tpu", "securityContext": {"privileged": true}}]}}}}'
#run as root
kubectl patch deployment ssd-tpu --patch '{"spec": {"template": {"spec": {"containers": [{"name": "ssd-tpu","image": "aslanpour/ssd:cpu-tpu", "securityContext": {"runAsUser": 0}}]}}}}'
#Mount -v /dev/bus/usb:    of the host
kubectl patch deployment ssd-tpu --patch '{"spec": {"template": {"spec": {"volumes": [{"name": "usb-devices", "hostPath": {"path": "/dev/bus/usb"}}]}}}}'
# to /dev/bus/usb of the container
kubectl patch deployment ssd-tpu --patch '{"spec": {"template": {"spec": {"containers": [{"name": "ssd-tpu", "volumeMounts":[{"mountPath": "/dev/bus/usb", "name": "usb-devices"}]}]}}}}'

#all together at once
kubectl patch deployment ssd-tpu --patch '{"spec": {"template": {"spec": {"containers": [{"name": "ssd-tpu","image": "aslanpour/ssd:cpu-tpu", "securityContext": {"privileged": true, "runAsUser": 0}, "volumeMounts": [{"mountPath": "/dev/bus/usb", "name": "usb-devices"}]}], "volumes": [{"name": "usb-devices", "hostPath": {"path": "/dev/bus/usb"}}]}}}}'
#Using curl
https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.24/#patch-deployment-v1-apps


#in OpenFaaS function (add -n openfaas-fn), as well as the above, you have to enable allowPrivilegeEscalation which is flase for function in OpenFaaS by default; otherwise, it conflicts with privilged.
kubectl patch deployment ssd-tpu --patch '{"spec": {"template": {"spec": {"containers": [{"name": "ssd-tpu","image": "aslanpour/ssd:cpu-tpu", "securityContext": {"allowPrivilegeEscalation": true}}]}}}}'
#all together becomes:
kubectl patch deployment ssd-tpu -n openfaas-fn --patch '{"spec": {"template": {"spec": {"containers": [{"name": "ssd-tpu","image": "aslanpour/ssd:cpu-tpu", "securityContext": {"privileged": true, "runAsUser": 0, "allowPrivilegeEscalation": true}, "volumeMounts": [{"mountPath": "/dev/bus/usb", "name": "usb-devices"}]}], "volumes": [{"name": "usb-devices", "hostPath": {"path": "/dev/bus/usb"}}]}}}}'


#patch, solution 2
#create a patch file as patch-file.yaml
k patch deployment ssd-tpu --type merge --patch-file patch-file.yaml

'''



'''scheduler a pod
#first terminal
kubectl proxy --port=8080 &

#another terminal
SERVER='localhost:8080'
NODENAME="w5"

k apply -f pod.yaml

apiVersion: v1
kind: Pod
metadata:
  name: nginxw5
spec:
  nodeName: silly
  containers:
  - name: nvidia
    image: nginx:1.14.2

PODNAME=$(kubectl --server $SERVER get pods -o json | jq '.items[] | select(.spec.nodeName == "silly") | .metadata.name')

 curl --header "Content-Type:application/json" --request POST --data '{"apiVersion":"v1", "kind": "Binding", "metadata": {"name": "'$PODNAME'"}, "target": {"apiVersion": "v1", "kind": "Node", "name": "'$NODENAME'"}}' http://$SERVER/api/v1/namespaces/default/pods/$PODNAME/binding/