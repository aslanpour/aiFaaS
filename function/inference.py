import datetime
import sys

def run_inference(MODEL_INFERENCE_REPEAT,MODEL_RUN_ON,interpreter_worker,img_cuda,worker_index,input_details_cpu,input_data,input_details_tpu,labels_gpu,MODEL_MIN_CONFIDENCE_THRESHOLD,output_details_cpu,boxes_idx_cpu,classes_idx_cpu,scores_idx_cpu,output_details_tpu,boxes_idx_tpu,classes_idx_tpu,scores_idx_tpu,labels_cpu,labels_tpu):
    who_executed=''
    detected_objects=[]
    #first inference may take longer due to loading the model (is not the case here) and/or enabled DVFS.
    inference_dur_first = 0
    inference_dur_second_to_last = 0
    for i in range(int(MODEL_INFERENCE_REPEAT)):
        start_rep = datetime.datetime.now(datetime.timezone.utc).astimezone().timestamp()

        # Perform the actual detection by running the model with the image as input
        # with lock:

        if MODEL_RUN_ON == 'gpu':
            #gpu
            #Detect
            detections = interpreter_worker.Detect(img_cuda)
            who_executed = 'gpu'

        else: #cpu or tpu
            if MODEL_RUN_ON == 'cpu': interpreter_worker[worker_index].set_tensor(input_details_cpu[worker_index][0]['index'] ,input_data)
            if MODEL_RUN_ON == 'tpu': interpreter_worker.set_tensor(input_details_tpu[0]['index'] ,input_data)
            #invoke
            if MODEL_RUN_ON == 'cpu': 
                interpreter_worker[worker_index].invoke()
                who_executed = 'cpu'
            if MODEL_RUN_ON == 'tpu': 
                interpreter_worker.invoke()
                who_executed = 'tpu'

        elapsed_rep = datetime.datetime.now(datetime.timezone.utc).astimezone().timestamp() - start_rep
        print('%.1fms' % (elapsed_rep * 1000), file=sys.stdout)
        if i == 0:
            inference_dur_first = elapsed_rep
        else:
            inference_dur_second_to_last += elapsed_rep


        # Retrieve detection results
        #gpu
        if MODEL_RUN_ON == 'gpu':
            #get objects
            for detection in detections:
                #print(detection)
                object_name = labels_gpu[detection.ClassID]
                confidence = detection.Confidence

                #filter
                if ((confidence > float(MODEL_MIN_CONFIDENCE_THRESHOLD)) and (confidence <= 1.0)):
                    #other obtained values by detection are Left, Top, Right, Bottom, Width, Height, Area and Center
                    detected_objects.append({"object": object_name, "confidence": int(confidence *100)})
                    print({"object": object_name, "confidence": int(confidence *100)})
            #print("detect............")
            #for object in detected_objects:
            #    print(object)
        #cpu
        elif MODEL_RUN_ON == 'cpu':
            boxes = interpreter_worker[worker_index].get_tensor(output_details_cpu[worker_index][boxes_idx_cpu[worker_index]]['index'])
            # Bounding box coordinates of detected objects
            classes = interpreter_worker[worker_index].get_tensor(output_details_cpu[worker_index][classes_idx_cpu[worker_index]]['index'])[0] # Class index of detected objects
            scores = interpreter_worker[worker_index].get_tensor(output_details_cpu[worker_index][scores_idx_cpu[worker_index]]['index'])[0] # Confidence of detected objects
        #tpu
        elif MODEL_RUN_ON == 'tpu':
            boxes = interpreter_worker.get_tensor(output_details_tpu[boxes_idx_tpu]['index'])
            # Bounding box coordinates of detected objects
            classes = interpreter_worker.get_tensor(output_details_tpu[classes_idx_tpu]['index'])[0] # Class index of detected objects
            scores = interpreter_worker.get_tensor(output_details_tpu[scores_idx_tpu]['index'])[0] # Confidence of detected objects
        else:
            pass

        #filter (for cpu and tpu)
        if MODEL_RUN_ON == 'cpu' or MODEL_RUN_ON == 'tpu':
            #detected_objects = []
            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > float(MODEL_MIN_CONFIDENCE_THRESHOLD)) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    # ymin = int(max(1,(boxes[i][0] * imH)))
                    # xmin = int(max(1,(boxes[i][1] * imW)))
                    # ymax = int(min(imH,(boxes[i][2] * imH)))
                    # xmax = int(min(imW,(boxes[i][3] * imW)))

                    # cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    if MODEL_RUN_ON == 'cpu':
                        object_name = labels_cpu[worker_index][int(classes[i])] # Look up object name from "labels" array using class index
                    else:
                        object_name = labels_tpu[int(classes[i])] # Look up object name from "labels" array using class index
                    #label = '%s: %d%%' % (object_name, int(scores[i ] *100)) # Example: 'person: 72%'
                    #print(label)
                    
                    detected_objects.append({"object": object_name, "confidence": int(scores[i] *100)})
                    print({"object": object_name, "confidence": int(scores[i] *100)})
    return detected_objects, who_executed,inference_dur_first,inference_dur_second_to_last
    #end inference loop