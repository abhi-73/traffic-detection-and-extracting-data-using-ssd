import cv2 #using opencv library
import csv #using comma sperated variable

#we are using( Mobile ssd pretrained model)
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' #configuration file
frozen_model = 'frozen_inference_graph.pb' #pretrained graph file

model = cv2.dnn_DetectionModel(config_file,frozen_model)#using deep nueral network and loading config and weight file


fields=['Object_ID','Classification','Confidence','Box','Time']#initializing header files for csv
labels= [] #initilizing empty list
file_name = "Labels.txt"#reading coco dataset file

with open(file_name,'rt') as fpt:
    labels=fpt.read().rstrip('\n') .split('\n')

    file = open("Data extracted.csv", "w") #opening CSV file

    writer = csv.writer(file)#writer object
    writer.writerow(fields)#writing fields

model.setInputSize(320,320)#defining pixels for video 320x320
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

cap = cv2.VideoCapture("samp1.mp4")# reading frame by frame
fps =cap.get(cv2.CAP_PROP_FPS)#30fps
frame_count = 0

if not cap.isOpened(): #video is not opened we use webcame
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN


while True:
    ret,frame = cap.read()
    ClassIndex , confidece ,bbox =model.detect(frame,confThreshold=0.55)

    print(bbox)
    print(str(confidece))
    print(ClassIndex)
    frame_count=frame_count+1
    time = float(frame_count) / fps  #timestamp
    if len(ClassIndex)!=0:
        for ClassIndex , conf ,boxes in zip(ClassIndex.flatten() , confidece.flatten() ,bbox):
            if ClassIndex<80:
                cv2.rectangle(frame,boxes,(255,0,0) , 2) #defining box enclosing the object
                cv2.putText(frame, labels[ClassIndex-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0))#putting the text above the box
    cv2.imshow('Object detection tutorial',frame) #defining window
    for w in range(len(confidece)):
        writer.writerow([ClassIndex,labels[ClassIndex-1],(confidece[w]*100),bbox,time]) #writing the fields in excel sheet


    if cv2.waitKey(2) & 0xFF == ord('q'): #press q key to close the active window
        break
cap.release()#all frames are released
cv2.destroyAllWindows() #destroy the window
file.close() #close the excel sheet