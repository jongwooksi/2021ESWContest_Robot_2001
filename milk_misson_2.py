import numpy as np
import argparse
import cv2
import serial
import time
from serialdata import *

def preprocessing(frame):
	
    img = cv2.Canny(frame, 50, 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img = cv2.dilate(img, kernel, iterations=2)

    #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #img = cv2.erode(img, kernel)

    #cv2.imshow("daa", img)
    #key = cv2.waitKey(1)

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    for c in contours:
        
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        point = cv2.boundingRect(approx)
        
        area = point[2] * point[3]
        
        if area > 1000:
        
            cv2.rectangle(frame, (point[0], point[1]), (point[0] + point[2], point[1]+point[3]), (0, 255, 0), 1)  
            #cv2.imshow('img', frame)
            #cv2.waitKey(1)      
            return point
            
    
    return [-1, -1, -1, -1]        

def loop(serial_port):
    TX_data_py2(serial_port, 21) # Head Down 60
    time.sleep(1)
    TX_data_py2(serial_port, 31)
    
    W_View_size = 320
    H_View_size = int(W_View_size / 1.333)

    FPS         = 5  #PI CAMERA: 320 x 240 = MAX 90


    cap = cv2.VideoCapture(0)

    cap.set(3, W_View_size)
    cap.set(4, H_View_size)
    cap.set(5, FPS)
    #cap.set(cv2.CAP_PROP_BUFFERSIZE,0)
    
    
    
    lower_red = np.array([0, 120, 40])
    upper_red = np.array([20, 255, 255])
   

    lower_red2 = np.array([160, 120, 40])
    upper_red2 = np.array([180, 255, 255])
    
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 90])
    
    lower_blue = np.array([90, 80, 30])
    upper_blue = np.array([140, 255, 255])
    
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([100, 255, 255])
    

    flag = False
    milk_flag = False
    drop_flag = False
    safeloc_flag = False
    flagcounter = 0
    count = 0
    areacount = 0
    loc = -1
    f = open("./data/area.txt","r")
    
    area = f.readline()
    f.close()
    
    f3 = open("./data/color.txt","r")
    color = f3.readline()
    f3.close()
    
    f2 = open("./data/arrow.txt","r")
    direction = f2.readline()
    f2.close()
    
    print(area)
    print(color)
    
    stage = 0
    step = 0
    straight_step = 0
    first_step_flag = True
    if area == "dangerous": # 위험지역
        while True:
            #wait_receiving_exit()
    
            _,frame = cap.read()
            if not count_frame_333():
                continue
            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            if stage < 3:
                if color == "red":
                    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
                    mask1 = cv2.inRange(img_hsv, lower_red2, upper_red2)
                    red_mask = mask0 + mask1
                    image_result = cv2.bitwise_and(frame, frame,mask = red_mask)
                    #time.sleep(1)
                    
                    [x, y, w, h] = preprocessing(image_result)
                 
                 
                elif color == "blue":
                    
                    blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
                    image_result = cv2.bitwise_and(frame, frame,mask = blue_mask)
                    #time.sleep(1)
                    
                    [x, y, w, h] = preprocessing(image_result)
                    
                            
                print( x, y, x+w, y+h)
                loc = (x + x + w)/2
                print(loc)
            
            
            if stage == 0: 
                if direction == "right":
                    if first_step_flag is True:
                        for i in range(2):
                            TX_data_py2(serial_port, 20)
                        first_step_flag = False
                    time.sleep(1)
                    if loc >= 140 and loc <= 180:
                        print("stage:", stage)
                        stage = 1
                        TX_data_py2(serial_port, 29) #Head Down 80
                        continue
                    else:
                        TX_data_py2(serial_port, 20)
                        straight_step += 1
                elif direction == "left":
                    if first_step_flag is True:
                        for i in range(2):
                            TX_data_py2(serial_port, 15)
                        first_step_flag = False
                    time.sleep(1)
                    if loc >= 140 and loc <= 180:
                        print("stage:", stage)
                        stage = 1
                        TX_data_py2(serial_port, 29) #Head Down 80 
                        continue
                    else:
                        TX_data_py2(serial_port, 15)
                        straight_step += 1
                    
            elif stage == 1:
                print(y + h)
                print(flagcounter)
                if flagcounter > 2:
                    stage = 2
                    
                    
                if color == "red" and y + h > 200:
                    flagcounter += 1
       
                elif color == "blue" and y + h > 200:
                    flagcounter += 1
       
                    
                else :
                    if  loc > 180:
                        TX_data_py2(serial_port, 20)
                    elif loc>10 and loc < 140:
                        TX_data_py2(serial_port, 15)
                    elif loc>=140 and loc<=180:
                        TX_data_py2(serial_port, 47)
                        time.sleep(1)
                        step+= 1
                    elif loc< 0 :
                        TX_data_py2(serial_port, 47)
                        time.sleep(1)
   
            elif stage == 2:
                if  loc > 170:
                    TX_data_py2(serial_port, 20) #Right
                
                    
                elif loc>10 and loc < 130:
                    TX_data_py2(serial_port, 15) #Left
                   
                
                elif loc>=130 and loc<=170:
                 
                    TX_data_py2(serial_port, 45) #Milk Up
                    stage = 3
                    continue
   
            
            elif stage == 3:
             
                '''
                dan_mask = cv2.inRange(img_hsv, lower_black, upper_black)
                dan_count = len(img_hsv[np.where(dan_mask != 0)])
                TX_data_py2(serial_port, 51)
                time.sleep(3)
                
                print(dan_count)
                print("areacount: {}".format(areacount))
                
                if dan_count < 15000:
                    areacount += 1
                    time.sleep(1)
                '''
                step_count = (int(1.5*step)) // 2
                for i in range(step_count+1):
                    TX_data_py2(serial_port, 51)
                    time.sleep(1)
                TX_data_py2(serial_port, 53)
                time.sleep(1)
                if direction == "right":
                    TX_data_py2(serial_port, 48)
                elif direction == "left":
                    TX_data_py2(serial_port, 49)
                time.sleep(1)
                for i in range(int(straight_step * 0.6)):
                    TX_data_py2(serial_port, 47)
                    time.sleep(1)
                cap.release()
                cv2.destroyAllWindows()
                exit(1)  
            time.sleep(1) 
            

    area_zone = [0,0]
    
    if area == "safe": # 안전지역
        first_flag = True
        straight_step = 0
        while True:
            _,frame = cap.read()
            if not count_frame_333():
                continue
            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            if stage < 3:
                if color == "red":
                    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
                    mask1 = cv2.inRange(img_hsv, lower_red2, upper_red2)
                    red_mask = mask0 + mask1
                    image_result = cv2.bitwise_and(frame, frame,mask = red_mask)
                    #time.sleep(1)
                    
                    [x, y, w, h] = preprocessing(image_result)
                 
             
                elif color == "blue":
                
                    blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
                    image_result = cv2.bitwise_and(frame, frame,mask = blue_mask)
                    #time.sleep(1)
                    
                    [x, y, w, h] = preprocessing(image_result)
              
                            
                print( x, y, x+w, y+h)
                loc = (x + x + w)/2
                print(loc)
                
            if stage == 0:
                if first_flag == True:
                    print("x location ",x)
                    if x > 160:
                        safeloc = "left"
                        for i in range(2):
                            TX_data_py2(serial_port, 9)
                            time.sleep(1)
                    elif x < 160:
                        safeloc = "right"
                        for i in range(2):
                            TX_data_py2(serial_port, 7)
                            time.sleep(1)
                    first_flag = False
                    
                if  loc > 180:
                    TX_data_py2(serial_port, 20)
                elif loc>10 and loc < 140:
                    TX_data_py2(serial_port, 15)
                elif loc>=140 and loc<=180:
                    stage = 1
                    TX_data_py2(serial_port, 29) #Head Down 80   
                    
            
            elif stage == 1:
                 #time.sleep(0.2)
                print(y + h)
                print(flagcounter)
                if flagcounter == 0:
                    time.sleep(1)
                    
                if flagcounter > 1:
                    stage = 2
                    
                if color == "red" and y + h > 200:
                    flagcounter += 1
       
                elif color == "blue" and y + h > 200:
                    flagcounter += 1
                else :
                    if  loc > 180:
                        TX_data_py2(serial_port, 20)
                    elif loc>10 and loc < 140:
                        TX_data_py2(serial_port, 15)
                    elif loc>=140 and loc<=180:
                        TX_data_py2(serial_port, 47)
                        straight_step += 1
                    elif loc< 0 :
                        TX_data_py2(serial_port, 47)
                        straight_step += 1
   
            elif stage == 2:
                if  loc > 170:
                    TX_data_py2(serial_port, 20) #Right
                elif loc>10 and loc < 130:
                    TX_data_py2(serial_port, 15) #Left
                elif loc>=130 and loc<=170:
                    stage = 3
                    continue
                    
            elif stage == 3:
                TX_data_py2(serial_port, 21)
                time.sleep(2)
                TX_data_py2(serial_port, 29) #original 31
                time.sleep(2)
                TX_data_py2(serial_port, 45) #Milk Up
                stage = 4
                
            elif stage == 4:
                dan_mask = cv2.inRange(img_hsv, lower_green, upper_green)
                safe_count = len(img_hsv[np.where(dan_mask != 0)])
                
                if safeloc == "right":
                    TX_data_py2(serial_port, 52) #Right
                    step += 1
                elif safeloc == "left":
                    TX_data_py2(serial_port, 50) #Left
                    step += 1
                   
                time.sleep(2)
                print(safe_count)
                print("area count : ", areacount)
                cv2.imshow("frame", dan_mask)
                cv2.waitKey(1)
                if safe_count > 18000: # safe_count > 8000 and safe_count < 18000
                    areacount += 1
                    
                if areacount > 3:
                    TX_data_py2(serial_port, 53)
                    stage = 5
                continue
                    
            
            elif stage == 5 :
                print(straight_step)
                if direction == "right":
                    if safeloc == "right":
                        print("LOCATION", safeloc)
                        for i in range(step*2):
                            TX_data_py2(serial_port, 50) #Left
                            time.sleep(1)
                        TX_data_py2(serial_port, 48)
                        time.sleep(1)
                        for i in range(straight_step):
                            TX_data_py2(serial_port, 50)
                            time.sleep(1)
                    elif safeloc == "left":
                        print("LOCATION", safeloc)
                        for i in range(int(step*1.5)):
                            TX_data_py2(serial_port, 52) #Right
                            time.sleep(1)
                        for i in range(2):
                            TX_data_py2(serial_port, 49)
                            time.sleep(1)
                        for i in range(straight_step):
                            TX_data_py2(serial_port, 47)
                            time.sleep(1)
                elif direction == "left":
                    if safeloc == "right":
                        print("LOCATION", safeloc)
                        for i in range(step*2):
                            TX_data_py2(serial_port, 50) #Left
                            time.sleep(1)
                        for i in range(2):
                            TX_data_py2(serial_port, 48)
                            time.sleep(1)
                        for i in range(straight_step):
                            TX_data_py2(serial_port, 47)
                            time.sleep(1)
                    elif safeloc == "left":
                        print("LOCATION", safeloc)
                        for i in range(int(step*1.5)):
                            TX_data_py2(serial_port, 52) #Right
                            time.sleep(1)
                        TX_data_py2(serial_port, 49)
                        time.sleep(1)
                        for i in range(straight_step):
                            TX_data_py2(serial_port, 52)
                            time.sleep(1)
                

          
                cap.release()
                cv2.destroyAllWindows()

                time.sleep(1)
                exit(1)

            time.sleep(1)
            
     
    
    
    
if __name__ == '__main__':

    BPS =  4800  # 4800,9600,14400, 19200,28800, 57600, 115200

       
    serial_port = serial.Serial('/dev/ttyS0', BPS, timeout=0.01)
    serial_port.flush() # serial cls
    
    
    serial_t = Thread(target=Receiving, args=(serial_port,))
    serial_t.daemon = True
    
    
    serial_d = Thread(target=loop, args=(serial_port,))
    serial_d.daemon = True
    
    print("start")
    serial_t.start()
    serial_d.start()
    
    #serial_t.join()
    serial_d.join()
    print("end")
   
