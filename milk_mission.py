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
            cv2.imshow('img', frame)
            cv2.waitKey(1)      
            return point
            
    
    return [-1, -1, -1, -1]        

def loop(serial_port):
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
    upper_black = np.array([180, 255, 50])
    
    lower_blue = np.array([90, 80, 30])
    upper_blue = np.array([140, 255, 255])
    
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([100, 255, 255])
    
    
    TX_data_py2(serial_port, 21) # Head Down 60
    time.sleep(0.5)
    TX_data_py2(serial_port, 31)
    flag = False
    milk_flag = False
    drop_flag = False
    safeloc_flag = False
    flagcounter = 0
    count = 0
    areacount = 0
    
    f = open("./data/area.txt","r")
    
    area = f.readline()
    f.close()
    
    f3 = open("./data/color.txt","r")
    color = f3.readline()
    f3.close()
    
    print(area)
    print(color)
    
    if area == "dangerous":
        while True:
            #wait_receiving_exit()
    
            _,frame = cap.read()
            if not count_frame():
                continue
            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            
            
            if drop_flag is True:
                
                dan_mask = cv2.inRange(img_hsv, lower_black, upper_black)
                dan_count = len(img_hsv[np.where(dan_mask != 0)])
                TX_data_py2(serial_port, 51)
                time.sleep(2)
                
                
                print(dan_count)
                
                if dan_count < 20000:
                    areacount += 1
                    
                else:
                    continue
                    
                if areacount > 3:
                    TX_data_py2(serial_port, 53)
                    break
                    
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
            
            
            
            if milk_flag is True:
                if  loc > 170:
                    TX_data_py2(serial_port, 20) #Right
                
                    
                elif loc>10 and loc < 130:
                    TX_data_py2(serial_port, 15) #Left
                   
                
                elif loc>=130 and loc<=170:
                    TX_data_py2(serial_port, 45) #Milk Up
                    
                    drop_flag = True
                    continue
                    
                
                    
                
            if flag is False and milk_flag is False:
                if  loc > 180:
                    TX_data_py2(serial_port, 20)
                    
                
                        
                elif loc>10 and loc < 140:
                    TX_data_py2(serial_port, 15)
                    
             
                
                elif loc>=140 and loc<=180:
                    flag = True
                    TX_data_py2(serial_port, 29) #Head Down 80   
                    
                elif loc < 0 :
                    TX_data_py2(serial_port, 47)
                    
                    
                


            if flag is True and milk_flag is False:
                #time.sleep(0.2)
                print(y + h)
                print(flagcounter)
                if flagcounter > 2:
                    milk_flag = True
                    
                    
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
                
                    elif loc< 0 :
                        TX_data_py2(serial_port, 47)
   
            time.sleep(1) 
            

    if area == "safe":
        while True:
            _,frame = cap.read()
            if not count_frame():
                continue
            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            if safeloc_flag is True:
                while True:
                    #wait_receiving_exit()
                    TX_data_py2(serial_port, 30)
                    dan_mask = cv2.inRange(img_hsv, lower_green, upper_green)
                    dan_count = len(img_hsv[np.where(dan_mask != 0)])
                    time.sleep(2)
       
                    print(dan_count)
                    if dan_count > 100 :
                        safeloc = "right"
                        print(safeloc)
                 
                        safeloc_flag = False
                        drop_flag = True
                        milk_flag = False
                        
                        TX_data_py2(serial_port, 21)
                        time.sleep(2)
                        TX_data_py2(serial_port, 31)
                        time.sleep(2)
                        TX_data_py2(serial_port, 45) #Milk Up
                    
                        break
                    else:
                        safeloc = "left"
                        break
       
       
            if drop_flag is True:
                
                dan_mask = cv2.inRange(img_hsv, lower_green, upper_green)
                dan_count = len(img_hsv[np.where(dan_mask != 0)])
                
                if safeloc == "right":
                    TX_data_py2(serial_port, 52) #Right
                
                    
                elif safeloc == "left":
                    TX_data_py2(serial_port, 50) #Left
                   
                
                time.sleep(2)
                print(dan_count)
                cv2.imshow('img', frame)
                cv2.waitKey(1)
                
                if dan_count > 5000 and dan_count < 12000:
                    areacount += 1
                    
 
                if areacount > 3:
                    TX_data_py2(serial_port, 53)
                    cap.release()
                    cv2.destroyAllWindows()

                    time.sleep(1)
                    exit(1)
                    break
                    
                continue
                    
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
            
            
            
            if milk_flag is True:
                if  loc > 170:
                    TX_data_py2(serial_port, 20) #Right
                
                    
                elif loc>10 and loc < 130:
                    TX_data_py2(serial_port, 15) #Left
                   
                
                elif loc>=130 and loc<=170:
                    
                    #TX_data_py2(serial_port, 54) 
                    safeloc_flag = True
                    continue
                    
                
                    
                
            if flag is False and milk_flag is False:
                
                
                
                if  loc > 180:
                    TX_data_py2(serial_port, 20)
                    
                
                        
                elif loc>10 and loc < 140:
                    TX_data_py2(serial_port, 15)
                    
             
                
                elif loc>=140 and loc<=180:
                    flag = True
                    TX_data_py2(serial_port, 29) #Head Down 80   
                    
                
                    
                


            if flag is True and milk_flag is False:
                #time.sleep(0.2)
                print(y + h)
                print(flagcounter)
                if flagcounter > 1:
                    milk_flag = True
                    
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
                
                    elif loc< 0 :
                        TX_data_py2(serial_port, 47)
   
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
   
