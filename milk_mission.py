import numpy as np
import argparse
import cv2
import serial
import time
from serialdata import *


def grayscale(img): 
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold): 
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size): 
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    x, y, gradient =  draw_lines(line_img, lines)
    return line_img, x, y, gradient


def weighted_img(img, initial_img, a=1, b=1., c=0.):
    return cv2.addWeighted(initial_img, a, img, b, c)
 
 
def draw_lines(img, lines, color=[0, 0, 255], thickness=3):
   
    x=-1
    y=-1
    gradient=0
    maxvalue = 0
    point=[0,0,0,0]
    if lines is None:
        return x, y, gradient
   
    for line in lines:
        for x1,y1,x2,y2 in line:
                           
            if y2 < 120 and y1 < 120 :
                continue
            
             
            if maxvalue < max(y1, y2):
                maxvalue = max(y1, y2)
            
                gradient = (y2-y1)/(x2-x1+0.00001)
                
                x = max(x1, x2)
                point[0] = x1
                point[1] = y1
                point[2] = x2
                point[3] = y2
       
    if point[0] == 0 and point[1] == 0:
         for line in lines:
            for x1,y1,x2,y2 in line:
                               
                
                 
                if maxvalue < max(y1, y2):
                    maxvalue = max(y1, y2)
                
                    gradient = (y2-y1)/(x2-x1+0.00001)
                    
                    x = max(x1, x2)
                    point[0] = x1
                    point[1] = y1
                    point[2] = x2
                    point[3] = y2
            
        
        
    cv2.line(img, (point[0], point[1]), (point[2], point[3]), color, thickness)
            
    return x, y, gradient


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
    W_View_size = 320
    H_View_size = int(W_View_size / 1.333)

    FPS         = 5  #PI CAMERA: 320 x 240 = MAX 90


    cap = cv2.VideoCapture(0)

    cap.set(3, W_View_size)
    cap.set(4, H_View_size)
    cap.set(5, FPS)
  
    
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
    loc = -1
    area_zone = [0,0]
    locStepLeftRight = 0
    
    
    f = open("./data/area.txt","r")
    
    area = f.readline()
    f.close()
    
    f3 = open("./data/color.txt","r")
    color = f3.readline()
    f3.close()
    
    print(area)
    print(color)
    
    stage = -1
    step = 0
    stepCountList = [0,0,0]
    rectangle_count = 0
    head_flag = 0
 
    if area == "dangerous":
        while True:
            _,frame = cap.read()
            
            if not count_frame():
                continue
            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            
            
            if stage < 3:
                if color == "red":
                    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
                    mask1 = cv2.inRange(img_hsv, lower_red2, upper_red2)
                    red_mask = mask0 + mask1
                    image_result = cv2.bitwise_and(frame, frame,mask = red_mask)
 
                    [x, y, w, h] = preprocessing(image_result)
                 
                 
                elif color == "blue":
                    
                    blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
                    image_result = cv2.bitwise_and(frame, frame,mask = blue_mask)
                    
                    [x, y, w, h] = preprocessing(image_result)
                    
                            
                print( x, y, x+w, y+h)
                loc = (x + x + w)/2
                print(loc)
            
            if stage == 0:
                if  loc > 180:
                    TX_data_py2(serial_port, 20)
                    time.sleep(1)
                    stepCountList[1] += 1
                
                        
                elif loc>10 and loc < 140:
                    TX_data_py2(serial_port, 15)
                    time.sleep(1)
                    stepCountList[0] += 1
                
                
                elif loc>=140 and loc<=180:
                    stage = 1
                    TX_data_py2(serial_port, 29) #Head Down 80   
                    continue
                    
                elif loc < 0 :
                    TX_data_py2(serial_port, 47)
                    time.sleep(1)
                    stepCountList[2] += 1
                   
                    
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
                        time.sleep(1)
                        stepCountList[1] += 1
                        
                    elif loc>10 and loc < 140:
                        TX_data_py2(serial_port, 15)
                        time.sleep(1)
                        stepCountList[0] += 1
     
                    elif loc>=140 and loc<=180:
                        TX_data_py2(serial_port, 47)
                        time.sleep(1)
                        stepCountList[2] += 1
                   
                    elif loc< 0 :
                        TX_data_py2(serial_port, 47)
                        time.sleep(1)
                        stepCountList[2] += 1
                
            elif stage == 2:
                if  loc > 170:
                    TX_data_py2(serial_port, 20) #Right
                    time.sleep(1)
                    stepCountList[1] += 1
    
                    
                elif loc>10 and loc < 130:
                    TX_data_py2(serial_port, 15) #Left
                    time.sleep(1)
                    stepCountList[0] += 1
                
                elif loc>=130 and loc<=170:
                    TX_data_py2(serial_port, 45) #Milk Up
                    time.sleep(1)
                    stage = 3
                    
                    continue
   
            
            elif stage == 3:

                dan_mask = cv2.inRange(img_hsv[200:,:], lower_black, upper_black)
                dan_count = len(img_hsv[200:,:][np.where(dan_mask != 0)])
                TX_data_py2(serial_port, 51)
                time.sleep(1)
                
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
                    
                '''
                if areacount >1:
                    TX_data_py2(serial_port, 53)
                    time.sleep(1)
                    stage = 5
                
                
                '''
                cap.release()
                cv2.destroyAllWindows()
                exit(1)  
                '''
                  
            elif stage == 5 :
                
                print(stepCountList)
                backStepLeftRight = stepCountList[0] - stepCountList[1]
               
                if backStepLeftRight < 0:
                    for i in range(abs(backStepLeftRight//4)):
                        TX_data_py2(serial_port, 58) 
                        time.sleep(1)
                    for i in range(abs(backStepLeftRight%4)):
                        TX_data_py2(serial_port, 15) 
                        time.sleep(1)
                
                elif backStepLeftRight > 0:
                    for i in range(abs(backStepLeftRight//4)):
                        TX_data_py2(serial_port, 59) 
                        time.sleep(1)
                    for i in range(abs(backStepLeftRight%4)):
                        TX_data_py2(serial_port, 20) 
                        time.sleep(1)
                
               
                backStep = stepCountList[2]
                        
                c = 2
                
                  
                    
                for i in range(backStep//c):
                    TX_data_py2(serial_port, 12) 
                    time.sleep(1)
                    
                for i in range(backStep%c):
                    TX_data_py2(serial_port, 32) 
                    time.sleep(1)
                  
                
            
                TX_data_py2(serial_port, 49)
                time.sleep(1)
                TX_data_py2(serial_port, 49)
                time.sleep(1)
  
                stage = 6
                
                
            elif stage == 6:   
               
                img = cv2.cvtColor(frame[:160], cv2.COLOR_BGR2HSV)
                
                lower_yellow = np.array([10, 100, 100])
                upper_yellow = np.array([50, 255, 255])
                mask = cv2.inRange(img, lower_yellow, upper_yellow)
                image_result = cv2.bitwise_and(frame[:160], frame[:160],mask = mask)

                gray_img = grayscale(image_result)
                blur_img = gaussian_blur(gray_img, 3)
                canny_img = canny(blur_img, 20, 30)
                
                
                hough_img, x, y, gradient = hough_lines(canny_img, 1, 1 * np.pi/180, 30, 0, 20 )
                result = weighted_img(hough_img, frame[:160])
                
                
                if  x == -1: 
                    TX_data_py2(serial_port, 15)
                    time.sleep(1)
             
                print("x",x)
                
                if  x > 170:
                    TX_data_py2(serial_port, 20)
                    time.sleep(1)
                    
                elif x>10 and x < 150:
                    TX_data_py2(serial_port, 15)
                    time.sleep(1)
                    
                elif x>=150 and x<=170: # orginal 140 ~ 180
                    print(gradient)
                    if gradient>0 and gradient< 10:
                        TX_data_py2(serial_port, 4)
                        time.sleep(1)
                        
                    
                    elif gradient<0 and gradient>-10:
                        TX_data_py2(serial_port, 6) 
                        time.sleep(1) 
                    
                    else:
                        break 
    
    
    
    if area == "safe":
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
            
                    
                    [x, y, w, h] = preprocessing(image_result)
                 
             
                elif color == "blue":
                
                    blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
                    image_result = cv2.bitwise_and(frame, frame,mask = blue_mask)
          
                    
                    [x, y, w, h] = preprocessing(image_result)
            
                loc = (x + x + w)/2
             
            if stage == -1:
                if x == -1:
                    if rectangle_count == 3 and head_flag is 0:
                        # head left 28
                        TX_data_py2(serial_port, 74)
                        time.sleep(0.5)
                        rectangle_count = 0
                        head_flag = 2
                        

                    elif rectangle_count == 3 and head_flag is 2:
                        # head right 30
                        TX_data_py2(serial_port, 75)
                        time.sleep(0.5)
                        rectangle_count = 0
                        head_flag = 4
                        
                        
                    elif rectangle_count == 3 and head_flag is 4:
                        # head right 30
                        TX_data_py2(serial_port, 21)
                        time.sleep(0.5)
                        rectangle_count = 0
                        head_flag = 0
                        
                    
                    rectangle_count +=1                         
                else: 
                    TX_data_py2(serial_port, 21)
                    time.sleep(1) 
                    stage = 0
                    locStep = abs(int( (loc-160)/40 ))
                    locStepLeftRight = int( (loc-160)/80)
                    if  head_flag == 2:
                        for i in range(locStep):
                            TX_data_py2(serial_port, 7)
                            time.sleep(0.5) 
                    elif head_flag == 4 :
                        for i in range(locStep):
                            TX_data_py2(serial_port, 9)
                            time.sleep(0.5) 
                     
                    continue
                        
                      
                    
            if stage == 0:
                if  loc > 180:
                    TX_data_py2(serial_port, 20)
                    time.sleep(0.5)
                    stepCountList[1]+=1
                elif loc>10 and loc < 140:
                    TX_data_py2(serial_port, 15)
                    time.sleep(0.5)
                    stepCountList[0]+=1
                elif loc>=140 and loc<=180:
                    stage = 1
                    if y < 180:  
                        TX_data_py2(serial_port, 11)
                        time.sleep(0.5)
                        stepCountList[2]+=4
                    
                    TX_data_py2(serial_port, 29) #Head Down 80 
                      
     
            elif stage == 1:
                print(y + h)
                print(flagcounter)
                if flagcounter == 0:
                    time.sleep(1)
                    
                if flagcounter == 1:
                    stage = 2
                    
                if color == "red" and y + h > 210:
                    flagcounter += 1
       
                elif color == "blue" and y + h > 210:
                    flagcounter += 1
                else :
                    if  loc > 180:
                        TX_data_py2(serial_port, 20)
                        time.sleep(0.5)
                        stepCountList[1]+=1
                    elif loc>10 and loc < 140:
                        TX_data_py2(serial_port, 15)
                        time.sleep(0.5)
                        stepCountList[0]+=1
                    elif loc>=140 and loc<=180:
                        TX_data_py2(serial_port, 47)
                        time.sleep(0.5)
                        stepCountList[2]+=1
                        
                    elif loc< 0 :
                        TX_data_py2(serial_port, 47)
                        time.sleep(0.5)
                        stepCountList[2]+=1
                        
   
            elif stage == 2:
                if  loc > 170:
                    TX_data_py2(serial_port, 20) #Right
                    time.sleep(0.5)
                    stepCountList[1]+=1
                elif loc>10 and loc < 130:
                    TX_data_py2(serial_port, 15) #Left
                    time.sleep(0.5)
                    stepCountList[0]+=1
                elif loc>=130 and loc<=170:
                    stage = 3
                    continue
                    
            elif stage == 3:
                if head_flag == 2: safeloc = "right"
                elif head_flag == 4: safeloc = "left"
                elif head_flag == 0: safeloc = "center"
                
             
                TX_data_py2(serial_port, 29) #original 31
                time.sleep(0.5)
                TX_data_py2(serial_port, 45) #Milk Up
                time.sleep(0.5)
                
                if head_flag is 0:
                    if  locStepLeftRight < 0:
                        for i in range(locStep):
                            TX_data_py2(serial_port, 52)
                            time.sleep(1) 
                    elif locStepLeftRight >0 :
                        for i in range(locStep):
                            TX_data_py2(serial_port, 50)
                            time.sleep(1) 
                     
                if head_flag is not 0:
                    if stepCountList[2] <4:
                        TX_data_py2(serial_port, 76)
                        time.sleep(0.5)
                        stepCountList[2]+=4
                
                stage = 4           
                
                    
            elif stage == 4:
                dan_mask = cv2.inRange(img_hsv[200:,:], lower_green, upper_green)
                safe_count = len(img_hsv[200:,:][np.where(dan_mask != 0)])
                
                if safeloc == "right":
                    TX_data_py2(serial_port, 52) #Right
                    time.sleep(0.5)
                    stepCountList[1]+=3
                elif safeloc == "left":
                    TX_data_py2(serial_port, 50) #Left
                    time.sleep(0.5)
                    stepCountList[0]+=3
                    
                elif safeloc == "center":
                    TX_data_py2(serial_port, 76) #Center 
                    time.sleep(1)
                    stepCountList[2]+=1
                
                tempCnt = 0
                
                if safeloc == "center": tempCnt = 1
                else : tempCnt = 2
                
                 
                time.sleep(2)
                print(safe_count)
                print("area count : ", areacount)
                cv2.imshow("frame", dan_mask)
                cv2.waitKey(1)
                if safe_count > 3000: # safe_count > 8000 and safe_count < 18000
                    areacount += 1
                    
                if areacount > tempCnt:
                    TX_data_py2(serial_port, 53)
                    time.sleep(1)
                    stage = 5
                    
                continue
           
            
            elif stage == 5 :
                
                print(stepCountList)
                backStepLeftRight = stepCountList[0] - stepCountList[1]
                if not safeloc == "center": 
                    if backStepLeftRight < 0:
                        for i in range(abs(backStepLeftRight//4)):
                            TX_data_py2(serial_port, 58) 
                            time.sleep(1)
                        for i in range(abs(backStepLeftRight%4)):
                            TX_data_py2(serial_port, 15) 
                            time.sleep(1)
                    
                    elif backStepLeftRight > 0:
                        for i in range(abs(backStepLeftRight//4)):
                            TX_data_py2(serial_port, 59) 
                            time.sleep(1)
                        for i in range(abs(backStepLeftRight%4)):
                            TX_data_py2(serial_port, 20) 
                            time.sleep(1)
                
               
                backStep = stepCountList[2]
                        
                c = 0
                
                if safeloc == "center": c = 3
                else : c = 2
                    
                    
                for i in range(backStep//c):
                    TX_data_py2(serial_port, 12) 
                    time.sleep(1)
                    
                for i in range(backStep%c):
                    TX_data_py2(serial_port, 32) 
                    time.sleep(1)
                  
                if backStep > 9:
                    TX_data_py2(serial_port, 32) 
                    time.sleep(1)
                
                if safeloc == "right":
                    TX_data_py2(serial_port, 49)
                    time.sleep(1)
                    TX_data_py2(serial_port, 49)
                    time.sleep(1)
        

                elif safeloc == "left":
                    TX_data_py2(serial_port, 48)
                    time.sleep(1)
                    
                else :
                    TX_data_py2(serial_port, 49)
                    time.sleep(1)
                    TX_data_py2(serial_port, 9)
                    time.sleep(1)
                    TX_data_py2(serial_port, 9)
                    time.sleep(1)
                    
                stage = 6
                
            elif stage == 6:   
               
                img = cv2.cvtColor(frame[:160], cv2.COLOR_BGR2HSV)
                
                lower_yellow = np.array([10, 100, 100])
                upper_yellow = np.array([50, 255, 255])
                mask = cv2.inRange(img, lower_yellow, upper_yellow)
                image_result = cv2.bitwise_and(frame[:160], frame[:160],mask = mask)

                gray_img = grayscale(image_result)
                blur_img = gaussian_blur(gray_img, 3)
                canny_img = canny(blur_img, 20, 30)
                
                
                hough_img, x, y, gradient = hough_lines(canny_img, 1, 1 * np.pi/180, 30, 0, 20 )
                result = weighted_img(hough_img, frame[:160])
                
                
                if  x == -1: 
                    TX_data_py2(serial_port, 15)
                    time.sleep(1)
             
                print("x",x)
                
                if  x > 170:
                    TX_data_py2(serial_port, 20)
                    time.sleep(1)
                    
                elif x>10 and x < 150:
                    TX_data_py2(serial_port, 15)
                    time.sleep(1)
                    
                elif x>=150 and x<=170: # orginal 140 ~ 180
                    print(gradient)
                    if gradient>0 and gradient< 10:
                        TX_data_py2(serial_port, 4)
                        time.sleep(1)
                        
                    
                    elif gradient<0 and gradient>-10:
                        TX_data_py2(serial_port, 6) 
                        time.sleep(1) 
                    
                    else:
                        break 
                
        
        cap.release()
        cv2.destroyAllWindows()

        time.sleep(1)
        exit(1)

      
     
    
    
    
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
   
