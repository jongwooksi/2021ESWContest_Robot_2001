import numpy as np
import argparse
import cv2
import serial
import time
from serialdata import *
import subprocess

#running summarization activity                
lower_red = np.array([0, 30, 60])
upper_red = np.array([20, 255, 150])
lower_red2 = np.array([160, 30, 60])
upper_red2 = np.array([180, 255, 150])
lower_blue = np.array([90, 50, 60])
upper_blue = np.array([130, 255, 150])
lower_yellow = np.array([10, 100, 100])
upper_yellow = np.array([50, 255, 255])
lower_blackArrow = np.array([0, 0, 0])
upper_blackArrow = np.array([180, 236, 52])
lower_green = np.array([35, 30, 30])
upper_green = np.array([100, 255, 255])
lower_blackdanger = np.array([0, 0, 0])
upper_blackdanger = np.array([180, 255, 50])            
            
#milk mission            
lower_red_milk = np.array([0, 30, 60])
upper_red_milk = np.array([20, 255, 150])
lower_red2_milk = np.array([160, 30, 60])
upper_red2_milk = np.array([180, 255, 150])
lower_black_milk = np.array([0, 0, 0])
upper_black_milk = np.array([180, 255, 90])
lower_blue_milk = np.array([90, 80, 30])
upper_blue_milk = np.array([140, 255, 255])
lower_green_milk = np.array([35, 30, 30])
upper_green_milk = np.array([100, 255, 255])

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

    
def finish():
    f = open("./data/result.txt","r")
    TX_data_py2(serial_port, 37)
    time.sleep(1)
    text = f.readline()
    print(text)
	
    for i in range(len(text)):    
       
        print(text[i])
        if text[i] == "A":
            TX_data_py2(serial_port, 39)
            
        elif text[i] == "B":

            TX_data_py2(serial_port, 40)
            
        elif text[i] == "C":
            
            TX_data_py2(serial_port, 41)
            
        elif text[i] == "D":
            
            TX_data_py2(serial_port, 42)
           
        
        time.sleep(2)
        
        
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
            
    return x, maxvalue, gradient


textFlag = 0

def Recog(template):
    #cv2.imshow('img', template)
    #cv2.waitKey(1)
    
    img = cv2.imread('ewsn.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
    th, tw = template.shape[:2]

    methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED']

    count = [0,0,0,0] 


    for i, method_name in enumerate(methods):
        img_draw = img.copy()
        method = eval(method_name)
        res = cv2.matchTemplate(img, template, method)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            match_val = 1-min_val
        else:
            top_left = max_loc
            match_val = max_val
        
        
        if method_name == 'cv2.TM_CCOEFF_NORMED':
            match_val += 0.1
            
        if method_name == 'cv2.TM_SQDIFF_NORMED':
            match_val += 0.08 
            
            
        print(match_val)    
        
        if match_val <= 0.80:
            continue
    
        bottom_right = (top_left[0] + tw, top_left[1] + th)
        
        if top_left[1] >= 0 and top_left[1] <= 162 and bottom_right[1] >= 0 and bottom_right[1] <= 162:
            count[0] += 1
        elif top_left[1] >= 163 and top_left[1] <= 361 and bottom_right[1] >= 163 and bottom_right[1] <= 361:
            count[1] += 1
        elif top_left[1] >= 362 and top_left[1] <= 560 and bottom_right[1] >= 362 and bottom_right[1] <= 560:
            count[2] += 1
        elif top_left[1] >= 561 and top_left[1] <= 720 and bottom_right[1] >= 561 and bottom_right[1] <= 720:
            count[3] += 1
        else:
            continue
        
    max_index = count.index(max(count))
    text = ""
    if max(count) == 3:
        if max_index == 0:
            text = "E"
        elif max_index == 1:
            text = "W"
        elif max_index == 2:
            text = "S"
        elif max_index == 3:
            text = "N"
        
    
    return text


def textImageProcessing(img, frame):

    img = cv2.Canny(img, 15, 40)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img = cv2.dilate(img, kernel, iterations=2)

    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.erode(img, kernel)

    #cv2.imshow("daa", img)
    #key = cv2.waitKey(1)

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        area = cv2.contourArea(c)
        approx = cv2.approxPolyDP(c, 0.1 * cv2.arcLength(c, True), True)


        if area > 3000:
            if len(approx) == 4:
                cv2.drawContours(frame, [approx], 0, (0, 0, 255), 5)
               
            
                left = list(tuple(c[c[:, :, 0].argmin()][0]))
                top = list(tuple(c[c[:, :, 1].argmin()][0]))
                right = list(tuple(c[c[:, :, 0].argmax()][0]))
                bottom = list(tuple(c[c[:, :, 1].argmax()][0]))

                x, y, w, h = cv2.boundingRect(c)

                distance_top = ((x - top[0])**2 + (y - top[1]) ** 2) ** 0.5

                distance_bottom = (((x+w) - bottom[0]) ** 2 + ((y+h) - bottom[1]) ** 2) ** 0.5

                return [[x, y], [x + w, y], [x, y + h], [x + w, y + h]], frame
             

    return [[-1,-1], [-1,-1], [-1,-1], [-1,-1]], frame

cnt = 0

def textRecogABCD(template):

    img = cv2.imread('abcd.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
    th, tw = template.shape[:2]

    methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED']

    count = [0,0,0,0] 


    for i, method_name in enumerate(methods):
        img_draw = img.copy()
        method = eval(method_name)
        res = cv2.matchTemplate(img, template, method)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            match_val = 1-min_val
          
        else:
            top_left = max_loc
            match_val = max_val 
        
        if method_name == 'cv2.TM_CCOEFF_NORMED':
            match_val += 0.14
            
       
        print(match_val)
        if match_val <= 0.85:
            continue
            
        bottom_right = (top_left[0] + tw, top_left[1] + th)
        #print("크기 : ",top_left, bottom_right)
        if top_left[1] >= 0 and top_left[1] <= 162 and bottom_right[1] >= 0 and bottom_right[1] <= 163:
            count[0] += 1
        elif top_left[1] >= 163 and top_left[1] <= 359 and bottom_right[1] >= 163 and bottom_right[1] <= 359:
            count[1] += 1
        elif top_left[1] >= 360 and top_left[1] <= 557 and bottom_right[1] >= 360 and bottom_right[1] <= 557:
            count[2] += 1
        elif top_left[1] >= 558 and top_left[1] <= 720 and bottom_right[1] >= 558 and bottom_right[1] <= 720:
            count[3] += 1
        
    max_index = count.index(max(count))
    text = ""
    if max(count) == 3:
        if max_index == 0:
            text = "A"
        elif max_index == 1:
            text = "B"
        elif max_index == 2:
            text = "C"
        elif max_index == 3:
            text = "D"

        
    return text


def RecogABCD(img_color):
    global lower_red 
    global upper_red 
    global lower_red2 
    global upper_red2
    global lower_blue
    global upper_blue
    
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    mask1 = cv2.inRange(img_hsv, lower_red2, upper_red2)

    red_mask = mask0 + mask1
    blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

    red_hsv = img_hsv.copy()
    blue_hsv = img_hsv.copy()
    

    red_count = len(red_hsv[np.where(red_mask != 0)])
    blue_count = len(blue_hsv[np.where(blue_mask != 0)])
    print(red_count)
    print(blue_count)
    print()
    if red_count > blue_count:
        color = "red"
        red_hsv[np.where(red_mask != 0)] = 0
        red_hsv[np.where(red_mask == 0)] = 255
   
    else:
        color = "blue"
        blue_hsv[np.where(blue_mask != 0)] = 0
        blue_hsv[np.where(blue_mask == 0)] = 255

    
    return color
    
   
def preprocessing(frame):
	
    img = cv2.Canny(frame, 50, 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img = cv2.dilate(img, kernel, iterations=2)



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
 
    #running summarization activity                
    global lower_red 
    global upper_red 
    global lower_red2 
    global upper_red2
    global lower_blue
    global upper_blue
    global lower_yellow 
    global upper_yellow
    global lower_blackArrow
    global upper_blackArrow 
    global lower_green
    global upper_green 
    global lower_blackdanger 
    global upper_blackdanger          
                
    #milk mission            
    global lower_red_milk
    global upper_red_milk 
    global lower_red2_milk 
    global upper_red2_milk
    global lower_black_milk 
    global upper_black_milk
    global lower_blue_milk
    global upper_blue_milk 
    global lower_green_milk
    global upper_green_milk

    W_View_size = 320
    H_View_size = int(W_View_size / 1.333)

    FPS         = 10  #PI CAMERA: 320 x 240 = MAX 90
    
    
    cap = cv2.VideoCapture(0)

    cap.set(3, W_View_size)
    cap.set(4, H_View_size)
    cap.set(5, FPS)  
    


    TX_data_py2(serial_port, 68)
    time.sleep(1)
    
    f = open("./data/result.txt","w")
    f.close()
    
    f = open("./data/start.txt","w")
    
    #Defense recognition ------------------------#
    
    while True:
        
        _,frame = cap.read()
        if not count_frame():
            continue
            
        frame = frame[90:,:]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dst = img.copy()

        points, frame = textImageProcessing(img, frame)



        if points[0][0] is -1:
            continue

        print(points)


        pts1 = np.float32([[ points[0], points[1], points[2], points[3]]])
        pts2 = np.float32([[0, 0], [128, 0], [0, 128], [128, 128]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        textimage = cv2.warpPerspective(dst, matrix, (128, 128))

        textimage = textimage[8:110, 8:110]
        textimage = cv2.resize(textimage, (128, 128))
      

        text = Recog(textimage)

        print("text : {} ".format(text))
        if text == "E":
            f.write(text)
            break
        elif text == "W":
            f.write(text)
            break
        elif text == "S":
            f.write(text)
            break
        elif text == "N":
            f.write(text)
            break
        
    
        
    TX_data_py2(serial_port, 26)
    time.sleep(1)
    TX_data_py2(serial_port, 29) 
    time.sleep(1)
    TX_data_py2(serial_port, 21) 
    time.sleep(1)
    TX_data_py2(serial_port, 11) 
    time.sleep(2)
    

        
    #Open the door ------------------------#
    while True:
        _,frame = cap.read()
        if not count_frame_333():
            continue
            
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(img, lower_yellow, upper_yellow)
        image_result = cv2.bitwise_and(frame, frame,mask = mask)
   
   
        gray_img = grayscale(image_result)
        blur_img = gaussian_blur(gray_img, 3)
        canny_img = canny(blur_img, 20, 30)
        
        hough_img, x, y, gradient = hough_lines(canny_img, 1, 1 * np.pi/180, 30, 0, 20 )
        result = weighted_img(hough_img, frame)
        #cv2.imshow('img', result)
        #cv2.waitKey(1)
        #continue
        #if get_distance() >= 2:
        f = open("./data/start.txt", 'r')
        text = f.readline()
        print(text)
        
        if text == "E":
            TX_data_py2(serial_port, 33)
            
        elif text == "W":
            TX_data_py2(serial_port, 34)
            
        elif text == "S":
            TX_data_py2(serial_port, 35)
            
        elif text == "N":
            TX_data_py2(serial_port, 36)
       
        time.sleep(1)
        TX_data_py2(serial_port, 44)
        time.sleep(1)
        TX_data_py2(serial_port, 12)
        time.sleep(1)
        #TX_data_py2(serial_port, 32)
        #time.sleep(1)
        break
        
    
        
        print(x)
        time.sleep(1) 
        
    time.sleep(1)  
    centerFlag = 0
    
    #To the Center-------------------------#  
    centercount = 0
    TX_data_py2(serial_port, 78)
    time.sleep(1)
                   
    while True:
        _,frame = cap.read()
        if not count_frame_333():
            continue
        
        frameResize = frame[200:,].copy()
        frameResize = cv2.resize(frameResize,(320,240))
        img = cv2.cvtColor(frameResize, cv2.COLOR_BGR2HSV)
                
        mask = cv2.inRange(img, lower_yellow, upper_yellow)
        image_result = cv2.bitwise_and(frameResize, frameResize,mask = mask)

        gray_img = grayscale(image_result)
        blur_img = gaussian_blur(gray_img, 3)
        canny_img = canny(blur_img, 20, 30)
        
        
        hough_img, x, y, gradient = hough_lines(canny_img, 1, 1 * np.pi/180, 30, 0, 20 )
        result = weighted_img(hough_img, frameResize)
        
        
      
        print("x",x)
        if x < 0 :
            #TX_data_py2(serial_port, 32)
            time.sleep(1)
            centercount += 1
        
        if  x > 175:
            TX_data_py2(serial_port, 20)
            time.sleep(1)
            
        elif x>10 and x < 145:
            TX_data_py2(serial_port, 15)
            time.sleep(1)
            
        elif x>=145 and x<=175: # orginal 140 ~ 180
            print(gradient)
            if gradient>0 and gradient< 30:
                TX_data_py2(serial_port, 4)
                time.sleep(1)
                
                
            
            elif gradient<0 and gradient>-30:
                TX_data_py2(serial_port, 6) 
                time.sleep(1) 
                
            else :
                centerFlag += 1
            
            
            if centerFlag > 3: break 
            
   
        
    TX_data_py2(serial_port, 43)
    time.sleep(1)
    TX_data_py2(serial_port, 54)
    time.sleep(1)
    
    #Arrow Recognition-------------------------#  
    
    left_count = 0
    right_count = 0
    Flag = False

    while True:
        _,frame = cap.read()
        if not count_frame_333():
            continue

        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        

        mask = cv2.inRange(hsv, lower_blackArrow, upper_blackArrow)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel)
        #cv2.imshow('uuu', mask)
        #cv2.waitKey(1)
        
        contours,_ = cv2.findContours(mask , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True),True)
          
            
            if area > 800 and area < 320*180:
                
                if Flag is False:
                    left = list(tuple(cnt[cnt[:, :, 0].argmin()][0]))
                    right = list(tuple(cnt[cnt[:, :, 0].argmax()][0]))

                    if int(left[0]) < 2 : 
                        TX_data_py2(serial_port, 62)
                        time.sleep(1)
                        Flag = True
                        
                    if int(right[0]) > 318 : 
                        TX_data_py2(serial_port, 63)
                        time.sleep(1)
                        Flag = True
                        
                points = []
               
                if len(approx)==7:
                    
                    for i in range(7):
                       points.append([approx.ravel()[2*i], approx.ravel()[2*i+1]])

                    points.sort()
                   
                    minimum = points[1][0] - points[0][0]
                    maximum = points[6][0] - points[5][0]

                    if maximum < minimum :
                        left_count += 1
                    else:
                        right_count += 1
                    
                    
        if left_count>right_count and left_count > 3:
            f = open("./data/arrow.txt", 'w')
            print("left")
            f.write("left")
            
            TX_data_py2(serial_port, 26)
            time.sleep(1)
            TX_data_py2(serial_port, 43)
            time.sleep(1)
            TX_data_py2(serial_port, 21)
            break
            
        if left_count<right_count and right_count > 3:
            f = open("./data/arrow.txt", 'w')
            print("right")
            f.write("right")
            TX_data_py2(serial_port, 26)
            time.sleep(1)
            TX_data_py2(serial_port, 43)
            time.sleep(1)
            TX_data_py2(serial_port, 21)
            break


    #TX_data_py2(serial_port, 77)
    #time.sleep(1)
    TX_data_py2(serial_port, 11)
    time.sleep(1)
    #TX_data_py2(serial_port, 11)
    #time.sleep(1)
    
    #set_distance()
    
    #while get_distance() < 1:
    #    TX_data_py2(serial_port, 47)
    #    time.sleep(1)
         
    f = open("./data/arrow.txt", 'r')
    direction = f.readline()
    print(direction)
    f.close()
    
    TX_data_py2(serial_port, 21)
    time.sleep(1)
    TX_data_py2(serial_port, 29)
    
    
    #TX_data_py2(serial_port, 21)
    #time.sleep(1)
    #TX_data_py2(serial_port, 29)
    
    #direction = "right"
    resultFile = open("./data/result.txt","a")
    
    #Move to the direction of the mission-------------------------#  
    for m in range(3):
        moveCnt = 0
        
        if m == 0: moveCnt = 6
        else: moveCnt = 13
            
        for i in range(moveCnt):
            if direction == 'left':
                TX_data_py2(serial_port,58)
                time.sleep(1)
                #if i%5 == 0:
                #    TX_data_py2(serial_port, 19)
                #    time.sleep(1) 
                    
            elif direction == 'right':
                TX_data_py2(serial_port, 59)
                time.sleep(1)
        
        
             
              
        while True:
            #wait_receiving_exit()
            _,frame = cap.read()
            if not count_frame():
                continue
            #time.sleep(2)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
            mask = cv2.inRange(img, lower_yellow, upper_yellow)
            image_result = cv2.bitwise_and(frame, frame,mask = mask)
            
            gray_img = grayscale(image_result)
            #blur_img = gaussian_blur(gray_img, 3)
            
            canny_img = canny(gray_img, 20, 30)
            
            
            hough_img, x, y, gradient = hough_lines(gray_img, 1, 1 * np.pi/180, 30, 0, 20 )
       
            print(x)
            
            if direction == "right":
                if x >= 0:
                    for i in range(3):
                        TX_data_py2(serial_port, 59)
                        time.sleep(1.5)
                    for i in range(4):
                        TX_data_py2(serial_port, 9)
                        time.sleep(1)
                    TX_data_py2(serial_port, 30)
                    break
                
                else:
                    TX_data_py2(serial_port, 59) 
                    time.sleep(2.5) 
                    
            elif direction == "left":
                if x >= 0:
                    TX_data_py2(serial_port, 26)
                    time.sleep(1)
                    for i in range(3):
                        TX_data_py2(serial_port, 58)
                        time.sleep(1.5)
                    
                    for i in range(5):
                        TX_data_py2(serial_port, 7)
                        time.sleep(1)
             
                    TX_data_py2(serial_port, 28)
                    break
                
                else:
                    TX_data_py2(serial_port, 58) 
                    time.sleep(2.5) 
                
        if direction == "left":
            TX_data_py2(serial_port, 28)
            TX_data_py2(serial_port, 31)
        elif direction == "right":
            TX_data_py2(serial_port, 30)
            TX_data_py2(serial_port, 31)
            

        
        #Recognition of dangerous zone or safe zone---------------#
        zoneSafe = 0
        zoneDanger = 0
        
        while True:
           
            _,frame = cap.read()
            if not count_frame_333():
                continue
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            safe_mask = cv2.inRange(hsv, lower_green, upper_green)

            dan_mask = cv2.inRange(hsv, lower_blackdanger, upper_blackdanger)
            

            safe_count = len(hsv[np.where(safe_mask != 0)])
            dan_count = len(hsv[np.where(dan_mask != 0)])
            
            print("safe_count {}".format(safe_count))
            print("dan_count {}".format(dan_count))
            if safe_count > 5500 and dan_count < 1000:
               zoneSafe += 1
            elif dan_count > 5500 and safe_count < 1000: 
               zoneDanger += 1    
            if zoneSafe > 2: # original safe_count = 15000 , dan_count = 30
               print("safe_zone")
               f = open("./data/area.txt", 'w')
               f.write("safe")
               f.close()
               TX_data_py2(serial_port, 38)
               time.sleep(3)
               TX_data_py2(serial_port, 21)
               break
               
            elif zoneDanger > 2: # original dan_count = 15000 , safe_count = 30
               print("dangerous_zone")
               f = open("./data/area.txt", 'w')
               f.write("dangerous")
               f.close()
               
               TX_data_py2(serial_port, 37)
               time.sleep(3)
               TX_data_py2(serial_port, 21)
               
               break
               
        
        rectangle_count = 0
        head_flag = False
            
        TX_data_py2(serial_port, 21)
        time.sleep(1)
        TX_data_py2(serial_port, 43)
        time.sleep(1)
        TX_data_py2(serial_port, 67)
        time.sleep(1)
        
        time.sleep(1)
        
        f = open("./data/area.txt","r")
        area = f.readline()
        f.close()
        
        f_dir = open("./data/arrow.txt", 'r')
        direction = f_dir.readline()
        print(direction)
        f_dir.close()
        
        
        
        f3 = open("./data/color.txt","w")
        
        
        # Finding the room board -------------------------------------#
        
        head_flag = 0
        while True:
           
            if not count_frame_333():
                continue
            _,frame = cap.read()
            frame = frame[60:160]
            frame = cv2.resize(frame, (480,220))
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dst = img.copy()

            points, frame = textImageProcessing(img, frame)

     
            if points[0][0] is -1:
                
                
                
                if rectangle_count == 15 and head_flag is 0:
                    # head left 28
                    TX_data_py2(serial_port, 72)
                    time.sleep(1)
                    rectangle_count = 0
                    head_flag = 1
                    continue
                elif rectangle_count == 15 and head_flag is 1:
                    # head right 30
                    TX_data_py2(serial_port, 70)
                    time.sleep(1)
                    rectangle_count = 0
                    head_flag = 2
                    continue
                elif rectangle_count == 15 and head_flag is 2:
                    # head right 30
                    TX_data_py2(serial_port, 21)
                    time.sleep(1)
                    rectangle_count = 0
                    head_flag = 3
                    continue
                elif rectangle_count == 15 and head_flag is 3:
                    # head right 30
                    TX_data_py2(serial_port, 71)
                    time.sleep(1)
                    rectangle_count = 0
                    head_flag = 4
                    continue
                elif rectangle_count == 15 and head_flag is 4:
                    # head right 30
                    TX_data_py2(serial_port, 73)
                    time.sleep(1)
                    rectangle_count = 0
                    head_flag = 0
                    continue
                elif rectangle_count == 15 and head_flag is 5:
                    # head right 30
                    TX_data_py2(serial_port, 75)
                    time.sleep(1)
                    rectangle_count = 0
                    head_flag = 6
                    continue
                elif rectangle_count == 15 and head_flag is 6:
                    # head right 30
                    TX_data_py2(serial_port, 74)
                    time.sleep(1)
                    rectangle_count = 0
                    head_flag = 0
                    continue        
                rectangle_count += 1
                continue

            print(points)


            pts1 = np.float32([[ points[0], points[1], points[2], points[3]]])
            pts2 = np.float32([[0, 0], [128, 0], [0, 128], [128, 128]])

            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            textimage = cv2.warpPerspective(dst, matrix, (128, 128))

            textimage = textimage[12:110, 12:110]
            textimage = cv2.resize(textimage, (128, 128))

            text = textRecogABCD(textimage)
            
            img_color =  cv2.warpPerspective(frame, matrix, (128, 128))
            img_color = img_color[12:110, 12:110]
            img_color = cv2.resize(img_color, (54, 54))

            color = RecogABCD(img_color)

            print("text : {} \ncolor : {}".format(text, color))
            
            if text =="A" or text =="B" or text=="C" or text=="D":
                TX_data_py2(serial_port, 26)
                time.sleep(1)
            
            if text == "A":
                f3.write(color)
                if area == "dangerous":
                    resultFile.write(text)
                    
                    
                if direction == "right":
                    TX_data_py2(serial_port, 19)
                    time.sleep(1)
                elif direction == "left":
                    TX_data_py2(serial_port, 25)
                    time.sleep(1)
                    
                break
            elif text == "B":
                f3.write(color)
                if area == "dangerous":
                    resultFile.write(text)
                    
                    
                if direction == "right":
                    TX_data_py2(serial_port, 19)
                    time.sleep(1)
                elif direction == "left":
                    TX_data_py2(serial_port, 25)
                    time.sleep(1)
                    
                break
            elif text == "C":
                f3.write(color)
                if area == "dangerous":
                    resultFile.write(text)
                    
                    
                if direction == "right":
                    TX_data_py2(serial_port, 19)
                    time.sleep(1)
                elif direction == "left":
                    TX_data_py2(serial_port, 25)
                    time.sleep(1)
                    
                break
            elif text == "D":
                f3.write(color)
                if area == "dangerous":
                    resultFile.write(text)
                    
                    
                if direction == "right":
                    TX_data_py2(serial_port, 19)
                    time.sleep(1)
                elif direction == "left":
                    TX_data_py2(serial_port, 25)
                    time.sleep(1)
                    
                break

        
        
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
        stage = -1
        step = 0
        stepCountList = [0,0,0]
        rectangle_count = 0
        head_flag = 0
        
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
                _,frame = cap.read()
                
                if not count_frame_333():
                    continue
                img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                
                
                if stage < 3:
                    if color == "red":
                        mask0 = cv2.inRange(img_hsv, lower_red_milk, upper_red_milk)
                        mask1 = cv2.inRange(img_hsv, lower_red2_milk, upper_red2_milk)
                        red_mask = mask0 + mask1
                        image_result = cv2.bitwise_and(frame, frame,mask = red_mask)
                        #cv2.imshow('img',image_result)
                        #cv2.waitKey(1)
                        [x, y, w, h] = preprocessing(image_result)
                     
                     
                    elif color == "blue":
                        blue_mask = cv2.inRange(img_hsv, lower_blue_milk, upper_blue_milk)
                        image_result = cv2.bitwise_and(frame, frame,mask = blue_mask)
                        
                        [x, y, w, h] = preprocessing(image_result)
                        
                                
                    print( x, y, x+w, y+h)
                    loc = (x + x + w)/2
                    print(loc)
                
                                    
                if stage == -1: 
                    TX_data_py2(serial_port, 21)
                    time.sleep(1) 
                    stage = 0
                    locStep = abs(int( (loc-160)/40 ))
                    locStepLeftRight = int( (loc-160)/80)
                    
                    if locStepLeftRight > 0 : 
                        for i in range(locStepLeftRight):
                            TX_data_py2(serial_port, 20)
                            time.sleep(0.5) 
                     
                    elif locStepLeftRight < 0 : 
                        for i in range(locStepLeftRight):
                            TX_data_py2(serial_port, 15)
                            time.sleep(0.5) 
                    TX_data_py2(serial_port, 11)
                    time.sleep(1)
                    stepCountList[2] += 4         
                    continue
                            
                
                elif stage == 0:
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
                        TX_data_py2(serial_port, 11)
                        time.sleep(1)
                        stepCountList[2] += 4
                       
                        
                elif stage == 1:
                    print(y + h)
                    print(flagcounter)
                    if flagcounter > 2:
                        stage = 2
                        
                        
                    if color == "red" and y + h > 220:
                        flagcounter += 1
           
                    elif color == "blue" and y + h > 220:
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

                    dan_mask = cv2.inRange(img_hsv[200:,:], lower_black_milk, upper_black_milk)
                    dan_count = len(img_hsv[200:,:][np.where(dan_mask != 0)])
                    TX_data_py2(serial_port, 51)
                    time.sleep(1)
                    
                    print(dan_count)
                    print("areacount: {}".format(areacount))
                    
                    stepCountList[2] -= 4
                    
                    if dan_count < 5000:
                        areacount += 1
                        time.sleep(1)
                   
                    if areacount >1:
                        TX_data_py2(serial_port, 53)
                        time.sleep(1)
                        stage = 5
                    
                    
                   
                      
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
                            
                    c = 4
                    if backStep > 0 :     
                        for i in range(backStep//c):
                            TX_data_py2(serial_port, 12) 
                            time.sleep(1)
                            
                        for i in range(backStep%c):
                            TX_data_py2(serial_port, 32) 
                            time.sleep(1)
                      
                    
                    if direction == "right":
                        TX_data_py2(serial_port, 49)
                        time.sleep(1)
                        TX_data_py2(serial_port, 9)
                        time.sleep(1)
                    elif direction == "left":
                        TX_data_py2(serial_port, 48)
                        time.sleep(1)
                        TX_data_py2(serial_port, 7)
                        time.sleep(1)
                     
                     
                    #TX_data_py2(serial_port, 49)
                    #time.sleep(1)
      
                    stage = 6
                    
                    
                elif stage == 6:   
                   
                    img = cv2.cvtColor(frame[40:200,:], cv2.COLOR_BGR2HSV)
                    
                    mask = cv2.inRange(img, lower_yellow, upper_yellow)
                    image_result = cv2.bitwise_and(frame[40:200,:], frame[40:200,:],mask = mask)

                    gray_img = grayscale(image_result)
                    blur_img = gaussian_blur(gray_img, 3)
                    canny_img = canny(blur_img, 20, 30)
                    
                    
                    hough_img, x, y, gradient = hough_lines(canny_img, 1, 1 * np.pi/180, 30, 0, 20 )
                    result = weighted_img(hough_img, frame[40:200,:])
                    
                    cv2.imshow("gggg", frame[40:200,:])
                    cv2.waitKey(1)
                    
                    if  x == -1: 
                        if direction == "right":
                            TX_data_py2(serial_port, 15)
                            time.sleep(1)
                        elif direction == "left":
                            TX_data_py2(serial_port, 20)
                            time.sleep(1)
                            
                    elif  gradient > -0.5 and gradient < 0.5:
                        if direction == "right":
                            TX_data_py2(serial_port, 15)
                            time.sleep(1)
                        elif direction == "left":
                            TX_data_py2(serial_port, 20)
                            time.sleep(1)

                        
                               
                    print("x",x)
                    print("y",y)
                    
                    if  x > 180:
                        TX_data_py2(serial_port, 20)
                        time.sleep(1)
                        
                    elif x>10 and x < 140:
                        TX_data_py2(serial_port, 15)
                        time.sleep(1)
                        
                    elif x>=140 and x<=180: # orginal 140 ~ 180
                        print(gradient)
                        if gradient>0 and gradient< 10:
                            TX_data_py2(serial_port, 4)
                            time.sleep(1)
                            
                        
                        elif gradient<0 and gradient>-10:
                            TX_data_py2(serial_port, 6) 
                            time.sleep(1) 
                        
                        else:
                            TX_data_py2(serial_port, 78)
                            time.sleep(1)
                            
                            break 
        
        
        milktempCount = 0
        
        if area == "safe":
            while True:
                _,frame = cap.read()
                if not count_frame_333():
                    continue
                
                zoneDirect = 0
                
                if stage == -1 : zoneDirect = 80
                
                img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                cv2.imshow('asd', img_hsv[zoneDirect:])
                cv2.waitKey(1)
                
                print(stage)
                if stage < 3:
                    if color == "red":
                        mask0 = cv2.inRange(img_hsv[zoneDirect:], lower_red_milk, upper_red_milk)
                        mask1 = cv2.inRange(img_hsv[zoneDirect:], lower_red2_milk, upper_red2_milk)
                        red_mask = mask0 + mask1
                        image_result = cv2.bitwise_and(frame[zoneDirect:], frame[zoneDirect:],mask = red_mask)
       
                        [x, y, w, h] = preprocessing(image_result)
                     
                 
                    elif color == "blue":
                    
                        blue_mask = cv2.inRange(img_hsv[zoneDirect:], lower_blue_milk, upper_blue_milk)
                        image_result = cv2.bitwise_and(frame[zoneDirect:], frame[zoneDirect:],mask = blue_mask)
              
                        
                        [x, y, w, h] = preprocessing(image_result)
                
                    loc = (x + x + w)/2
                print("x : ",x)
                
                if stage == -1:
                    if x == -1 :
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
                            
                        milktempCount = 0
                        rectangle_count +=1                         
                    else: 
                        milktempCount+= 1
                        if milktempCount < 2:
                            continue
                        
                        else :    
                            TX_data_py2(serial_port, 21)
                            time.sleep(1) 
                            
                            locStep = abs(int( (loc-160)/40 ))
                            locStepLeftRight = int( (loc-160)/80)
                            
                            #if ((loc-160)/40) > 3:
                            #    head_flag = 2
                            #elif ((loc-160)/40) < -3:
                            #    head_flag = 4
                            
                            if  x < 140:
                                head_flag = 2
                                for i in range(2):
                                    TX_data_py2(serial_port, 7)
                                    time.sleep(1) 
                            elif x > 180 :
                                head_flag = 4
                                for i in range(2):
                                    TX_data_py2(serial_port, 9)
                                    time.sleep(1) 
                            stage = 0 
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
                        
                    if color == "red" and y + h > 220:
                        flagcounter += 1
           
                    elif color == "blue" and y + h > 220:
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
                    dan_mask = cv2.inRange(img_hsv[200:,:], lower_green_milk, upper_green_milk)
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
                    #cv2.imshow("frame", dan_mask)
                    #cv2.waitKey(1)
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
                    else : c = 4
                        
                        
                    for i in range(backStep//c):
                        TX_data_py2(serial_port, 12) 
                        time.sleep(1)
                        
                    for i in range(backStep%c):
                        TX_data_py2(serial_port, 32) 
                        time.sleep(1)
                      
                    if backStep > 9:
                        TX_data_py2(serial_port, 32) 
                        time.sleep(1)
                    
                    if direction == "right":
                        if safeloc == "right":
                            TX_data_py2(serial_port, 49)
                            time.sleep(1)
                            TX_data_py2(serial_port, 9)
                            time.sleep(1)
                

                        elif safeloc == "left":
                            TX_data_py2(serial_port, 49)
                            time.sleep(1)
                            
                            
                        else :
                            TX_data_py2(serial_port, 49)
                            time.sleep(1)
                            
                          
                            
                    if direction == "left":
                        if safeloc == "left":
                            TX_data_py2(serial_port, 48)
                            time.sleep(1)
                            TX_data_py2(serial_port, 7)
                            time.sleep(1)
                            

                        elif safeloc == "right":
                            TX_data_py2(serial_port, 48)
                            time.sleep(1)
                            
                            
                        else :
                            TX_data_py2(serial_port, 48)
                            time.sleep(1)
                          
                           
                    stage = 6
                    
                elif stage == 6:   
                   
                    img = cv2.cvtColor(frame[40:200,:], cv2.COLOR_BGR2HSV)
                    
                  
                    mask = cv2.inRange(img, lower_yellow, upper_yellow)
                    image_result = cv2.bitwise_and(frame[40:200,:], frame[40:200,:],mask = mask)

                    gray_img = grayscale(image_result)
                    blur_img = gaussian_blur(gray_img, 3)
                    canny_img = canny(blur_img, 20, 30)
                    
                    
                    hough_img, x, y, gradient = hough_lines(canny_img, 1, 1 * np.pi/180, 30, 0, 20 )
                    result = weighted_img(hough_img, frame[40:200,:])
                    cv2.imshow("gggg", frame[40:200,:])
                    cv2.waitKey(1)
                    
                    print("x, y",x, y)
                    
                    if  x == -1: 
                        if direction == "right":
                            TX_data_py2(serial_port, 15)
                            time.sleep(1)
                        elif direction == "left":
                            TX_data_py2(serial_port, 20)
                            time.sleep(1)
                            
                    elif  gradient > -0.5 and gradient < 0.5: 
                        if direction == "right":
                            TX_data_py2(serial_port, 15)
                            time.sleep(1)
                        elif direction == "left":
                            TX_data_py2(serial_port, 20)
                            time.sleep(1)
                        

                    
                    if  x > 180:
                        TX_data_py2(serial_port, 20)
                        time.sleep(1)
                        
                    elif x>10 and x < 140:
                        TX_data_py2(serial_port, 15)
                        time.sleep(1)
                        
                    elif x>=140 and x<=180: # orginal 140 ~ 180
                        print(gradient)
                        if gradient>0 and gradient< 10:
                            TX_data_py2(serial_port, 4)
                            time.sleep(1)
                            
                        
                        elif gradient<0 and gradient>-10:
                            TX_data_py2(serial_port, 6) 
                            time.sleep(1) 
                        
                        
                        else:
                            TX_data_py2(serial_port, 78)
                            time.sleep(1)
                           
                            
                            break 
       
        
        time.sleep(1)
        
    resultFile.close()
    
    f = open("./data/arrow.txt", 'r')
    arrow = f.readline()
    
    for i in range(4):
        if arrow == 'left':
            TX_data_py2(serial_port,58)
            time.sleep(1)
        elif arrow == 'right':
            TX_data_py2(serial_port, 59)
            time.sleep(1)
    
    while True:
        #wait_receiving_exit()
        if not count_frame_1():
            continue
        _,frame = cap.read()
        #img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        
        #lower_black = np.array([0, 0, 0])
        #upper_black = np.array([180, 255, 50])
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #black_mask = cv2.inRange(hsv, lower_blackArrow, upper_blackArrow)
        #black_count = len(hsv[np.where(black_mask != 0)])
        #print("black count", black_count)
       
         
        
        if arrow == 'left':
            for i in range(5):
                TX_data_py2(serial_port, 58) 
                time.sleep(1)
            TX_data_py2(serial_port, 56)
            time.sleep(1)
            finish()
            break
            
        elif arrow == 'right':
            for i in range(5):
                TX_data_py2(serial_port, 59) 
                time.sleep(1)
            TX_data_py2(serial_port, 55)
            time.sleep(1)
            finish()
            break
        
        
        time.sleep(1) 
            
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
    
    serial_d.join()
    
    print("end")
    
    serial_port.flush() # serial cls
    #subprocess.run(["python3 exit.py"], shell=True)
   
