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
lower_red_milk = np.array([0, 30, 100])
upper_red_milk = np.array([20, 255, 180])
lower_red2_milk = np.array([160, 30, 100])
upper_red2_milk = np.array([180, 255, 180])
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
	
    for i in range(2):    
       
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
            
    return x, y, gradient


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

    img = cv2.imread('../abcd.jpg')
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
    
    
               
    rectangle_count = 0
    head_flag = False
        
    TX_data_py2(serial_port, 21)
    TX_data_py2(serial_port, 43)
    TX_data_py2(serial_port, 67)
    
    
    time.sleep(1)
 
    
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
        
   
    
    time.sleep(1) 
        
    cap.release()
    cv2.destroyAllWindows()
   
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
   
