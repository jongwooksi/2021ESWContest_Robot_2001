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


def textImageProcessing(img, frame):

    img = cv2.Canny(img, 25, 45)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    img = cv2.dilate(img, kernel, iterations=2)

    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    img = cv2.erode(img, kernel)

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        area = cv2.contourArea(c)
        approx = cv2.approxPolyDP(c, 0.05 * cv2.arcLength(c, True), True)

        
        if area > 7000 :
            cv2.drawContours(frame,[approx],0,(255,0,0),5)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
              
                
                return [[x, y], [x + w, y], [x, y + h], [x + w, y + h]], frame
              
    return [[-1,-1], [-1,-1], [-1,-1], [-1,-1]], frame

textFlag = 0

def textRecog(textimage):
    global textFlag
    
    textimage = cv2.cvtColor(textimage, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    textimage = cv2.dilate(textimage, kernel)
    
    template = np.zeros((128, 128), np.uint8) + 255
    
    template[37:91, 37:91] = textimage
    
    img = cv2.imread('real_input4.jpg')
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
            match_val = min_val
        else:
            top_left = max_loc
            match_val = max_val
    
        bottom_right = (top_left[0] + tw, top_left[1] + th)
  
        if top_left[1] >= 0 and top_left[1] <= 171 and bottom_right[1] >= 0 and bottom_right[1] <= 171:
            count[0] += 1
        elif top_left[1] >= 172 and top_left[1] <= 361 and bottom_right[1] >= 172 and bottom_right[1] <= 361:
            count[1] += 1
        elif top_left[1] >= 362 and top_left[1] <= 521 and bottom_right[1] >= 362 and bottom_right[1] <= 521:
            count[2] += 1
        elif top_left[1] >= 522 and top_left[1] <= 720 and bottom_right[1] >= 522 and bottom_right[1] <= 720:
            count[3] += 1
        
    max_index = count.index(max(count))
    
    if max_index == 0:
        text = "W"
    elif max_index == 1:
        text = "E"
    elif max_index == 2:
        text = "N"
    elif max_index == 3:
        text = "S"
    
    return text

def Recog(textimage, img_color):
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

   
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 60])
    mask = cv2.inRange(img_hsv, lower, upper)

    hsv = img_hsv.copy()


    hsv[np.where(mask != 0)] = 0
    hsv[np.where(mask == 0)] = 255
    text = textRecog(hsv)


    return text


def loop(serial_port):
 
    W_View_size = 320
    H_View_size = int(W_View_size / 1.333)

    FPS         = 10  #PI CAMERA: 320 x 240 = MAX 90
    
    
    cap = cv2.VideoCapture(0)

    cap.set(3, W_View_size)
    cap.set(4, H_View_size)
    cap.set(5, FPS)  
    


    TX_data_py2(serial_port, 68)
    time.sleep(1)
    
    f = open("./data/start.txt","w")

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
        textimage = cv2.resize(textimage, (54, 54))
      

        img_color =  cv2.warpPerspective(frame, matrix, (128, 128))
        img_color = img_color[8:110, 8:110]
        img_color = cv2.resize(img_color, (54, 54))

        text = Recog(textimage, img_color)

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
    TX_data_py2(serial_port, 2) 
    time.sleep(2)

    while True:
        _,frame = cap.read()
        if not count_frame_333():
            continue
            
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_yellow = np.array([10, 100, 100])
        upper_yellow = np.array([50, 255, 255])
        mask = cv2.inRange(img, lower_yellow, upper_yellow)
        image_result = cv2.bitwise_and(frame, frame,mask = mask)
   
   
        gray_img = grayscale(image_result)
        blur_img = gaussian_blur(gray_img, 3)
        canny_img = canny(blur_img, 20, 30)
        
        hough_img, x, y, gradient = hough_lines(canny_img, 1, 1 * np.pi/180, 30, 0, 20 )
        result = weighted_img(hough_img, frame)
        

        if get_distance() >= 2:
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
           
            TX_data_py2(serial_port, 47)
            time.sleep(1)
            TX_data_py2(serial_port, 44)
            time.sleep(1)
            TX_data_py2(serial_port, 32)
            time.sleep(1)
            TX_data_py2(serial_port, 32)
            time.sleep(1)
            TX_data_py2(serial_port, 32)
            time.sleep(1)
            
            break
        
        
        if gradient>0 and gradient< 2:
            TX_data_py2(serial_port, 4)
            time.sleep(1)
            continue
        
        elif gradient<0 and gradient>-2:
            TX_data_py2(serial_port, 6) 
            time.sleep(1) 
            continue
           
        if  x == -1:
            continue
            
        if  x > 185:
            TX_data_py2(serial_port, 20)
            time.sleep(1) 
          
                
        elif x>10 and x < 135:
            TX_data_py2(serial_port, 15)
            time.sleep(1)  
           
        
        elif x>=135 and x<=185:
            TX_data_py2(serial_port, 47)  
            time.sleep(1) 
            
        
        print(x)
        time.sleep(1) 
        
    time.sleep(3)    
    while True:
        _,frame = cap.read()
        if not count_frame_333():
            continue
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_yellow = np.array([10, 100, 100])
        upper_yellow = np.array([50, 255, 255])
        mask = cv2.inRange(img, lower_yellow, upper_yellow)
        image_result = cv2.bitwise_and(frame, frame,mask = mask)

        gray_img = grayscale(image_result)
        blur_img = gaussian_blur(gray_img, 3)
        canny_img = canny(blur_img, 20, 30)
        
        
        hough_img, x, y, gradient = hough_lines(canny_img, 1, 1 * np.pi/180, 30, 0, 20 )
        result = weighted_img(hough_img, frame)
        

        if  x == -1: 
            print("No Line")
            break
     
        print("x",x)
        
        if  x > 180:
            TX_data_py2(serial_port, 20)
            
        elif x>10 and x < 140:
            TX_data_py2(serial_port, 15)
             
        elif x>=140 and x<=180: # orginal 140 ~ 180
            print("Center")
            break 
            
        
    TX_data_py2(serial_port, 43)
    time.sleep(1)
    TX_data_py2(serial_port, 54)
    time.sleep(1)
     
    left_count = 0
    right_count = 0
    Flag = False
     
    while True:
        _,frame = cap.read()
        if not count_frame_333():
            continue

        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 0, 0])
        upper_red = np.array([180, 236, 52])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel)

        contours,_ = cv2.findContours(mask , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True),True)
          
            
            if area > 2000 and area < 320*180:
                
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
                    
                    cv2.drawContours(frame,[approx],0,(0,0,0),5)
                    
              
        if left_count>right_count and left_count > 10:
            f = open("./data/arrow.txt", 'w')
            print("left")
            f.write("left")
            
            TX_data_py2(serial_port, 26)
            time.sleep(1)
            TX_data_py2(serial_port, 43)
            time.sleep(1)
            TX_data_py2(serial_port, 21)
            exit(1)
            
        if left_count<right_count and right_count > 10:
            f = open("./data/arrow.txt", 'w')
            print("right")
            f.write("right")
            TX_data_py2(serial_port, 26)
            time.sleep(1)
            TX_data_py2(serial_port, 43)
            time.sleep(1)
            TX_data_py2(serial_port, 21)
            break


    TX_data_py2(serial_port, 2)
    time.sleep(1)
            
    f.close()
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
    
	
    
