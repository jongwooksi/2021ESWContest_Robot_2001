# -*- coding: utf-8 -*-

import numpy as np
import argparse
import cv2
import serial
import time
import pytesseract
from serialdata import *


def textImageProcessing(img, frame):

    img = cv2.Canny(img, 15, 40)

    #cv2.imshow("dd", img)
    #key = cv2.waitKey(1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img = cv2.dilate(img, kernel, iterations=1)

    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    img = cv2.erode(img, kernel)
    
    cv2.imshow("daa", img)
    key = cv2.waitKey(1)

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        area = cv2.contourArea(c)
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)


        if area > 2000:
            if len(approx) == 4:
                cv2.drawContours(frame, [approx], 0, (0, 0, 255), 5)
                #cv2.imshow("d", frame)
                #key = cv2.waitKey(1)
                left = list(tuple(c[c[:, :, 0].argmin()][0]))
                top = list(tuple(c[c[:, :, 1].argmin()][0]))
                right = list(tuple(c[c[:, :, 0].argmax()][0]))
                bottom = list(tuple(c[c[:, :, 1].argmax()][0]))

                x, y, w, h = cv2.boundingRect(c)
              
                
                distance_top = ((x - top[0])**2 + (y - top[1]) ** 2) ** 0.5

                distance_bottom = (((x+w) - bottom[0]) ** 2 + ((y+h) - bottom[1]) ** 2) ** 0.5
                
                return [[x, y], [x + w, y], [x, y + h], [x + w, y + h]], frame
                '''
                if distance_top < w * 0.1 or distance_top > w * 0.9:
                    print("회전없음")
                    return [[x, y], [x + w, y], [x, y + h], [x + w, y + h]], frame

                elif distance_bottom < w * 0.1 or distance_bottom > w * 0.9 :
                    print("회전없음")
                    return [[x, y], [x + w, y], [x, y + h], [x + w, y + h]], frame

                elif distance_top > (w/2):  
                    print("왼쪽")
                    return [left, top, bottom, right], frame

                else:
                    print("오른쪽")
                    return [top, right, left, bottom], frame
                '''

    return [[-1,-1], [-1,-1], [-1,-1], [-1,-1]], frame

def textRecog(textimage):
    textimage = cv2.cvtColor(textimage, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    textimage = cv2.dilate(textimage, kernel)
    
    result = np.zeros((128, 128), np.uint8) + 255
    result[45:99, 25:79] = textimage
    result[45:99, 79:123] = textimage[:,10:]
    
    
    cv2.imshow("aa", result)
    cv2.waitKey(1)

    text_image = pytesseract.image_to_string(result)
    text_image.replace(" ","")
    text_image.rstrip() 
    text_image = text_image[0:1]
    print(text_image)
    if text_image == "E":
        text = "E"
    elif text_image == "W":
        text = "W"
    elif text_image == "S" or text_image == "Y":
        text = "S"
    elif text_image == "N":
        text = "N"
    else :
        text = "error"

    if text == "error":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        textimage = cv2.dilate(textimage, kernel)

        result = np.zeros((128, 128), np.uint8) + 255
        result[45:99, 25:79] = textimage
        result[45:99, 79:123] = textimage[:,10:]
    

    
        text_image = pytesseract.image_to_string(result, lang='eng')
        text_image.replace(" ","")
        text_image.rstrip() 
        text_image = text_image[0:1]
        print(text_image)
        if text_image == "E":
            text = "E"
        elif text_image == "W":
            text = "W"
        elif text_image == "S" or text_image == "Y" :
            text = "S"
        elif text_image == "N":
            text = "N"
        else:
            text = "error"

    if text == "error":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        textimage = cv2.erode(textimage, kernel)

        result = np.zeros((128, 128), np.uint8) + 255
        result[45:99, 25:79] = textimage
        result[45:99, 79:123] = textimage[:,10:]
    
        print(text_image)
        #cv2.imshow("canny", result)
        #key = cv2.waitKey(1)

        text_image = pytesseract.image_to_string(result, lang='eng')
        text_image.replace(" ","")
        text_image.rstrip() 
        text_image = text_image[0:1]
        
        if text_image == "E":
            text = "E"
        elif text_image == "W":
            text = "W"
        elif text_image == "S" or text_image == "Y" :
            text = "S"
        elif text_image == "N":
            text = "N"
        else:
            text = "error"

    return text

def Recog(textimage, img_color):
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

   
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 40])
    mask = cv2.inRange(img_hsv, lower, upper)

    hsv = img_hsv.copy()


    hsv[np.where(mask != 0)] = 0
    hsv[np.where(mask == 0)] = 255
    text = textRecog(hsv)


    return text

def loop(serial_port):
    W_View_size = 320
    H_View_size = int(W_View_size / 1.333)

    FPS         = 5  #PI CAMERA: 320 x 240 = MAX 90

    TX_data_py2(serial_port, 21)
    time.sleep(1)
    TX_data_py2(serial_port, 43)
    time.sleep(1)
    TX_data_py2(serial_port, 59)
    time.sleep(1)
    TX_data_py2(serial_port, 54)
    
    cap = cv2.VideoCapture(0)

    cap.set(3, W_View_size)
    cap.set(4, H_View_size)
    cap.set(5, FPS)  
    
    
    while True:
        #wait_receiving_exit()
        _,frame = cap.read()
        frame = frame[50:,:]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dst = img.copy()

        points, frame = textImageProcessing(img, frame)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)

        if key == 27:
            break

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
            TX_data_py2(serial_port, 33)
            break
                
        elif text == "W":
            TX_data_py2(serial_port, 34)
            break
        elif text == "S":
            TX_data_py2(serial_port, 35)
            break
        elif text == "N":
            TX_data_py2(serial_port, 36)
            break
                


    cap.release()
    cv2.destroyAllWindows()
    TX_data_py2(serial_port, 26)
    time.sleep(2)
    TX_data_py2(serial_port, 58)
    print('recog')
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
         
    
   
  








