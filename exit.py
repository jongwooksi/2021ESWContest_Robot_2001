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



def weighted_img(img, initial_img, a=1, b=1., c=0.):
    return cv2.addWeighted(initial_img, a, img, b, c)
 
 
    
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
        
def loop(serial_port):
    
    W_View_size = 320
    H_View_size = int(W_View_size / 1.333)

    FPS         = 1  #PI CAMERA: 320 x 240 = MAX 90
    TX_data_py2(serial_port, 21)
    time.sleep(1)
    TX_data_py2(serial_port, 43)
    time.sleep(1)
    TX_data_py2(serial_port, 31)
    time.sleep(1)
    
    cap = cv2.VideoCapture(0)

    cap.set(3, W_View_size)
    cap.set(4, H_View_size)
    cap.set(5, FPS)  

    #TX_data_py2(serial_port, 29)
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
        #if not count_frame_1():
        #    continue
        _,frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        black_count = len(hsv[np.where(black_mask != 0)])
        print("black count", black_count)
       
         
        
        if arrow == 'left':
            TX_data_py2(serial_port, 58) 
            time.sleep(1)
            if black_count >= 4000:
                TX_data_py2(serial_port, 56)
                time.sleep(1)
                finish()
                break
            
        elif arrow == 'right':
            TX_data_py2(serial_port, 59) 
            time.sleep(1)
            if black_count >= 4000:
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
    
    #serial_t.join()
    serial_d.join()
    print("end")
    
	
    
