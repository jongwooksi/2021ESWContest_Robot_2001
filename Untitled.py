#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np

# 입력이미지와 템플릿 이미지 읽기
img = cv2.imread('./eswn/real_input4.jpg')
template = cv2.imread('./eswn/N0.jpg')
th, tw = template.shape[:2]
cv2.imshow('template', template)

# 3가지 매칭 메서드 순회
methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED']

count = [0,0,0,0] #WENS


for i, method_name in enumerate(methods):
    img_draw = img.copy()
    method = eval(method_name)
    # 템플릿 매칭   ---①
    res = cv2.matchTemplate(img, template, method)
    # 최대, 최소값과 그 좌표 구하기 ---②
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(method_name, min_val, max_val, min_loc, max_loc)

    # TM_SQDIFF의 경우 최소값이 좋은 매칭, 나머지는 그 반대 ---③
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        match_val = min_val
    else:
        top_left = max_loc
        match_val = max_val
    # 매칭 좌표 구해서 사각형 표시   ---④      
    bottom_right = (top_left[0] + tw, top_left[1] + th)
    print("크기 : ",top_left, bottom_right)
    if top_left[1] >= 0 and top_left[1] <= 171 and bottom_right[1] >= 0 and bottom_right[1] <= 171:
        count[0] += 1
    elif top_left[1] >= 172 and top_left[1] <= 361 and bottom_right[1] >= 172 and bottom_right[1] <= 361:
        count[1] += 1
    elif top_left[1] >= 362 and top_left[1] <= 521 and bottom_right[1] >= 362 and bottom_right[1] <= 521:
        count[2] += 1
    elif top_left[1] >= 522 and top_left[1] <= 720 and bottom_right[1] >= 522 and bottom_right[1] <= 720:
        count[3] += 1
    
max_index = count.index(max(count))
print(max_index)
        
    #cv2.rectangle(img_draw, top_left, bottom_right, (0,0,255),2)
    # 매칭 포인트 표시 ---⑤
    #cv2.putText(img_draw, str(match_val), top_left, \
    #            cv2.FONT_HERSHEY_PLAIN, 2,(0,255,0), 1, cv2.LINE_AA)
    #cv2.imshow(method_name, img_draw)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# W : 0~171 , E : 172 ~ 361, N : 362 ~ 521, S : 522 ~ 720


# In[ ]:




