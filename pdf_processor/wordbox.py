import cv2 
import numpy as np
import os
import math

def get_string(img_path, output_dir, page_num):
    
    img = cv2.imread(img_path)

    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    rgb_planes = cv2.split(img)
    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        # bg_img = cv2.medianBlur(dilated_img, 15)
        bg_img = cv2.GaussianBlur(dilated_img, (5,5),0)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    img = cv2.merge(result_planes)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # kernel = np.ones((2, 2), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    # img = cv2.erode(img, kernel, iterations=1) 

    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # kernel_line = np.ones((5, 5), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_line)
    processed_img_path = 'processed.png'
    cv2.imwrite(processed_img_path, img)
    
    return draw_contours(processed_img_path, output_dir, page_num)

def draw_contours(img_path, output_dir, page_num): 
    img = cv2.imread(img_path,0)
    original_img = img.copy() 
    cv2.imwrite(os.path.join(output_dir, 'original.png'), original_img)

    
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Threshold the image
    # _, threshed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshed = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 1)

    # # Remove vertical table borders
    lines = cv2.HoughLinesP(threshed, 1, np.pi/90, 40, minLineLength=70, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate the angle of the line
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Ignore lines that are vertical or near-vertical
            if abs(angle) > 45:
                cv2.line(threshed, (x1, y1), (x2, y2), (0, 0, 0), 6)        
    lines_removed=threshed 
    cv2.imwrite(os.path.join(output_dir, 'lines_removed.png'), lines_removed)

 # Remove horizontal table borders
    kernel = np.ones((7, 1), np.uintp) 
    opened = cv2.morphologyEx(lines_removed, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(os.path.join(output_dir, 'opened.png'), opened)

    #perform erosion for noise removal
    kernel = np.ones((2,2), np.uint8)
    erosion = cv2.erode(opened, kernel)
    cv2.imwrite(os.path.join(output_dir, 'erosion.png'), erosion)

    #dialation
    kernel = np.ones((7,20), np.uint8)
    dilated = cv2.dilate(erosion, kernel, iterations=1)
    cv2.imwrite(os.path.join(output_dir, 'dilated.png'), dilated)

    #closing to fill the smalls gaps left by dialation
    kernel = np.ones((5,10), np.uint8)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(output_dir, 'closed.png'), closed)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = original_img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(output_dir, 'contours.png'), img_contours)

    x1=1
    boxes = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        if w > 20 and h > 20:
            top_increase_factor = 13 
            bottom_increase_factor = 4
            y -= top_increase_factor
            h += (top_increase_factor + bottom_increase_factor)
            cv2.rectangle(original_img, (x, y), (x+7+w, y+7+h), (0, 255, 0), 2)
            word_img = original_img[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_dir, f'page_{page_num+1}_word_{i+1}.png'), word_img)
            boxes.append((x, y, w, h))

    for box in boxes:
        x, y, w, h = box
    filename = os.path.join(output_dir, 'word_boxes.png')
    cv2.imwrite(filename, original_img)

    return len(boxes)


