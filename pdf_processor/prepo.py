import cv2
import numpy as np
import os
import math
import fitz  # PyMuPDF
import sys

def extract_pages_from_pdf(pdf_path, temp_dir):
    pdf_document = fitz.open(pdf_path)
    page_images = []
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        img = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        img_array = np.frombuffer(img.samples, dtype=np.uint8).reshape((img.h, img.w, 3))
        page_image_path = os.path.join(temp_dir, f'page_{page_num + 1}.png')
        cv2.imwrite(page_image_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        page_images.append(page_image_path)
    pdf_document.close()
    return page_images

def process_image(img_path, output_dir, page_num):
    
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

    # _, binary_image = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    Dkernel = np.ones((4,4), np.uint8)
    Ekernel = np.ones((5,5), np.uint8)
    
    img = cv2.erode(img, Ekernel, iterations=1) 
    img = cv2.dilate(img, Dkernel, iterations=1)

    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # kernel_line = np.ones((5, 5), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_line)
    processed_img_path = 'processed.png'
    cv2.imwrite(processed_img_path, img)
    

    img = cv2.imread(processed_img_path,0)
    original_img = img.copy() 
    
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Threshold the image
    _, threshed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshed = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 1)

    # Remove vertical table borders
    lines = cv2.HoughLinesP(threshed, 1, np.pi/90, 40, minLineLength=40, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate the angle of the line
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Ignore lines that are vertical or near-vertical
            if abs(angle) > 45:
                cv2.line(threshed, (x1, y1), (x2, y2), (0, 0, 0), 6)        
    lines_removed=threshed 

    # Remove horizontal table borders
    kernel = np.ones((5, 1), np.uintp) 
    opened = cv2.morphologyEx(lines_removed, cv2.MORPH_OPEN, kernel)

    #perform erosion for noise removal
    kernel = np.ones((2,2), np.uint8)
    erosion = cv2.erode(opened, kernel)

    #dialation
    kernel = np.ones((2,80), np.uint8)
    dilated = cv2.dilate(erosion, kernel, iterations=1)

    #closing to fill the smalls gaps left by dialation
    kernel = np.ones((2,40), np.uint8)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    img_contours = original_img.copy()
    cv2.drawContours(img_contours, sorted_contours, -1, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(output_dir, f'contours/page{page_num+1}_contours.png'), img_contours)

    x1=1
    boxes = []
    count_bound = 0
    for i, cnt in enumerate(sorted_contours):
        if cv2.contourArea(cnt) > 3500:  # Set a minimum contour area
            _,_,w,h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            if (1.5<aspect_ratio): 
                if h < img.shape[0] * 0.7 and w < img.shape[0] * 0.8:   # Avoid very large boxes that span most of the image height and width
                    count_bound+=1
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    
                    # Increase top and bottom height
                    top_increase_factor = 6   
                    bottom_increase_factor = 5  
                    center_y = sum([point[1] for point in box]) / 4
                    for point in box:
                        if point[1] < center_y:
                            point[1] -= top_increase_factor
                        else:
                            point[1] += bottom_increase_factor

                    width = int(rect[1][0])
                    height = int(rect[1][1]) + (top_increase_factor + bottom_increase_factor) 

                    src_pts = box.astype("float32")
                    # Coordinate of the points in box points after the rectangle has been straightened
                    dst_pts = np.array([[0, height-1],
                                        [0, 0],
                                        [width-1, 0],
                                        [width-1, height-1]], dtype="float32")

                    # The perspective transformation matrix
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(original_img, M, (width, height))

                    if height > width:
                        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

                    # Save the cropped image
                    cv2.imwrite(os.path.join(output_dir, f'page_{page_num + 74}_line_{x1}.png'), warped)
                    x1+=1
                    boxes.append(box) 

    # image with bounding boxes
    print(len(boxes))
    for box in boxes:
        cv2.drawContours(original_img,[box],0,(0,255,0),1, lineType=cv2.LINE_AA)
    filename = os.path.join(output_dir, f'bound/page{page_num+1}_bounding_boxes.png')
    cv2.imwrite(filename, original_img)
    print('Saved as', filename)
    pass


def main(pdf_path, output_directory):
    temp_dir = 'temp_images'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    page_images = extract_pages_from_pdf(pdf_path, temp_dir)
    for page_num, page_image in enumerate(page_images):
        process_image(page_image, output_directory, page_num)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python getLines.py pdfPath outputDirectory")
        sys.exit(1)
    pdf_path = sys.argv[1]
    output_directory = sys.argv[2]
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    main(pdf_path, output_directory)
