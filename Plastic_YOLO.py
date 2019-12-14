import os
import time
import cv2
import numpy as np
from model.yolo_model import YOLO

def process_image(img):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image

def get_classes(file):
    """Get classes name.

    # Argument:
        file: classes name for database.

    # Returns
        class_names: List, classes name.

    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names

def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    global cX, cY
    area = []
    cl_list = []
    cl_max = None
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
        
        if all_classes[cl] == 'bottle':
            area_temp = (top - right)*(left - bottom)
            area.append(area_temp)
            cl_list.append(cl)
    
    print(area)
    
    if len(area) != 0:  
        cl_max = np.argmax(area)
        
    print(cl_max)
    max_area_flag = 0
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
        
        
        print(max_area_flag)
        if all_classes[cl] == 'bottle':
            if max_area_flag == cl_max:
                cv2.rectangle(image, (top, left), (right, bottom), (0, 0, 255), 2)
                cv2.circle(image,center = (int((top+right)/2),int((left+bottom)/2)),radius = 5,color=(255,0,0),thickness = -1)
                cv2.putText(image, '{} {} {}'.format('Main Target',all_classes[cl], score),
                            (top, left - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 1,
                            cv2.LINE_AA)
                (cX,cY) = (int((top+right)/2),int((left+bottom)/2))
                
            else:
                cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
                cv2.circle(image,center = (int((top+right)/2),int((left+bottom)/2)),radius = 5,color=(255,0,0),thickness = -1)
                cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                            (top, left - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 1,
                            cv2.LINE_AA)  
            
            max_area_flag = max_area_flag + 1

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print(len(area))
    if len(area) != 0:

        if cX > (image.shape[1]/2) + 0.15*image.shape[1]:
            print('Rotate Laptop to the Right')

        elif cX < (image.shape[1]/2) - 0.15*image.shape[1]:
            print('Rotate Laptop to the Left')

        else:
            print('Head Straight')

    else:
        print("Can't find bottle")

    print()
    
def detect_image(image, yolo, all_classes):
    """Use yolo v3 to detect images.

    # Argument:
        image: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.

    # Returns:
        image: processed image.
    """
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()

    print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)

    return image

def detect_video(yolo, all_classes):
    """Use yolo v3 to detect video.

    # Argument:
        video: video file.
        yolo: YOLO, yolo model.
        all_classes: all classes name.
    """
    camera = cv2.VideoCapture(0)
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    #camera.set(cv2.CAP_PROP_FPS, 15)
    while True:
        res, frame = camera.read()

        if not res:
            break

        image = detect_image(frame, yolo, all_classes)
        cv2.imshow("detection", image)

        if cv2.waitKey(110) & 0xff == 27:
                break

    camera.release()
    cv2.destroyAllWindows()
    
yolo = YOLO(0.6, 0.5)
file = 'data/coco_classes.txt'
all_classes = get_classes(file)

if __name__ == "__main__":
    detect_video(yolo,all_classes)
    camera = cv2.VideoCapture(0)
    camera.release()
    cv2.destroyAllWindows()