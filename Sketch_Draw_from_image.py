import cv2
image=cv2.imread('PIC1.jpg')
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
inverted = 255 - gray_image
blur = cv2.GaussianBlur(inverted ,(251,151),0)
invertedBlur = 255-blur
sketch = cv2.divide(gray_image , invertedBlur,scale = 250.0)
cv2.imwrite('pencil_sketch.jpg',sketch)
cv2.imshow('image',sketch)

