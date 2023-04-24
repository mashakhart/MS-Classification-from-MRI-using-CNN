import cv2
  
image = cv2.imread(r'C:\Users\mkara\OneDrive\Desktop\exampe3\MS-negative\1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
cv2.imshow('Original image',image)
cv2.imshow('Gray image', gray)

print(image.shape)
print(gray.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()