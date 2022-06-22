from itertools import permutations
import cv2
import json
import os

img_path = './bolt2bg_2/trainB'
img_list = os.listdir(img_path)

for i, file_name in enumerate(img_list):
    img = cv2.imread(os.path.join(img_path, file_name))
    b,g,r = cv2.split(img)
    bgr_list = list(permutations([b, g, r], 3))

    for j in range(len(bgr_list)):
        transform_img = cv2.merge(bgr_list[j])
        cv2.imwrite('./bolt2bg_2/trainB/' + str(i) + '-' + str(j).zfill(6) + '.jpg', transform_img)
        cv2.waitKey(0)
    
# cv2.waitKey(0)
# cv2.destroyAllWindows()