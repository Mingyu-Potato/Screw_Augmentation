import cv2
import numpy as np
import json
import sys

# 한 이미지와, 한 이미지 당 나사와 나사 풀려있는 부분에 대한 labelled data인 json파일을 입력으로 하여, labelling이 되어 있는 부분을 같은 이미지 사이즈로 crop하는 파일이다.

# if img is None and polylines are False, just return the bolt_list and bg_list.
# Otherwise, img is not None and polylines are True, draw the lines at ROI.(Only if you want to check operation)
# 이미지의 json 데이터를 통해, 나사 있는 데이터와 없는 데이터를 분류해서 리스트로 넣어주는 함수이다.
def bolt_bg_list(json_file, img=None, polylines=False):
    bolt_list = []
    bg_list = []
    for i in json_file["shapes"]:
        if i['label'] == 'bolt':
            bolt_list.append(i['points'])

        elif i['label'] == 'bg':
            bg_list.append(i['points'])


    if img is not None and polylines == True:
        for i in json_file["shapes"]:
            if i['label'] == 'bolt':
                pts = np.array(i['points']).astype(np.int64)
                cv2.polylines(img, [pts], True, (0,255,0), 5)
            elif i['label'] == 'bg':
                pts = np.array(i['points']).astype(np.int64)
                cv2.polylines(img, [pts], True, (0,0,255), 5)

        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.imshow('test', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        sys.exit('종료')

    return bolt_list, bg_list


# shape : one object, coord : dot coordinates
# crop할 이미지의 size를 구해주는 함수이다.
# 나사 이미지들과 나사 풀린 이미지들 중에서 가장 큰 이미지를 기준으로 한다.(출력 크기는 4의 배수이다.)
def get_boundary_size(bolt_list, bg_list, sigma):
    max_width, max_height = 0, 0
    for i, shape in enumerate(bolt_list + bg_list):
        max_x, min_x = shape[0][0], shape[0][0]
        max_y, min_y = shape[0][1], shape[0][1]

        for coord in shape:
            if coord[0] >= max_x:
                max_x = coord[0]
            elif coord[0] <= min_x:
                min_x = coord[0]
            if coord[1] >= max_y:
                max_y = coord[1]
            elif coord[1] <= min_y:
                min_y = coord[1]

        width = int(max_x - min_x)
        height = int(max_y - min_y)

        if max_width == 0 or max_height == 0:
            max_width, max_height = width, height

        if width >= max_width:
            max_width = width
        if height >= max_height:
            max_height = height

    boundary_width = (max_width * sigma) // 4 * 4
    boundary_height = (max_height * sigma) // 4 * 4
    boundary_size = (boundary_width, boundary_height)

    return boundary_size



# 나사 이미지들과 나사 풀린 이미지들을 boundary size를 기준으로 잘라서 저장해주는 함수이다.
def crop_list(bolt_list, bg_list, img, boundary_size, file_num=None, save_path=None, save_file=False, polylines=False):
    for i, shape in enumerate(bolt_list):
        max_x, min_x = shape[0][0], shape[0][0]
        max_y, min_y = shape[0][1], shape[0][1]

        for coord in shape:
            if coord[0] >= max_x:
                max_x = coord[0]
            elif coord[0] <= min_x:
                min_x = coord[0]
            if coord[1] >= max_y:
                max_y = coord[1]
            elif coord[1] <= min_y:
                min_y = coord[1]

        width = int(max_x - min_x)
        height = int(max_y - min_y)

        if img is not None and polylines == True:
            pt1 = np.array((min_x - ((boundary_size[0] - width) / 2), min_y - ((boundary_size[1] - height) / 2))).astype(np.int64)
            pt2 = np.array((max_x + ((boundary_size[0] - width) / 2), max_y + ((boundary_size[1] - height) / 2))).astype(np.int64)
            cv2.rectangle(img, pt1, pt2, (0,0,0), 10)

        if polylines == False and save_file == True:
            x1 = int(min_x - ((boundary_size[0] - width) / 2)) // 4 * 4
            x2 = int(max_x + ((boundary_size[0] - width) / 2)) // 4 * 4
            y1 = int(min_y - ((boundary_size[1] - height) / 2)) // 4 * 4
            y2 = int(max_y + ((boundary_size[1] - height) / 2)) // 4 * 4

            cropped_bolt = img[y1:y2, x1:x2]
            cv2.imwrite(save_path['bolt'] + f'{file_num}_{str(i).zfill(7)}.jpg', cropped_bolt)


    for i, shape in enumerate(bg_list):
        max_x, min_x = shape[0][0], shape[0][0]
        max_y, min_y = shape[0][1], shape[0][1]

        for coord in shape:
            if coord[0] >= max_x:
                max_x = coord[0]
            elif coord[0] <= min_x:
                min_x = coord[0]
            if coord[1] >= max_y:
                max_y = coord[1]
            elif coord[1] <= min_y:
                min_y = coord[1]

        width = int(max_x - min_x)
        height = int(max_y - min_y)

        if img is not None and polylines == True:
            pt1 = np.array((min_x - ((boundary_size[0] - width) / 2), min_y - ((boundary_size[1] - height) / 2))).astype(np.int64)
            pt2 = np.array((max_x + ((boundary_size[0] - width) / 2), max_y + ((boundary_size[1] - height) / 2))).astype(np.int64)
            cv2.rectangle(img, pt1, pt2, (0,0,0), 10)

        if polylines == False and save_file == True:
            x1 = int(min_x - ((boundary_size[0] - width) / 2)) // 4 * 4
            x2 = int(max_x + ((boundary_size[0] - width) / 2)) // 4 * 4
            y1 = int(min_y - ((boundary_size[1] - height) / 2)) // 4 * 4
            y2 = int(max_y + ((boundary_size[1] - height) / 2)) // 4 * 4

            cropped_bg = img[y1:y2, x1:x2]
            cv2.imwrite(save_path['bg'] + f'{file_num}_{str(i).zfill(7)}.jpg', cropped_bg)

    if img is not None and polylines == True:
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.imshow('test', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



file_start_num = 1232
file_end_num = 1236

for i in range(file_start_num, file_end_num):
    img_path = f'/home/krri/brtdata/송전철탑/송전철탑/송전철탑 주요 체결부 시편 사진/DSC0{i}.JPG'
    img = cv2.imread(img_path)
    json_path = f'/home/krri/brtdata/송전철탑/송전철탑/송전철탑 주요 체결부 시편 GT 파일/DSC0{i}.json'
    json_file = json.load(open(json_path, 'r'))

    bolt_list, bg_list = bolt_bg_list(json_file, img, False)

    sigma = 1.2
    boundary_size = get_boundary_size(bolt_list, bg_list, sigma=sigma)
    
    save_path = {'bolt' : './bolt2bg_2/testA/', 'bg' : './bolt2bg_2/testB/'}
    crop_list(bolt_list, bg_list, img, boundary_size, file_num=str(i), save_path=save_path, save_file=True, polylines=False)
