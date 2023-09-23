import json
import cv2
import numpy as np
import matplotlib.pylab as plt
import os
import random



#json dosyasi yukleme...
with open("taha_drivable_28.json", 'r') as f:
    data1 = json.load(f)

#region of interest kodunun tanimlanmasi...
def region_of_interest(img):
    mask = np.zeros_like(img)
    match_mask_color = 255
    #cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#dosyalara erisim...
directory = "/home/taha/Desktop/Codes/Aesk_Odevler/aesk_odev_5_denemee"
j = int(0)
a = int(0)
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        image_name = os.path.basename(file_path)
        if image_name.endswith("jpg"):
            #print(filename)

            img = cv2.imread(file_path)

            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # blur = cv2.GaussianBlur(gray, (3, 3), 0)
            # canny = cv2.Canny(gray, 50, 150)
            #path_name = "/home/taha/Desktop/Codes/Aesk_Odevler/aesk_odev_4/gt_binary_image/"
            file_save_path = "/home/taha/Desktop/Codes/Aesk_Odevler/aesk_odev_5_denemee/yeni_resimler/" + filename


            height = img.shape[0]
            width = img.shape[1]
            region_of_interest_vertices = [

                (0, height),
                (width / 2, height / 1.9),
                (width, height)
            ]


            cropped_image = region_of_interest(img)

            """
            lines = cv2.HoughLines(canny, 1, np.pi / 180.0, 120, np.array([]))
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(cropped_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            """

            #image'leri okuma kismi...
            axis = []

            frames = data1['frames']
            for h in range(2):
                if len(data1['frames'][a]['labels']) <= h:
                    None
                else:
                    for j in data1['frames'][a]['labels'][h]['poly2d'][0]['vertices']:
                        # for i in data1['frames'][k]['labels'][h]['poly2d'][0]['vertices'][j]:
                        # print(i)
                        axis.append(j)


                points = np.array(axis).reshape((-1, 1, 2))
                points = points.astype(np.int32)
                color1 = random.randrange(255)
                color2 = random.randrange(255)
                color3 = random.randrange(255)
                #son_hal = cv2.polylines(cropped_image, [points], True, (color1, color2, color3), 2)
                son_hal = cv2.fillPoly(cropped_image, [points], (color1, color2, color3), cv2.LINE_8)

                print("foto : ", axis)
                #son_hal = cv2.cvtColor(son_hal, cv2.COLOR_BGR2GRAY)

                cv2.imshow("son", son_hal)


                axis = []

                """

                            if len(data1['frames'][k]['labels'][h]['poly2d'][0]['vertices']) <= j:
                                None

                            else:
                                for i in data1['frames'][k]['labels'][h]['poly2d'][0]['vertices'][j]:
                                    print(j, ":", i)
                                    axis = [i] + axis
                                print("axis", axis)
                                #axis = []
                        """

                #cv2.imshow('cropped image', cropped_image)


            a +=1
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #cv2.imwrite(file_save_path, son_hal)




