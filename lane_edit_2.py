import json
import cv2
import numpy as np
import os
import random

# json dosyasının yüklenmesi
with open("65_export_2023-01-25_21-02-43.json", 'r') as f:
    data1 = json.load(f)

# region of interest fonksiyonu
def region_of_interest(img):
    mask = np.zeros_like(img)
    match_mask_color = 255
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# dosyalara erişim
image_directory = "/home/taha/Desktop/Codes/Aesk_Odevler/new_model/get_image"
save_directory = "/home/taha/Desktop/Codes/Aesk_Odevler/new_model/get_instance_image"

for image_name in os.listdir(image_directory):
    if image_name.endswith(".jpg"):
        image_path = os.path.join(image_directory, image_name)
        img = cv2.imread(image_path)

        height = img.shape[0]
        width = img.shape[1]
        region_of_interest_vertices = [
            (0, height),
            (width / 2, height / 1.9),
            (width, height)
        ]

        cropped_image = region_of_interest(img)

        # image dosyalarının adları ile JSON dosyasındaki adların karşılaştırılması

        for frame in data1['frames']:
            frame_name = frame['name'].split("/")[-1]
            if frame_name == image_name:

                for label in frame['labels']:
                    if 'poly2d' in label:
                        axis = []
                        for vertices in label['poly2d'][0]['vertices']:
                            axis.append(vertices)

                            points = np.array(axis).reshape((-1, 1, 2))
                            points = points.astype(np.int32)
                            color1 = random.randrange(255)
                            color2 = random.randrange(255)
                            color3 = random.randrange(255)
                            son_hal = cv2.polylines(cropped_image, [points], False, (color1, color2, color3), 2)


                            print("foto : ", axis)
                            son_hal = cv2.cvtColor(son_hal, cv2.COLOR_BGR2GRAY)
                            if not axis == []:
                                cv2.imshow("ilk", img)
                                cv2.imshow("son", son_hal)


        # son halini kaydet
        save_path = os.path.join(save_directory, image_name)
        #cv2.imwrite(save_path, son_hal)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
