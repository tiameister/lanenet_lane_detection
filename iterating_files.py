
import os

#dosyalara erisip isimlerini istedigim gibi duzenleme...


#dosyalara erisim...
directory = "/home/taha/Desktop/Codes/Aesk_Odevler/aesk_odev_4/gt_binary_image"


for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    new_file_name = directory + filename
    if os.path.isfile(file_path):
        if filename.endswith("jpg"):
            os.rename(directory + '/' + filename, directory + '/' + new_file_name)
            #print(filename)


