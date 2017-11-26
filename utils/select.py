import random
import os
import codecs
import shutil
from sqlalchemy.sql.expression import except_

count =0



        
root = "/home/kohill/lstm/dataset/dataset"
name_list =  os.listdir(os.path.join(root,"all"))
name_label_list =list( [x[:-4] for x in name_list])
name_unique = list(set([x[:-4] for x in name_list]))
random_name_list = list(random.choice(name_unique) for _ in range(count))
for x in random_name_list:
    try:
        shutil.move("/home/kohill/lstm/dataset/dataset/all/"+x+".jpg","/home/kohill/lstm/dataset/dataset/val")
        shutil.move("/home/kohill/lstm/dataset/dataset/all/"+x+".xml","/home/kohill/lstm/dataset/dataset/val")
    except:
        pass



