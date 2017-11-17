from generateLabelCPM import *

with open('pose_io/data.json', 'r') as f:
    data = json.load(f)
    
num_batches = len(data)
keys = data.keys() #[0:200]
dirpath = 'coco_generatedImages/'

#generatekey = keys[0:10]

start = time.time()
for i, key in enumerate(keys):
    image, mask, heatmap, pagmap = getImageandLabel(data[key])
    cv.imwrite(dirpath + str(key) + '_image.jpg', image)
    np.save(dirpath + str(key) + '_mask.npy', mask)
    np.save(dirpath + str(key) + '_heat.npy', heatmap)
    np.save(dirpath + str(key) + '_pag.npy',pagmap)
    if i%100 == 0:
        print i
end = time.time()
print (end-start)/60

start = time.time()
for key in keys:
    #image, mask, heatmap, pagmap = getImageandLabel(data[key])
    image = cv.imread(dirpath + str(key) + '_image.jpg')
    mask = np.load(dirpath + str(key) + '_mask.npy')
    heatmap = np.load(dirpath + str(key) + '_heat.npy')
    pagmap = np.load(dirpath + str(key) + '_pag.npy')
end = time.time()
print (end-start)/60
