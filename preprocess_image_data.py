from pycocotools.coco import COCO

dataDir = 'trained'
dataType = 'train2017'
annFile = 'instances_{}.json'.format(dataDir, dataType)

coco = COCO(annFile)
