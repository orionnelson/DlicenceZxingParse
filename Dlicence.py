
from barfind import *
from PIL import Image
import PIL
import cv2
import os
from pyzxing import BarCodeReader
from aamva import AAMVA
import json

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,help = "path to the image file")
args = vars(ap.parse_args())
image2 = args["image"]
#image2 = "testlicence11.jpg"
print()

#load the image and convert it to grayscale
licence = cv2.imread(image2)
barcode = findbarcode(licence)
cv2.imwrite('temp.jpg', barcode)

reader = BarCodeReader()
results = reader.decode('temp.jpg')
# Or file pattern for multiple files
res = results[0]
out = res.get('raw')
out = out.decode('ascii')
amv = AAMVA(format=[2])
decoded = amv._decodeBarcode(out)
#decoded = AAMVA.decode(results)
print(decoded)
#print(output)
#qr = qrtools.QR()
#print(pdf417_decode('C:\\Users\\orion\\OneDrive\\Desktop\\Drivers Licence Barcode\\temp.jpg'))
