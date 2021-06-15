#INSTRUCTIONS ON USAGE
CALLING WITHING THE GIVEN FOLDER 
Dlicence.py --image somelicence.jpg or png

#Preconditions :

Licence is not Upsidown -
All four Corners are visible - 
Photo is not potato resolution -
There is No reflection off the licence -

Takes licence and uses OpenCV to Find a Barcode then crop to it
Requires Zxing-master and some other things.

pip install -r requirements.txt 

**Configure Zxing Barcode Reader and Add it to Path For Parsing the files

Read about it here 

https://www.or9.ca/blog/opencv-update/


CODE USED IN THIS PROJECT
https://github.com/rechner/py-aamva # AAMVA PARSING FOR CANADIAN LICENCE

https://github.com/zxing/zxing  # ZEBRA CROSSING BARCODE READER
