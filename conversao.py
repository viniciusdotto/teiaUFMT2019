import glob
files = glob.glob("/home/vinicius/Imagens/TEIA/CLASSES/avePet/*.jpg", recursive=False)

from PIL import Image
for i in range (len(files)):
	im = Image.open(files[i])
	width, height = im.size	
	if width > height:
		im = im.rotate(90, expand = 1)  
	im = im.resize((100,178))
	im.save(files[i])
	print (files[i])