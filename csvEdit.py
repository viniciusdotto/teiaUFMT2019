import csv

count = -1 
aux = ''
with open('images.csv', newline='') as File:  
	reader = csv.reader(File, delimiter=',', quotechar=',',
                        quoting=csv.QUOTE_MINIMAL)

	myFile = open('example.csv', 'w')
	writer = csv.writer(myFile)
	for row in reader:
		row_aux = row[0]
		row_part = row_aux[36:]
		row_part = row_part[:row_part.find('/')]
		print(row_part)
		if aux != row_part:
			count = count+1
		linha = [row_aux, count]
		writer.writerow(linha)
		aux = row_part


