from pdf2image import convert_from_path

dpi = 1000  
width = 1700  
height = 1300  
images = convert_from_path('sample_pdf2.pdf',poppler_path = r"C:\\poppler-24.02.0\\Library\\bin",dpi=dpi,size=(width,height))

for i in range(len(images)):
    images[i].save('C:\\Projects\\ArrowDetection\\images\\page'+str(i)+'.jpg','JPEG')