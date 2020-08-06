# extract text from all the images in a folder 
# storing the text in a single file 
from PIL import Image 
import pytesseract 
import os 

pytesseract.pytesseract.tesseract_cmd= r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
      
def main(): 
    # path for the folder for getting the raw images 
    path =r"C:\Users\user\Desktop\ANPR System\images"
  
    # link to the file in which output needs to be kept 
    fullTempPath =r"C:\Users\user\Desktop\ANPR System\images\outputFile.txt"
  
    # iterating the images inside the folder 
    for imageName in os.listdir(path): 
        inputPath = os.path.join(path, imageName) 
        img = Image.open(inputPath) 
  
        # applying ocr using pytesseract for python 
        text = pytesseract.image_to_string(img, lang ="eng") 
  
        # saving the  text for appending it to the output.txt file 
        # a + parameter used for creating the file if not present 
        # and if present then append the text content 
        file1 = open(fullTempPath, "a+") 
  
        # providing the name of the image 
        file1.write(imageName+"\n") 
  
        # providing the content in the image 
        file1.write(text+"\n") 
        file1.close()  
  
    # for printing the output file 
    file2 = open(fullTempPath, 'r') 
    print(file2.read()) 
    file2.close()         
  
  
if __name__ == '__main__': 
    main() 