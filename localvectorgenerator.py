import os
import pytesseract
from PIL import Image

# Path to the folder containing PNG files
folder_path = 'testdata/'

# List to store the text from each image
text_list = []

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        # Open the image file
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)
        
        # Use pytesseract to extract text from the image
        text = pytesseract.image_to_string(img)
        
        # Append the extracted text to the list
        text_list.append(text)

# Print the extracted text from each image
for i, text in enumerate(text_list):
    print(f"Text from image {i+1}:\n{text}\n")