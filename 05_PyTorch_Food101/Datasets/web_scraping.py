import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
import base64
import os

FOOD = 'french_fries'
URL = 'https://www.google.com/search?q=french+fries&client=safari&sca_esv=b657747ebb3c27f5&rls=en&udm=2&biw=1271&bih=840&sxsrf=AHTn8zoESYSCtq5vEP4O30btXOvhIXHwSQ%3A1743545298249&ei=0mPsZ5D9DomlhbIPrtv1uQU&oq=french&gs_lp=EgNpbWciBmZyZW5jaCoCCAAyDRAAGIAEGLEDGEMYigUyCBAAGIAEGLEDMggQABiABBixAzIIEAAYgAQYsQMyChAAGIAEGEMYigUyCBAAGIAEGLEDMgoQABiABBhDGIoFMggQABiABBixAzIKEAAYgAQYQxiKBTIIEAAYgAQYsQNI2xVQqwdY6A1wAXgAkAEAmAFLoAGbA6oBATa4AQPIAQD4AQGYAgegAtcDwgIGEAAYBxgewgIFEAAYgATCAgcQIxgnGMkCwgILEAAYgAQYsQMYgwGYAwCIBgGSBwE3oAecIbIHATa4B84D&sclient=img'

driver = webdriver.Safari()
driver.get(URL)
pics = driver.find_elements(by=By.TAG_NAME, value='img')
list_images = [pic.get_attribute('src') for pic in pics if str(pic.get_attribute('src'))[:10] == 'data:image']
print(len(list_images))
print(list_images[0])
print(list_images[0].split(',')[1].rstrip('/'))
try:
    os.mkdir(f'./{FOOD}')
except FileExistsError:
    print('Directory already exists...')
for index, pic in enumerate(list_images):
    with open(f'./{FOOD}/{FOOD}{index:0>3}.jpeg', 'wb') as f:
        f.write(base64.b64decode(pic.split(',')[1]))

    size = os.path.getsize(f'./{FOOD}/{FOOD}{index:0>3}.jpeg')
    if size < 5_000:
        os.remove(f'./{FOOD}/{FOOD}{index:0>3}.jpeg')
print(len(os.listdir(f'./{FOOD}')))
driver.quit()

