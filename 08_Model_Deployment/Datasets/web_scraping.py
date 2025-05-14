import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
import base64
import os

FOOD = 'cup_cakes'
URL = 'https://www.google.com/search?q=cup+cakes&client=safari&sca_esv=d57c1a3d738cc115&rls=en&channel=31&udm=2&biw=1452&bih=780&sxsrf=AHTn8zo0tyIlZHsPDmgazb4CnGlNcIlG6Q%3A1747227915083&ei=C5UkaIX0BPC-hbIPq_Sb4QU&ved=0ahUKEwjFpeGZg6ONAxVwX0EAHSv6JlwQ4dUDCBE&uact=5&oq=cup+cakes&gs_lp=EgNpbWciCWN1cCBjYWtlczIHECMYJxjJAjIHEAAYgAQYCjIHEAAYgAQYCjIHEAAYgAQYCjIHEAAYgAQYCjIHEAAYgAQYCjIHEAAYgAQYCjIHEAAYgAQYCjIHEAAYgAQYCjIHEAAYgAQYCkj-C1AAWJEJcAB4AJABAJgBaKABugWqAQM4LjG4AQPIAQD4AQGYAgmgAvgFwgIKEAAYgAQYQxiKBcICEBAAGIAEGLEDGEMYgwEYigXCAggQABiABBixA8ICCxAAGIAEGLEDGIMBwgIFEAAYgATCAgoQABiABBixAxgKwgINEAAYgAQYsQMYQxiKBZgDAJIHAzguMaAHozyyBwM4LjG4B_gF&sclient=img'

print(os.getcwd())

driver = webdriver.Safari()
driver.get(URL)
pics = driver.find_elements(by=By.TAG_NAME, value='img')
list_images = [pic.get_attribute('src') for pic in pics if str(pic.get_attribute('src'))[:10] == 'data:image']
print(len(list_images))
print(list_images[0])
print(list_images[0].split(',')[1].rstrip('/'))
try:
    os.mkdir(f'./final_test/{FOOD}')
except FileExistsError:
    print('Directory already exists...')
for index, pic in enumerate(list_images):
    with open(f'./final_test/{FOOD}/{FOOD}{index:0>3}.jpeg', 'wb') as f:
        f.write(base64.b64decode(pic.split(',')[1]))

    size = os.path.getsize(f'./final_test/{FOOD}/{FOOD}{index:0>3}.jpeg')
    if size < 5_000:
        os.remove(f'./final_test/{FOOD}/{FOOD}{index:0>3}.jpeg')
print(len(os.listdir(f'./final_test/{FOOD}')))
driver.quit()

