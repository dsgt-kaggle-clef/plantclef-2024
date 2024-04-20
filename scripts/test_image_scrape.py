from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import os
import sys
import requests
import time
from chromedriver_py import binary_path

searchword1 = 'Bromiuls sterilis'
searchword2 = 'plant'
searchurl = f'https://www.google.com/search?q={searchword1}+{searchword2}&source=lnms&tbm=isch'
dirs = 'pictures' 
maxcount = 100

svc = Service(binary_path)
svc.start()

if not os.path.exists(dirs):
    os.mkdir(dirs)

def download_google_staticimages():
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    # options.add_argument('--headless')  # Uncomment this line if you want to run Chrome in headless mode

    try:
        browser = webdriver.Chrome(service=svc, options=options)
    except Exception as e:
        print('Chromedriver not found in this environment.')
        print(f'Please install chromedriver on your machine. Exception: {e}')
        sys.exit()

    browser.set_window_size(1280, 1024)
    browser.get(searchurl)
    time.sleep(1)

    print('Getting images. This may take a few moments...')

    for _ in range(100):
        browser.execute_script("window.scrollBy(0, 1000)")

    page_source = browser.page_source 
    soup = BeautifulSoup(page_source, 'html.parser')
    images = soup.find_all('img')

    urls = []
    for image in images:
        try:
            url = image['src']
            if url.startswith('https://'):
                urls.append(url)
        except Exception as e:
            print('Error extracting image URLs:', e)

    count = 0
    for url in urls[:maxcount]:
        try:
            res = requests.get(url, verify=False, stream=True)
            if res.status_code == 200:
                with open(os.path.join(dirs, f'img_{count}.jpg'), 'wb') as f:
                    f.write(res.content)
                    count += 1
        except Exception as e:
            print('Failed to download image:', e)

    browser.quit()
    return count

# Main block
def main():
    t0 = time.time()
    count = download_google_staticimages()
    t1 = time.time()

    total_time = t1 - t0
    print('\n')
    print(f'Download completed. Successful count = {count}.')
    print(f'Total time: {total_time:.2f} seconds.')

if __name__ == '__main__':
    main()

svc.stop()

