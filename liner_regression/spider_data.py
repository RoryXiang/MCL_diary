import requests
import json
import time


def search_forset(retx, rety, set_num, yr, num_pce, origprc):
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (
    myAPIstr, set_num)
    print(searchURL)
    res = requests.get(searchURL)
    print(res.text)


if __name__ == '__main__':
    retX, retY = 1,2
    search_forset(retX, retY, 8288, 2006, 800, 49.99)