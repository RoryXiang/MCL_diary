# -*- coding: utf-8 -*-
"""
@Time : 2019/1/15 11:11
@Author : yang
@Site : 
@File : bdxy_all.py
@Software: PyCharm
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import os
from redis import StrictRedis
import json
# from baiduxinyong.repeat  import RedisFilterContainer
# from repeat import RedisFilterContainer
import threading
# from baiduxinyong.connect_monogo import save
import time


def change_ip():
    while 1:
        try:
            re = requests.get("http://116.196.118.3:1314/getIp")
            print(re.text)
            c = {"http": f"http://{str(re.text)}",
                 "https": f"https://{str(re.text)}"}
            return c
        except Exception as e:
            print(e, '获取ip失败')


def load(ketword):
    p = change_ip()
    print(p)
    url = 'https://xin.baidu.com/s?q=' + ketword + '&t=0'
    # url = 'https://xin.baidu.com/s?q=普奥云信息科技(北京)有限公司&t=0'
    headers = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'}
    while True:
        try:
            res = requests.get(url, headers=headers, proxies=p, timeout=20)
            time_date = time.strftime("%Y-%m-%d", time.localtime())
            soup = BeautifulSoup(res.text, "lxml")

            tag = soup.find_all(title=ketword)
            for i in tag:
                if 'href' in str(i):
                    if i.get('href'):
                        onepage = "https://xin.baidu.com" + i.get('href')
                        print(onepage)
                        break

                if "src" in str(i):
                    print(str(i))
                    # c = re.findall("src.*?',", str(i))
                    # pic_url = c[0][5:-2]
                    # print(pic_url)
                    # pic = requests.get(pic_url).content
                    # if not os.path.exists("images/"+time_date):
                    #     os.makedirs("images/"+time_date)
                    # with open("images/"+time_date+"/" + ketword + ".jpg", "wb") as f:
                    #     f.write(pic)
            onepageres = requests.get(
                onepage, headers=headers, proxies=p, timeout=20)
            # get bid =====================
            bid = re.search(r"baiducode\">\d+</span>", onepageres.text)
            bid = bid.group()
            bid = re.search(r"\d+", bid).group()
            # print("*" * 10, bid)
            # =============================
            # get tk ======================
            keyword = re.search(r"getAttribute.+?\)", onepageres.text)
            keyword = keyword.group()
            keyword = keyword.split("'")[1]
            print("?????", keyword)
            tk = re.search(r"{}=.*?>".format(keyword), onepageres.text).group()
            tk = tk.split("\"")[1]
            print("******", tk)
            #  ============================

            # get js ======================
            js_str = re.search(r"function mix[\s\S]+return[\s\S]+?;}",
                               onepageres.text).group()
            print("******", js_str)
            # =============================
            idd = re.findall("document.getElementById.*?;", onepageres.text)

            print(idd[0][25:30])
            tot = re.findall(idd[0][25:30] + ".*</div>", onepageres.text)
            tot = re.findall(idd[0][25:30] + ".*</div>", onepageres.text)
            print(tot)

        except Exception as e:
            print(e, "没找到对应的图片")
            p = change_ip()
        if res:
            # print('爬取成功'+ ketword)
            break
        time.sleep(10)


def connection_db6():
    clinet = StrictRedis(host="172.16.63.61", port="6379",
                         password="redis123456", db=6)
    return clinet


if __name__ == '__main__':
    res = load("普奥云信息科技(北京)有限公司")
