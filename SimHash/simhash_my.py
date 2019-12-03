# 实现simhash实现原理

# coding:utf-8
'''
Created on 2015年9月24日
@author: likaiguo
'''
from __future__ import division, unicode_literals

from _collections import defaultdict
import collections
import datetime
import hashlib
import logging
import re
import sys
import time

from bs4 import BeautifulSoup

class Simhash(object):

    def __init__(self, value, f=64, reg=r'[\w\u4e00-\u9fcc]+', hashfunc=None):
        """
        `f` is the dimensions of fingerprints
        `reg` is meaningful only when `value` is basestring and describes
        what is considered to be a letter inside parsed string. Regexp
        object can also be specified (some attempt to handle any letters
        is to specify reg=re.compile(r'\w', re.UNICODE))
        `hashfunc` accepts a utf-8 encoded string and returns a unsigned
        integer in at least `f` bits.
        """

        self.f = f
        self.reg = reg
        self.value = None

        if hashfunc is None:
            def _hashfunc(x):
                # 一些缓存,这个值可以继续扩大
                return int(hashlib.md5(x).hexdigest(), 16)

            self.hashfunc = _hashfunc
        else:
            self.hashfunc = hashfunc

        if isinstance(value, Simhash):
            self.value = value.value
        elif isinstance(value, collections.Iterable):
            self.build_by_features(value)
        else:
            raise Exception('Bad parameter with type {}'.format(type(value)))

    def _slide(self, content, width=4):
        return [content[i:i + width] for i in range(max(len(content) - width + 1, 1))]

    def _tokenize(self, content):
        """
        分词
        """
        content = content.lower()
        content = ''.join(re.findall(self.reg, content))
        ans = self._slide(content)
        return ans

    def build_by_text(self, content):
        features = self._tokenize(content)
        return self.build_by_features(features)

    def build_by_features(self, features):
        hashs = [self.hashfunc(w.encode('utf-8')) for w in features]
        v = [0] * self.f
        masks = [1 << i for i in range(self.f)]
        print(hashs)
        print(len(str(hashs[0])))
        print("mask", masks)
        for h in hashs:
            for i in range(self.f):
                v[i] += 1 if h & masks[i] else -1
        print(v, len(v))
        # 这一步可以把hash码转换成64位one-hot编码
        ans = 0
        for i in range(self.f):
            if v[i] >= 0:
                ans |= masks[i]
        print(ans)
        self.value = ans

    def distance(self, another):
        """
        计算海明距离，海明距离在二进制中表现为 xor，数出1的个数
        """
        assert self.f == another.f
        x = (self.value ^ another.value) & ((1 << self.f) - 1)
        print("???, ", x)
        ans = 0
        while x:
            ans += 1
            x &= x - 1
        return ans


sm = Simhash("i love u")

print(Simhash("i realy true true love love love love you").distance(Simhash("i realy true true love love love love v")))

# v = sm.build_by_features(["i", "love", "you"])
# print(v)
