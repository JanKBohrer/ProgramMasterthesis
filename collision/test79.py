#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:23:29 2019

@author: jdesk
"""

import sys
from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size

import numpy as np

A = np.arange(10)

B = np.ones( (10,10) )

C = 3.346343
D = 50.215

qq = (A,B,C,D)

qql = [A,B,C,D]

qq2 = (A,B,C,D,A,B,C,D)

print(id(A))
print(id(B))
print(id(C))
print(id(D))
print(id(qq))

print()

print(sys.getsizeof(A))
print(sys.getsizeof(B))
print(sys.getsizeof(C))
print(sys.getsizeof(D))
print(sys.getsizeof(qq))
print(sys.getsizeof(qq2))
print(sys.getsizeof(qql))

print()

print(getsize(A))
print(getsize(B))
print(getsize(C))
print(getsize(D))
print(getsize(qq))
print(getsize(qq2))
print(getsize(qql))

print()

print(A)

qq[0][0] = 2000

print(A)

(a,b,c,d) = qq

(a2,b2,c2,d2) = qql

print()
print(a,b,c,d)

print()
print(a2,b2,c2,d2)

print()
print(A,a,a2)
a[1] = 1111
print()
print(A,a,a2)

a2[1] = 2222
print()
print(A,a,a2)

print()
print(id(A)-id(a),id(A)-id(a2))

print()
print(A)
print(qq)

A[3] = 9876

print()
print(A)
print(qq)
print(qq[0])



