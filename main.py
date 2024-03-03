import math
import tkinter as tk


def draw(shader, width, height):
    image = bytearray((0, 0, 0) * width * height)
    for y in range(height):
        for x in range(width):
            pos = (width * y + x) * 3
            color = shader(x / width, y / height)
            normalized = [max(min(int(c * 255), 255), 0) for c in color]
            image[pos:pos + 3] = normalized
    header = bytes(f'P6\n{width} {height}\n255\n', 'ascii')
    return header + image


def main(shader):
    label = tk.Label()
    img = tk.PhotoImage(data=draw(shader, 256, 256)).zoom(2, 2)
    label.pack()
    label.config(image=img)
    tk.mainloop()


def shader(x, y):
    if (x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) > 0.16:
        return 0, 0, 0
    return x, 1-y, 0

def shader2(x, y):
    if ((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) > 0.16):
        return 0, 0, 0

    if ((x- 0.5)*(x-0.5) + (y-0.5)*(y-0.5) < 0.002):
        return 0, 0, 0
    a = y- 2
    return x, 1-y, 0




def _generate_clone_spec( param1=None,
                            param2=None,
                         param3 =None,
                        param4=None,
                        ) :
    return(1)
def print_hello_world():
# indented with tab
     print("Hello, World!")
def print_goodbye_world():
    # indented with 4 spaces
 print("Goodbye, World!")
def print_hi_world():
    # indented with 4 spaces
        print("Hi, World!")

import time

startTime  =  time.time()
for i in range(0, 100000):
    alpha = range(ord('А'), ord('Я') + 1)
    ''.join([chr(c) for c in alpha])

endTime = time.time() #время конца замера
totalTime = endTime - startTime
#вычисляем затраченное время
print("Время, затраченное на выполнение данного кода = ",
       totalTime)

startTime1 = time.time()
for i in range(0, 100):
    a2 = 'A'     + 'Б' + 'В' + 'Г' +'Д' + 'Е'+ 'Ё'+ 'Ж' + 'З'+ 'И'+ 'К'+ 'Л'+ 'М'+ 'Н'+ 'О'+ 'П'+ 'Р'+ 'С'+ 'Т'+ 'У'+ 'Ф'+ 'Х'+ 'Ц'+ 'Ч'+ 'Ш'+ 'Щ'+ 'Ь'+ 'Ы'+ 'Ъ'+ 'Э'+ 'Ю'+ 'Я'

endTime1 = time.time () #время конца замера
totalTime1 = endTime1 - startTime1
### вычисляем затраченное время


print("Время, затраченное на выполнение данного кода = ",   totalTime1)
RULES_LIST = [
('Name1', 1, 'Long string upto 40 chars'),
    ('Name2', 2, 'Long string upto 40 chars'),
    ('Name3', 3, 'Long string upto 40 chars'),
    ('Name4', 4, 'Long string upto 40 chars'),
     ('Name5', 5, 'Long string upto 40 chars'),
    ('Name6', 6, 'Long string upto 40 chars'),
    ('Name7',7, 'Long string upto 40 chars') ,
    ('Name8', 8 , 'Long string upto 40 chars' )]
x = 0
if x    in   [1, 2, 3]:#e223, 224
    print(x)

result = 4<<2
result2 = 10%2
result3 = 10*2 #e226
my_tuple = 1,  2 #e241
my_tuple2 = 1,   2,  3#e242
def  func():
    if 1  in [1, 2, 3]: d= 1#274
    if(True):#275
        pass
    pass
def    func():#273
    pass
class MyClass(object):
    def func1():
        pass
    def func2():
        pass
class User(object):

    @property

    def name(self):
        pass

def outer():
    def inner(): pass #e704
    x = False
    if x != True:
        print("false")
    if x != None:
        print("ne nol'")
    def inner2():
        pass
import numpy, sys, os #e401

a = "lvjelegjegheghehggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg"
b = 0
print('fffffffffffffffff '\
      'hhhhhhhhhhhhhhhhhhhh'\
      'nnnnnnnnnnn '\
      '"nnnnnnbbbbb."')#e502
from numpy import monkey; monkey.patch_all()
print('hi, Vika!'); #e703
nums = [1, 2, 3 ,4]
if not nums in nums:
    print(nums)

class User:
    def get_info(self):print("hi")

if type(User) == User:#721
    print(User.name)
try:
    user = User.objects.get(pk=user_id)
    user.send_mail('Hello world')
except:
    logger.error('An error occurred!')