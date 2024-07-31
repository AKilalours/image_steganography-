from stegano import lsb
from os.path import isfile,join
from helpers import *
import time                                                                 #install time ,opencv,numpy modules
import cv2
import numpy as np
import math
import os
import shutil
from subprocess import call,STDOUT 
from termcolor import cprint 
import moviepy.editor
from moviepy.editor import *
from werkzeug.utils import secure_filename
from PIL import Image 
import PIL 

# f_name="0.png"
# secret_dec=lsb.reveal(f_name)
# print(secret_dec)


def frame_extraction(files):
    if not os.path.exists("./tmp"):
        os.makedirs("tmp")
    temp_folder="./tmp"
    print("[INFO] tmp directory is created")

    vidcap = cv2.VideoCapture("uploads/{}".format(files))
    count = 0

    while True:
        success, image = vidcap.read()
        if not success:
            break
        cv2.imwrite(os.path.join(temp_folder, "{:d}.png".format(count)), image)
        count += 1
    print("video encodesucess")  

def decrypt(secret_dec,s):
    encoded = blockConverter(secret_dec)
    enlength = len(encoded)
    A = int(encoded[0],2)
    B = int(encoded[1],2)
    C = int(encoded[2],2)
    D = int(encoded[3],2)
    cipher = []
    cipher.append(A)
    cipher.append(B)
    cipher.append(C)
    cipher.append(D)
    r=12
    w=32
    modulo = 2**32
    lgw = 5
    C = (C - s[2*r+3])%modulo
    A = (A - s[2*r+2])%modulo
    for j in range(1,r+1):
        i = r+1-j
        (A, B, C, D) = (D, A, B, C)
        u_temp = (D*(2*D + 1))%modulo
        u = ROL(u_temp,lgw,32)
        t_temp = (B*(2*B + 1))%modulo 
        t = ROL(t_temp,lgw,32)
        tmod=t%32
        umod=u%32
        C = (ROR((C-s[2*i+1])%modulo,tmod,32)  ^u)  
        A = (ROR((A-s[2*i])%modulo,umod,32)   ^t) 
    D = (D - s[1])%modulo 
    B = (B - s[0])%modulo
    orgi = []
    orgi.append(A)
    orgi.append(B)
    orgi.append(C)
    orgi.append(D)
    return cipher,orgi      

def decode_string(key,files):
    # files=files[ :-4]
    # files="{}.mp4".format(files)
    frame_extraction(files)
    secret=[]
    root="./tmp/"
    for i in range(len(os.listdir(root))):
        f_name="{}{}.png".format(root,i)
        secret_dec=lsb.reveal(f_name)
        print(secret_dec)
        # if secret_dec == None:
        #     print("none")
        secret.append(secret_dec)
    # hidden_data=lsb.reveal("images\savedimage.png")

    # print(hidden_data)
    print ("DECRYPTION: ")
    #key='A WORD IS A WORD'
    # key =input("Enter Key(0-16 characters): ")
    if len(key) <16:
        key = key + " "*(16-len(key))
    key = key[:16]
                         
    print ("UserKey: "+key )
    s = generateKey(key)
    
    cipher,orgi = decrypt(secret_dec,s)
    sentence = deBlocker(orgi)
    print ("\nEncrypted String list: ",cipher)
    print ("Encrypted String: " + secret_dec)
    print ("Length of Encrypted String: ",len(secret_dec))

    print ("\nDecrypted String list: ",orgi)
    print ("Decrypted String: " + sentence )
    print ("Length of Decrypted String: ",len(sentence))


key="jilla"
files="finish.mp4"
decode_string(key,files)    