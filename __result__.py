import numpy as np
import sys
from PIL import Image, ImageDraw 
from matplotlib import pyplot as plt

def detect(model, path):
    image = Image.open(path)
    pix = image.load()

    n_spectrum = 3
    width = image.size[0]
    height = image.size[1]

    # creat vector
    picture_vector = []
    for chanel in range(n_spectrum):
        for y in range(height):
            for x in range(width):
                picture_vector.append(pix[x, y][chanel])

    picture_vector = np.array(picture_vector).astype('uint8')
    picture_tensor = picture_vector.reshape([n_spectrum, height, width]).transpose(1, 2, 0)

    picture_tensor = picture_tensor.transpose(2,0,1)

    step = 10; coordinates = []
    for y in range(int((height-(80-step))/step)):
        for x in range(int((width-(80-step))/step) ):
            area = cutting(x*step, y*step, picture_tensor)
            result = model.predict(area)
            if result[0][1] > 0.90 and not_near(x*step,y*step, 88, coordinates):
                coordinates.append([[x*step, y*step], result])
                print(result)

    for e in coordinates:
        x = e[0][0]
        y = e[0][1]
        acc = e[1][0][1]
        thickness = 5
        for i in range(80):
            for ch in range(3):
                for th in range(thickness):
                    picture_tensor[ch][y+i][x-th] = -1

        for i in range(80):
            for ch in range(3):
                for th in range(thickness):
                    picture_tensor[ch][y+i][x+th+80] = -1
        
        for i in range(80):
            for ch in range(3):
                for th in range(thickness):
                    picture_tensor[ch][y-th][x+i] = -1
        
        for i in range(80):
            for ch in range(3):
                for th in range(thickness):
                    picture_tensor[ch][y+th+80][x+i] = -1

    #picture_tensor = picture_tensor.transpose(2,0,1)
    picture_tensor = picture_tensor.transpose(1,2,0)

    plt.figure(1, figsize = (15, 30))

    plt.subplot(3,1,1)
    plt.imshow(picture_tensor)

    plt.show()

def cutting(x, y, picture_tensor):
    area_study = np.arange(3*80*80).reshape(3, 80, 80)
    for i in range(80):
        for j in range(80):
            area_study[0][i][j] = picture_tensor[0][y+i][x+j]
            area_study[1][i][j] = picture_tensor[1][y+i][x+j]
            area_study[2][i][j] = picture_tensor[2][y+i][x+j]
    area_study = area_study.reshape([-1, 3, 80, 80])
    area_study = area_study.transpose([0,2,3,1])
    area_study = area_study / 255
    sys.stdout.write('\rX:{0} Y:{1}  '.format(x, y))
    return area_study  

def not_near(x, y, s, coordinates):
    result = True
    for e in coordinates:
        if x+s > e[0][0] and x-s < e[0][0] and y+s > e[0][1] and y-s < e[0][1]:
            result = False
    return result
    
'''
def show_ship(x, y, acc, thickness=5, picture_tensor):  
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y+i][x-th] = -1

    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y+i][x+th+80] = -1
        
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y-th][x+i] = -1
        
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y+th+80][x+i] = -1
'''