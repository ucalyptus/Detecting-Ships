# Detecting Ships using Deep Learning
![image](https://cdn-images-1.medium.com/max/1000/1*DcO07U2GAS_AkWQXCzXdQA.png)

## Steps involved
### 1.Preparing Data
![ship1](https://carbon.now.sh/?bg=rgba(171%252C%2520184%252C%2520195%252C%25201)&t=monokai&wt=none&l=python&ds=true&dsyoff=20px&dsblur=68px&wc=true&wa=true&pv=56px&ph=56px&ln=false&fm=Hack&fs=14px&lh=133%2525&si=false&es=2x&wm=false&code=%252523output%252520encoding%25250Ay%252520%25253D%252520np_utils.to_categorical(output_data%25252C%2525202)%25250A%252523shuffle%252520all%252520indexes%25250Aindexes%252520%25253D%252520np.arange(4000)%25250Anp.random.shuffle(indexes)%25250AX_train%252520%25253D%252520X%25255Bindexes%25255D.transpose(%25255B0%25252C2%25252C3%25252C1%25255D)%25250Ay_train%252520%25253D%252520y%25255Bindexes%25255D%25250A%252523normalization%25250AX_train%252520%25253D%252520X_train%252520%25252F%252520255%25250A)

### 2. Network

```
#network design
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(80, 80, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #40x40
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #20x20
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #10x10
model.add(Dropout(0.25))

model.add(Conv2D(32, (10, 10), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #5x5
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))
```

### 3. BBox generation code
```
def cutting(x, y):
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
```

## To view full code , [click here](https://nbviewer.jupyter.org/github/ucalyptus/Detecting-Ships/blob/master/detecting-ships.ipynb)

## [LinkedIn](https://linkedin.com/in/sayantan-das-95b50a125/)
## [Github](https://github.com/ucalyptus)
## [Medium](https://medium.com/@sayantandas30011998)
