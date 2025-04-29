import numpy as np 
import pandas as pd 
import seaborn as sns 
from sklearn import linear_model
import sklearn.metrics as sm 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from PIL import Image


img= Image.open("dataset/2.bmp")
arimg = np.array(img)

print(arimg.shape)
print(arimg[14][21][0])


plt.imshow(arimg)
plt.show()


arimg2 = arimg[:, :, 0]
ar3 = arimg2.reshape(4096)

x = np.empty((0, 4096))
for i in range(1, 21):
    im = Image.open(f"dataset/{i}.bmp")
    p =  np.array(im)
    # reshape images
    p2 = p[:, :, 0]
    p2 = p2.flatten()   
    p2 = p2/255  # 0-1 
    x = np.append(x, [p2], axis=0 )


y = np.append([0] * 10, [1] * 10)

plt.imshow(x[0].reshape(64, 64))
plt.show()

X_train, X_test ,y_train, y_test = train_test_split(x, y, test_size=0.2)
model = linear_model.LogisticRegression()
model.fit(X_train,y_train)

out = model.predict(X_test)

plt.imshow(X_test[1].reshape(64, 64))
plt.show()

er = y_test -out


T = 0
F = 0

for a in er:
    if (a == 0):
        T += 1

    if (a != 0):
        F += 1

print("T : " , T)
print("F : " , F)


T_D = (T * 100)/360
print(f"tedad hads haye dorost:{T_D}")

# ---------------- 2 ------------------


input_1 = Image.open("dataset/t2.bmp")
ar1 = np.array(input_1)
plt.imshow(input_1)
plt.show()

ar2 = ar1[:, :, 0]
ar_flat = ar2.flatten()

t1 = np.array([ar_flat])
w = model.predict(t1)
plt.imshow(t1.reshape(64, 64))
plt.show()