#  OpenCV 学习笔记 #

##  0、opencv的安装

``` python
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple

```

## 一、计算机图像



一般都是三通道图像，RGB，值为0-255。单通道图片只有一个通道



```python 
cv2.IMREAD_COLOR#表示彩色图
cv2.IMREAD_GRAYSCALE#表示灰度图
```



## 二、OpenCV 常用的接口

### 1.图像的基本操作

#### 1.1常用imread读取图像，返回的是一个三维数组

```python
import numpy as np
import cv2
img = cv2.imread('test.jpg')#读取test,jpg文件  返回的是BGR,虽然很多地方我们见到的都是RGB 返回的img是numpy格式
```

  

#### 1.2用imshow()展示图片

``` python
img = cv2.imread('test.jpg')
cv2.imshow("title",img)#传入两个变量，分别为title和img
cv2.waitKey(0)#停顿xxx毫秒，特别的为0的时候表示按下任意键继续
```



#### 1.3属性（shape属性）

```python
#展示图像的属性
print(img.shape)
#output:(366, 500, 3)表示366*500*3
```

#### 1.4保存

```python
#将这张灰度图进行保存
cv2.imwrite("test_hui.jpg",img2)
```



#### 1.5读取视频

```python
#读取视频
video = cv2.VideoCapture('video.mp4')
if video.isOpened():
    open,frame = video.read()
else:
    open = False

while open:
    if frame is None:
        break
    if open ==True:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('title',gray)
        cv2.waitKey(100)

    open, frame = video.read()

video.release()
```

#### 1.6图像切片

```python
#bgr图像切片
img3 = cv2.imread("test.jpg")
b,g,r = cv2.split(img3)
img4 = cv2.merge((b,g,r))
cv2.imshow("title",img4)
cv2.waitKey(0)
```

改变通道值，产生图片

```python
#产生红绿蓝色通道的图片
img4 = cv2.imread("test.jpg")
img4_red = img4.copy()#一定要加上copy，因为如果直接赋值的话是浅拷贝
img4_red[:,:,0] = 0
img4_red[:,:,1] = 0
cv2.imwrite("test_red.jpg",img4_red)

img4_green = img4.copy()
img4_green[:,:,0] = 0
img4_green[:,:,2] = 0
cv2.imwrite("test_green.jpg",img4_green)

img4_blue = img4.copy()
img4_blue[:,:,1] = 0
img4_blue[:,:,2] = 0
cv2.imwrite("test_blue.jpg",img4_blue)
```

#### 1.7 边界填充

```python
cv2.copyMakeBorder(image, top, bottom, left, right, borderType)
```

#### 1.8 图像改变大小resize函数

```python
#图像融合
img5 = cv2.imread("test.jpg")
img6 = cv2.imread("pika.jpg")
h= img6.shape[0]
w= img6.shape[1]
img5 = cv2.resize(img5,(w,h))
img5 = img5 + img6
cv2.imshow("title",img5)
cv2.waitKey(0)
```

#### 1.9图像阈值操作

```python
cv2.threshold (src, thresh, maxval, type)
```

type的取值

| type                  | 解释                          |
| --------------------- | ----------------------------- |
| cv2.THRESH_BINARY     | 二进制阈值化，非黑即白        |
| cv2.THRESH_BINARY_INV | 反二进制阈值化，非白即黑      |
| cv2.THRESH_TRUNC      | 截断阈值化 ，大于阈值设为阈值 |
| cv2.THRESH_TOZERO     | 阈值化为0 ，小于阈值设为0     |
| cv2.THRESH_TOZERO_INV | 反阈值化为0 ，大于阈值设为0   |

### 2.图像处理操作

#### 2.1滤波

```python
#滤波

#均值滤波
img = cv2.imread("noise1.jpg")
cv2.imshow("title",img)
cv2.waitKey(0)
blur = cv2.blur(img,(3,3))#(3,3)为卷积核
cv2.imshow("title",blur)
cv2.waitKey(0)

#方框滤波,调用的函数跟均值滤波不一样但其实效果是一样的
img = cv2.imread("noise1.jpg")
cv2.imshow("title",img)
cv2.waitKey(0)
box = cv2.boxFilter(img,-1,(3,3),normalize=True)#(3,3)为卷积核
cv2.imshow("title",box)
cv2.waitKey(0)

# #高斯滤波
img = cv2.imread("noise1.jpg")
cv2.imshow("title",img)
cv2.waitKey(0)
box = cv2.GaussianBlur(img,(5,5),1)#(3,3)为卷积核
cv2.imshow("title",box)
cv2.waitKey(0)

# #中值滤波
img = cv2.imread("noise1.jpg")
cv2.imshow("title",img)
cv2.waitKey(0)
box = cv2.medianBlur(img,5)
cv2.imshow("title",box)
cv2.waitKey(0)
```

#### 2.2腐蚀

```python
#腐蚀
img = cv2.imread("noise1.jpg")
cv2.imshow("title",img)
cv2.waitKey(0)
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img,kernel,iterations=1)
cv2.imshow("title",erosion)
cv2.imwrite("noise1_ero.jpg",erosion)
cv2.waitKey(0)
```

#### 2.3 膨胀

```python
#膨胀
img = cv2.imread("noise1_ero.jpg")
cv2.imshow("title",img)
cv2.waitKey(0)
kernel = np.ones((3,3),np.uint8)
dige_dilate = cv2.dilate(img,kernel,iterations= 1)
cv2.imshow("title",dige_dilate)
cv2.imwrite("noise1_ero_dilate.jpg",dige_dilate)
cv2.waitKey(0)
```



#### 2.4开运算

```python
#开运算:先腐蚀再膨胀

img = cv2.imread("noise1.jpg")
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
cv2.imshow('title',opening)
cv2.waitKey(0)
```

#### 2.5闭运算

```python
#闭运算:先膨胀再腐蚀
img = cv2.imread("noise1.jpg")
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)#形态学运算函数
cv2.imshow('title',opening)
cv2.waitKey(0)
```



#### 2.6梯度计算

```python
#梯度计算 = 膨胀 - 腐蚀
pie = cv2.imread("noise.jpg")
kernel = np.ones((7,7),np.uint8)
gradient = cv2.morphologyEx(pie,cv2.MORPH_GRADIENT,kernel)
cv2.imshow('title',gradient)
cv2.waitKey(0)
```



#### 2.7礼帽和黑帽



```python
# 礼帽和黑帽
# 礼帽 = 原图像 - 开运算结果
img = cv2.imread('noise1.jpg')
kernel = np.ones((7,7),np.uint8)
tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
cv2.imshow('title',tophat)
cv2.waitKey(0)



# 黑帽 = 闭运算 - 原图像

img = cv2.imread('noise1.jpg')
kernel = np.ones((7,7),np.uint8)
blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
cv2.imshow('title',blackhat)
cv2.waitKey(0)
```



### 3.图像边缘处理

#### 3.1sobel算子

dst = cv2.Sobel(src,ddepth,dx,dy,ksize)

* ddepth:图像的深度
* dx dy分别表示水平和竖直的方向
* ksize是sobel算子的大小



```python
# sobel算子
img = cv2.imread("cycle.jpg")
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
cv2.imshow("title",sobelx)
cv2.waitKey(0)


sobelx = cv2.convertScaleAbs(sobelx)
cv2.imshow("title",sobelx)
cv2.waitKey(0)
```

```python
# sobel算子
img = cv2.imread("cycle.jpg")
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
cv2.imshow("title",sobelx)
cv2.waitKey(0)

sobelx = cv2.convertScaleAbs(sobelx)
cv2.imshow("title",sobelx)
cv2.waitKey(0)

#sobel 边缘检测
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
cv2.imshow("title",sobelxy)
cv2.waitKey(0)
```

#### 3.2 sobel边缘检测

```python
#sobel边缘检测实例
img = cv2.imread("noise.jpg",cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
cv2.imshow("title",sobelxy)
cv2.waitKey(0)
```

#### 3.3 Scharr算子(更敏感)

```python
#scharrx 边缘检测
img = cv2.imread("noise.jpg",cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Scharr(img,cv2.CV_64F,1,0)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Scharr(img,cv2.CV_64F,0,1)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
cv2.imshow("title",sobelxy)
cv2.waitKey(0)
```

#### 3.4 laplacian算子

**(敏感，但同时对噪音也比较敏感,一般跟其他方法一起使用)**

```python
#laplacian 边缘检测
img = cv2.imread("noise.jpg",cv2.IMREAD_GRAYSCALE)
laplacian = cv2.Laplacian(img,cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)
cv2.imshow("title",laplacian)
cv2.waitKey(0)
```



#### 3.5 Canny 边缘检测

* 1.使用高斯滤波，消除噪声
* 2.计算每个像素点的梯度强度和方向
* 3.采用非极大值抑制，消除杂散效应
* 4.采用双阈值检测确定正式和潜在的边缘
* 5.采用抑制孤立的若边缘最终完成边缘检测



### 4.图像轮廓检测

cv2.findContours(image, mode, method)  

mode: 轮廓检索模式	

* cv2.RETR_EXTERNAL   	表示只检测外轮廓 
* cv2.RETR_LIST                   检测所有轮廓，保存到一条链表上去
* cv2.RETR_CCOMP		   检测所有轮廓，建立两个等级的轮廓，上面的一层为外边界，里面的一层为空洞的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。 
* cv2.RETR_TREE                检索所有的轮廓，并重构嵌套轮廓的整个层次。

method: 轮廓检索模式	

* cv2.CHAIN_APPROX_NONE         存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1 

* cv2.CHAIN_APPROX_SIMPLE       压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息 

```python
#边缘检测
#1. 对图像进行二值处理
img  = cv2.imread("noise.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
cv2.imshow("title",thresh)
cv2.waitKey(0)
#2.调用接口处理轮廓
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#3.绘制轮廓
draw_img = img.copy()
res = cv2.drawContours(draw_img,contours,-1,(0,0,255),2)#-1表示画出所有的轮廓
cv2.imshow("title",res)
cv2.waitKey(0)
#4.计算轮廓面积
cnt = contours[0]
print(cv2.contourArea(cnt))
#5.计算轮廓周长
print(cv2.arcLength(cnt,True))
```

### 5.轮廓近似

```python
#轮廓近似
img  = cv2.imread("noise.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt = contours[1]
draw_img = img.copy()
epsilon = 0.005*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)
res = cv2.drawContours(draw_img,[approx],-1,(0,0,255),2)
cv2.imshow("title",res)
cv2.waitKey(0)
```

### 6.模板匹配

- TM_SQDIFF：计算平方不同，计算出来的值越小，越相关
- TM_CCORR：计算相关性，计算出来的值越大，越相关
- TM_CCOEFF：计算相关系数，计算出来的值越大，越相关
- TM_SQDIFF_NORMED：计算归一化平方不同，计算出来的值越接近0，越相关
- TM_CCORR_NORMED：计算归一化相关性，计算出来的值越接近1，越相关
- TM_CCOEFF_NORMED：计算归一化相关系数，计算出来的值越接近1，越相关

```python
# 模板匹配
img = cv2.imread('lena.jpg', 0)
template = cv2.imread('face.jpg', 0)
h, w = template.shape[:2]
res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
img2 = img.copy()
top_left = min_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img2, top_left, bottom_right, 255, 2)
cv2.imshow("title",img2)
cv2.waitKey(0)
```

**多个对象匹配**

```python
#模板匹配多个对象
img_rgb = cv2.imread('mario.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('mario_coin.jpg', 0)
h, w = template.shape[:2]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
# 取匹配程度大于%80的坐标
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):  # *号表示可选参数
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)

cv2.imshow('img_rgb', img_rgb)
cv2.waitKey(0)
```

### 7.图像直方图

#### cv2.calcHist(images,channels,mask,histSize,ranges)

- images: 原图像图像格式为 uint8 或 ﬂoat32。当传入函数时应 用中括号 [] 括来例如[img]

- channels: 同样用中括号括来它会告函数我们统幅图 像的直方图。如果入图像是灰度图它的值就是 [0]如果是彩色图像 的传入的参数可以是 [0][1][2] 它们分别对应着 BGR。

- mask: 掩模图像。统整幅图像的直方图就把它为 None。但是如 果你想统图像某一分的直方图的你就制作一个掩模图像并 使用它。

- histSize:BIN 的数目。也应用中括号括来

- ranges: 像素值范围常为 [0,256]

  ```python
  #图像直方图与模板匹配
  img = cv2.imread('test.jpg',cv2.IMREAD_GRAYSCALE) #0表示灰度图
  plt.hist(img.ravel(),256)
  plt.show()
  img = cv2.imread('test.jpg')
  color = ('b','g','r')
  for i,col in enumerate(color):
      histr = cv2.calcHist([img],[i],None,[256],[0,256])
      plt.plot(histr,color = col)
      plt.xlim([0,256])
  plt.show()
  ```

### 8.图像均衡化

```python
#图像均衡化
img = cv2.imread('test.jpg',cv2.IMREAD_GRAYSCALE) #0表示灰度图 #clahe
plt.hist(img.ravel(),256)
plt.show()
equ = cv2.equalizeHist(img)
plt.hist(equ.ravel(),256)
plt.show()
res = np.hstack((img,equ))
cv2.imshow("title",res)
cv2.waitKey()
```

**自适应直方图均衡化(即将图像切成小块，每小块进行均衡化)**

```python
#图像均衡化
img = cv2.imread('test.jpg',cv2.IMREAD_GRAYSCALE) #0表示灰度图 #clahe
plt.hist(img.ravel(),256)
plt.show()
equ = cv2.equalizeHist(img)
plt.hist(equ.ravel(),256)
plt.show()
res = np.hstack((img,equ))
cv2.imshow("title",res)
cv2.waitKey()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
res_clahe = clahe.apply(img)
res = np.hstack((img,equ,res_clahe))
cv2.imshow("title",res)
cv2.waitKey(0)
```

### 9.傅里叶变换

#### 傅里叶变换的作用

- 高频：变化剧烈的灰度分量，例如边界
- 低频：变化缓慢的灰度分量，例如一片大海

#### 滤波

- 低通滤波器：只保留低频，会使得图像模糊
- 高通滤波器：只保留高频，会使得图像细节增强







- opencv中主要就是cv2.dft()和cv2.idft()，输入图像需要先转换成np.float32 格式。
  得到的结果中频率为0的部分会在左上角，通常要转换到中心位置，可以通过shift变换来实现。
- cv2.dft()返回的结果是双通道的（实部，虚部），通常还需要转换成图像格式才能展示（0,255）。

```python
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg',0)

img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
# 得到灰度图能表示的形式
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
```

**低通滤波**

```python
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg',0)

img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = int(rows/2) , int(cols/2)     # 中心位置

# 低通滤波
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# IDFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])
plt.show()
```

**高通滤波**

```python
# 高通滤波
img = cv2.imread('lena.jpg',0)

img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = int(rows/2) , int(cols/2)     # 中心位置

# 高通滤波
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0
# IDFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])
plt.show()
```

### 10.信用卡数字识别

```python
import cv2

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts] #用一个最小的矩形，把找到的形状包起来x,y,h,w
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
```

```python
# 导入工具包
from imutils import contours
import numpy as np
import argparse
import cv2
import myutils

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
   help="path to input image")
ap.add_argument("-t", "--template", required=True,
   help="path to template OCR-A image")
args = vars(ap.parse_args())

# 指定信用卡类型
FIRST_NUMBER = {
   "3": "American Express",
   "4": "Visa",
   "5": "MasterCard",
   "6": "Discover Card"
}
# 绘图展示
def cv_show(name,img):
   cv2.imshow(name, img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
# 读取一个模板图像
img = cv2.imread(args["template"])
cv_show('img',img)
# 灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('ref',ref)
# 二值图像
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('ref',ref)

# 计算轮廓
#cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
#返回的list中每个元素都是图像中的一个轮廓

refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,refCnts,-1,(0,0,255),3) 
cv_show('img',img)
print (np.array(refCnts).shape)
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0] #排序，从左到右，从上到下
digits = {}

# 遍历每一个轮廓
for (i, c) in enumerate(refCnts):
   # 计算外接矩形并且resize成合适大小
   (x, y, w, h) = cv2.boundingRect(c)
   roi = ref[y:y + h, x:x + w]
   roi = cv2.resize(roi, (57, 88))

   # 每一个数字对应每一个模板
   digits[i] = roi

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

#读取输入图像，预处理
image = cv2.imread(args["image"])
cv_show('image',image)
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray',gray)

#礼帽操作，突出更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel) 
cv_show('tophat',tophat) 
# 
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, #ksize=-1相当于用3*3的
   ksize=-1)


gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

print (np.array(gradX).shape)
cv_show('gradX',gradX)

#通过闭操作（先膨胀，再腐蚀）将数字连在一起
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel) 
cv_show('gradX',gradX)
#THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
thresh = cv2.threshold(gradX, 0, 255,
   cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] 
cv_show('thresh',thresh)

#再来一个闭操作

thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel) #再来一个闭操作
cv_show('thresh',thresh)

# 计算轮廓

threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
   cv2.CHAIN_APPROX_SIMPLE)

cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3) 
cv_show('img',cur_img)
locs = []

# 遍历轮廓
for (i, c) in enumerate(cnts):
   # 计算矩形
   (x, y, w, h) = cv2.boundingRect(c)
   ar = w / float(h)

   # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
   if ar > 2.5 and ar < 4.0:

      if (w > 40 and w < 55) and (h > 10 and h < 20):
         #符合的留下来
         locs.append((x, y, w, h))

# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x:x[0])
output = []

# 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
   # initialize the list of group digits
   groupOutput = []

   # 根据坐标提取每一个组
   group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
   cv_show('group',group)
   # 预处理
   group = cv2.threshold(group, 0, 255,
      cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
   cv_show('group',group)
   # 计算每一组的轮廓
   digitCnts,hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_SIMPLE)
   digitCnts = contours.sort_contours(digitCnts,
      method="left-to-right")[0]

   # 计算每一组中的每一个数值
   for c in digitCnts:
      # 找到当前数值的轮廓，resize成合适的的大小
      (x, y, w, h) = cv2.boundingRect(c)
      roi = group[y:y + h, x:x + w]
      roi = cv2.resize(roi, (57, 88))
      cv_show('roi',roi)

      # 计算匹配得分
      scores = []

      # 在模板中计算每一个得分
      for (digit, digitROI) in digits.items():
         # 模板匹配
         result = cv2.matchTemplate(roi, digitROI,
            cv2.TM_CCOEFF)
         (_, score, _, _) = cv2.minMaxLoc(result)
         scores.append(score)

      # 得到最合适的数字
      groupOutput.append(str(np.argmax(scores)))

   # 画出来
   cv2.rectangle(image, (gX - 5, gY - 5),
      (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
   cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
      cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

   # 得到结果
   output.extend(groupOutput)

# 打印结果
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)
```

### 11.图像扫描与透视变换

```python
# 导入工具包
import numpy as np
import argparse
import cv2

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned") 
args = vars(ap.parse_args())

def order_points(pts):
	# 一共4个坐标点
	rect = np.zeros((4, 2), dtype = "float32")

	# 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
	# 计算左上，右下
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# 计算右上和左下
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts):
	# 获取输入坐标点
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# 计算输入的w和h值
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# 变换后对应坐标位置
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# 计算变换矩阵
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# 返回变换后结果
	return warped

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv2.resize(image, dim, interpolation=inter)
	return resized

# 读取输入
image = cv2.imread(args["image"])
#坐标也会相同变化
ratio = image.shape[0] / 500.0
orig = image.copy()


image = resize(orig, height = 500)

# 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# 展示预处理结果
print("STEP 1: 边缘检测")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 轮廓检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# 遍历轮廓
for c in cnts:
	# 计算轮廓近似
	peri = cv2.arcLength(c, True)
	# C表示输入的点集
	# epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
	# True表示封闭的
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# 4个点的时候就拿出来
	if len(approx) == 4:
		screenCnt = approx
		break

# 展示结果
print("STEP 2: 获取轮廓")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 透视变换
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# 二值处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('scan.jpg', ref)
# 展示结果
print("STEP 3: 变换")
cv2.imshow("Original", resize(orig, height = 650))
cv2.imshow("Scanned", resize(ref, height = 650))
cv2.waitKey(0)
```

### 12.用tesseract进行OCR检测

```python
from PIL import Image
import pytesseract
import cv2
import os

preprocess = 'blur' #thresh

image = cv2.imread('scan.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if preprocess == "thresh":
    gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

if preprocess == "blur":
    gray = cv2.medianBlur(gray, 3)
    
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)
    
text = pytesseract.image_to_string(Image.open(filename))
print(text)
os.remove(filename)

cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)                                   
```

### 13.角点检测

#### cv2.cornerHarris()

- img： 数据类型为 ﬂoat32 的入图像
- blockSize： 角点检测中指定区域的大小
- ksize： Sobel求导中使用的窗口大小
- k： 取值参数为 [0,04,0.06]

```python
import cv2 
import numpy as np

img = cv2.imread('test_1.jpg')
print ('img.shape:',img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
print ('dst.shape:',dst.shape)

img[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('dst',img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()

```

### 14.opencv SIFT函数

```python
import cv2
import numpy as np

img = cv2.imread('test_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#得到特征点
sift = cv2.SIFT_create()
kp = sift.detect(gray, None)
img = cv2.drawKeypoints(gray, kp, img)

cv2.imshow('drawKeypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#计算特征
kp, des = sift.compute(gray, kp)	
print (np.array(kp).shape)
```

### 15.特征匹配

#### Brute-Force蛮力匹配

```python
import cv2 
import numpy as np
import matplotlib.pyplot as plt
img1 = cv2.imread('box.png', 0)
img2 = cv2.imread('box_in_scene.png', 0)
def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cv_show('img1',img1)
cv_show('img2',img2)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# crossCheck表示两个特征点要互相匹，例如A中的第i个特征点与B中的第j个特征点最近的，并且B中的第j个特征点到A中的第i个特征点也是 
#NORM_L2: 归一化数组的(欧几里德距离)，如果其他特征计算方法需要考虑不同的匹配计算方式
bf = cv2.BFMatcher(crossCheck=True)


#一对一的匹配
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None,flags=2)
cv_show('img3',img3)


#k对最佳匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
cv_show('img3',img3)

```

### 16.cv2.pointPolygonTest(contour, (x,y), False)

```python
result = cv2.pointPolygonTest(contour, (x,y), False) 
#参数说明：参数1：某一轮廓列表、参数2：像素点坐标、参数3：如果为True则输出该像素点到轮廓最近距离。如果为False，则输出为1表示在轮廓内，0为轮廓上，-1为轮廓外。
```





