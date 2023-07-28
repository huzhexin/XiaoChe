
import cv2
import numpy as np
import matplotlib.pyplot as plt#Matplotlib是RGB
def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()
# #展示图像数据
# img = cv2.imread("1.jpg",cv2.IMREAD_GRAYSCALE)
# h,w = img.shape
# img = cv2.resize(img,(int(h/2),int(w/2)))
# h,w = img.shape
# print(img.shape)
# with open("1.txt",'a') as f:
#     for i in range(0,h):
#         for j in range(0,w):
#             if img[i][j]<=9:
#                 f.write(" "+str(img[i][j])+" ")
#             elif(10<=img[i][j]<=99):
#                 f.write(str(img[i][j])+" ")
#             else:
#                 f.write(str(img[i][j]))
#         f.write("\n")





# #展示图像
# img1 = cv2.imread('test.jpg')
# cv2.imshow("title",img1)#传入两个变量，分别为title和img
# cv2.waitKey(0)
#
# #展示图像的属性
# print(img1.shape)
#
# #可以进行灰度图读取
# img2 = cv2.imread('test.jpg',cv2.IMREAD_GRAYSCALE)
# cv2.imshow("title",img2)
# cv2.waitKey(0)
#
# #将这张灰度图进行保存
# cv2.imwrite("test_hui.jpg",img2)






# #读取视频
# video = cv2.VideoCapture('video.mp4')
# if video.isOpened():
#     open,frame = video.read()
# else:
#     open = False
#
# while open:
#     if frame is None:
#         break
#     if open ==True:
#         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         cv2.imshow('title',gray)
#         cv2.waitKey(100)
#
#     open, frame = video.read()
#
# video.release()

# #bgr图像切片
# img3 = cv2.imread("test.jpg")
# b,g,r = cv2.split(img3)
# img4 = cv2.merge((b,g,r))
# cv2.imshow("title",img4)
# cv2.waitKey(0)


# #产生红绿蓝色通道的图片
# img4 = cv2.imread("test.jpg")
# img4_red = img4.copy()
# img4_red[:,:,0] = 0
# img4_red[:,:,1] = 0
# cv2.imwrite("test_red.jpg",img4_red)
#
# img4_green = img4.copy()
# img4_green[:,:,0] = 0
# img4_green[:,:,2] = 0
# cv2.imwrite("test_green.jpg",img4_green)
#
# img4_blue = img4.copy()
# img4_blue[:,:,1] = 0
# img4_blue[:,:,2] = 0
# cv2.imwrite("test_blue.jpg",img4_blue)

# #图像融合
# img5 = cv2.imread("test.jpg")
# img6 = cv2.imread("pika.jpg")
# h= img6.shape[0]
# w= img6.shape[1]
# img5 = cv2.resize(img5,(w,h))
# img5 = img5 + img6
# cv2.imshow("title",img5)
# cv2.waitKey(0)



# #制作噪音
#
# peppers = cv2.imread("noise.jpg", 0)
# row, column = peppers.shape
# noise_salt = np.random.randint(0, 256, (row, column))
# noise_pepper = np.random.randint(0, 256, (row, column))
# rand = 0.1
# noise_salt = np.where(noise_salt < rand * 256, 255, 0)
# noise_pepper = np.where(noise_pepper < rand * 256, -255, 0)
# peppers.astype("float")
# noise_salt.astype("float")
# noise_pepper.astype("float")
# salt = peppers + noise_salt
# pepper = peppers + noise_pepper
# salt = np.where(salt > 255, 255, salt)
# pepper = np.where(pepper < 0, 0, pepper)
# cv2.imshow("salt", salt.astype("uint8"))
# cv2.imshow("pepper", pepper.astype("uint8"))
# cv2.waitKey()
# cv2.imwrite("noise1.jpg",salt.astype("uint8"))




# 滤波
#
# #均值滤波
# img = cv2.imread("noise1.jpg")
# cv2.imshow("title",img)
# cv2.waitKey(0)
# blur = cv2.blur(img,(3,3))#(3,3)为卷积核
# cv2.imshow("title",blur)
# cv2.waitKey(0)
#
# #方框滤波,调用的函数跟均值滤波不一样但其实效果是一样的
# img = cv2.imread("noise1.jpg")
# cv2.imshow("title",img)
# cv2.waitKey(0)
# box = cv2.boxFilter(img,-1,(3,3),normalize=True)#(3,3)为卷积核
# cv2.imshow("title",box)
# cv2.waitKey(0)
#
# # #高斯滤波
# img = cv2.imread("noise1.jpg")
# cv2.imshow("title",img)
# cv2.waitKey(0)
# box = cv2.GaussianBlur(img,(5,5),1)#(3,3)为卷积核
# cv2.imshow("title",box)
# cv2.waitKey(0)
#
# # #中值滤波
# img = cv2.imread("noise1.jpg")
# cv2.imshow("title",img)
# cv2.waitKey(0)
# box = cv2.medianBlur(img,5)
# cv2.imshow("title",box)
# cv2.waitKey(0)


# #腐蚀
# img = cv2.imread("noise1.jpg")
# cv2.imshow("title",img)
# cv2.waitKey(0)
# kernel = np.ones((3,3),np.uint8)
# erosion = cv2.erode(img,kernel,iterations=1)
# cv2.imshow("title",erosion)
# cv2.imwrite("noise1_ero.jpg",erosion)
# cv2.waitKey(0)


# #膨胀
# img = cv2.imread("noise1_ero.jpg")
# cv2.imshow("title",img)
# cv2.waitKey(0)
# kernel = np.ones((3,3),np.uint8)
# dige_dilate = cv2.dilate(img,kernel,iterations= 1)
# cv2.imshow("title",dige_dilate)
# cv2.imwrite("noise1_ero_dilate.jpg",dige_dilate)
# cv2.waitKey(0)


# #开运算:先腐蚀再膨胀
# img = cv2.imread("noise1.jpg")
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)#形态学运算函数
# cv2.imshow('title',opening)
# cv2.waitKey(0)

# #闭运算:先膨胀再腐蚀
# img = cv2.imread("noise1.jpg")
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)#形态学运算函数
# cv2.imshow('title',opening)
# cv2.waitKey(0)

# #梯度计算 = 膨胀 - 腐蚀

# pie = cv2.imread("noise.jpg")
# kernel = np.ones((7,7),np.uint8)
# gradient = cv2.morphologyEx(pie,cv2.MORPH_GRADIENT,kernel)
# cv2.imshow('title',gradient)
# cv2.waitKey(0)


# # 礼帽和黑帽
# # 礼帽 = 原图像 - 开运算结果
# img = cv2.imread('noise1.jpg')
# kernel = np.ones((7,7),np.uint8)
# tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
# cv2.imshow('title',tophat)
# cv2.waitKey(0)
#
#
#
# # 黑帽 = 闭运算 - 原图像
#
# img = cv2.imread('noise1.jpg')
# kernel = np.ones((7,7),np.uint8)
# blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
# cv2.imshow('title',blackhat)
# cv2.waitKey(0)



# # sobel算子
# img = cv2.imread("cycle.jpg")
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
# cv2.imshow("title",sobelx)
# cv2.waitKey(0)
#
# sobelx = cv2.convertScaleAbs(sobelx)
# cv2.imshow("title",sobelx)
# cv2.waitKey(0)
#
# #sobel 边缘检测
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
# sobely = cv2.convertScaleAbs(sobely)
# sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
# cv2.imshow("title",sobelxy)
# cv2.waitKey(0)



# #sobel边缘检测实例
# img = cv2.imread("noise.jpg",cv2.IMREAD_GRAYSCALE)
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
# sobelx = cv2.convertScaleAbs(sobelx)
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
# sobely = cv2.convertScaleAbs(sobely)
# sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
# cv2.imshow("title",sobelxy)
# cv2.waitKey(0)



# #scharrx 边缘检测
# img = cv2.imread("noise.jpg",cv2.IMREAD_GRAYSCALE)
# sobelx = cv2.Scharr(img,cv2.CV_64F,1,0)
# sobelx = cv2.convertScaleAbs(sobelx)
# sobely = cv2.Scharr(img,cv2.CV_64F,0,1)
# sobely = cv2.convertScaleAbs(sobely)
# sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
# cv2.imshow("title",sobelxy)
# cv2.waitKey(0)


# #laplacian 边缘检测
# img = cv2.imread("noise.jpg",cv2.IMREAD_GRAYSCALE)
# laplacian = cv2.Laplacian(img,cv2.CV_64F)
# laplacian = cv2.convertScaleAbs(laplacian)
# cv2.imshow("title",laplacian)
# cv2.waitKey(0)


# #Canny 检测
# img = cv2.imread("noise.jpg",cv2.IMREAD_GRAYSCALE)
# v1 = cv2.Canny(img,80,150)
# cv2.imshow("title",v1)
# cv2.waitKey(0)


# #边缘检测
# #1. 对图像进行二值处理
# img  = cv2.imread("noise.jpg")
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
# cv2.imshow("title",thresh)
# cv2.waitKey(0)
# #2.调用接口处理轮廓
# contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# #3.绘制轮廓
# draw_img = img.copy()
# res = cv2.drawContours(draw_img,contours,-1,(0,0,255),2)#-1表示画出所有的轮廓
# cv2.imshow("title",res)
# cv2.waitKey(0)
# #4.计算轮廓面积
# cnt = contours[0]
# print(cv2.contourArea(cnt))
# #5.计算轮廓周长
# print(cv2.arcLength(cnt,True))






# #轮廓近似
# img  = cv2.imread("noise.jpg")
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
# contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# cnt = contours[1]
# draw_img = img.copy()
# epsilon = 0.005*cv2.arcLength(cnt,True)
# approx = cv2.approxPolyDP(cnt,epsilon,True)
# res = cv2.drawContours(draw_img,[approx],-1,(0,0,255),2)
# cv2.imshow("title",res)
# cv2.waitKey(0)


# # 模板匹配
# img = cv2.imread('lena.jpg', 0)
# template = cv2.imread('face.jpg', 0)
# h, w = template.shape[:2]
# res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# img2 = img.copy()
# top_left = min_loc
# bottom_right = (top_left[0] + w, top_left[1] + h)
# cv2.rectangle(img2, top_left, bottom_right, 255, 2)
# cv2.imshow("title",img2)
# cv2.waitKey(0)

# #模板匹配多个对象
# img_rgb = cv2.imread('mario.jpg')
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# template = cv2.imread('mario_coin.jpg', 0)
# h, w = template.shape[:2]
#
# res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
# threshold = 0.8
# # 取匹配程度大于%80的坐标
# loc = np.where(res >= threshold)
# for pt in zip(*loc[::-1]):  # *号表示可选参数
#     bottom_right = (pt[0] + w, pt[1] + h)
#     cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)
#
# cv2.imshow('img_rgb', img_rgb)
# cv2.waitKey(0)


# #图像直方图
# img = cv2.imread('test.jpg',cv2.IMREAD_GRAYSCALE) #0表示灰度图
# plt.hist(img.ravel(),256)
# plt.show()
# img = cv2.imread('test.jpg')
# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()

# #图像均衡化
# img = cv2.imread('test.jpg',cv2.IMREAD_GRAYSCALE) #0表示灰度图 #clahe
# plt.hist(img.ravel(),256)
# plt.show()
# equ = cv2.equalizeHist(img)
# plt.hist(equ.ravel(),256)
# plt.show()
# res = np.hstack((img,equ))
# cv2.imshow("title",res)
# cv2.waitKey()
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# res_clahe = clahe.apply(img)
# res = np.hstack((img,equ,res_clahe))
# cv2.imshow("title",res)
# cv2.waitKey(0)



# #滤波
#
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# img = cv2.imread('lena.jpg',0)
#
# img_float32 = np.float32(img)
#
# dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
# # 得到灰度图能表示的形式
# magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
#
# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()





# #低通滤波
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# img = cv2.imread('lena.jpg',0)
#
# img_float32 = np.float32(img)
#
# dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
#
# rows, cols = img.shape
# crow, ccol = int(rows/2) , int(cols/2)     # 中心位置
#
# # 低通滤波
# mask = np.zeros((rows, cols, 2), np.uint8)
# mask[crow-30:crow+30, ccol-30:ccol+30] = 1
#
# # IDFT
# fshift = dft_shift*mask
# f_ishift = np.fft.ifftshift(fshift)
# img_back = cv2.idft(f_ishift)
# img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
#
# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
# plt.title('Result'), plt.xticks([]), plt.yticks([])
# plt.show()

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