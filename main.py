import numpy as np
import importlib.util
import cv2

try:
    cv2 = importlib.import_module('cv2')
except ImportError:
    cv2_spec = importlib.util.find_spec('cv2')#eeeeeeeeeeeeeeeeeeeeeee幹
    cv2 = importlib.util.module_from_spec(cv2_spec)
    cv2_spec.loader.exec_module(cv2)

# 创建一个空函数，稍后会用作回调函数
def do_nothing(x):
    pass

# 创建一个窗口以显示图像kkkk
cv2.namedWindow('Result')

# 创建一个动态调整控制条，可以用来调整阈值
cv2.createTrackbar('Threshold', 'Result', 150, 255, do_nothing)

# 读取图像
image = cv2.imread('yoloimg_01.jpg')

while True:
    # 获取当前阈值的值
    threshold_value = cv2.getTrackbarPos('Threshold', 'Result')

    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用阈值处理将图像转换为二进制图像，以便更容易检测黑点
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # 检测图像中的轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 迭代所有轮廓，并绘制圆圈标记黑点
    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        # 假设黑点的面积很小，请根据您的图像调整此阈值
        if area < 100:
            # 获取轮廓的外接圆
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            # 绘制圆圈标记黑点
            cv2.circle(image, center, radius, (0, 0, 255), 2)

    # 显示结果图像
    cv2.imshow('Result', image)

    # 检查按键事件，如果按下ESC键，退出循环
    key = cv2.waitKey(1)
    if key == 27:
        break

# 关闭所有窗口
cv2.destroyAllWindows()
