# License-Plate-Recognition
License Plate Recognition For Car With Python And OpenCV

#### 用python3+opencv3做的中国车牌识别，包括算法和客户端界面，只有2个文件，surface.py是界面代码，predict.py是算法代码，界面不是重点所以用tkinter写得很简单。

### 使用方法：
版本：python3.4.4，opencv3.4和numpy1.14和PIL5<br>
下载源码，并安装python、numpy、opencv的python版、PIL，运行surface.py即可

### 算法实现：
算法思想来自于网上资源，先使用图像边缘和车牌颜色定位车牌，再识别字符。车牌定位在predict方法中，为说明清楚，完成代码和测试后，加了很多注释，请参看源码。车牌字符识别也在predict方法中，请参看源码中的注释，需要说明的是，车牌字符识别使用的算法是opencv的SVM， opencv的SVM使用代码来自于opencv附带的sample，StatModel类和SVM类都是sample中的代码。SVM训练使用的训练样本来自于github上的EasyPR的c++版本。由于训练样本有限，你测试时会发现，车牌字符识别，可能存在误差，尤其是第一个中文字符出现的误差概率较大。源码中，我上传了EasyPR中的训练样本，在train\目录下，如果要重新训练请解压在当前目录下，并删除原始训练数据文件svm.dat和svmchinese.dat。

##### 额外说明：算法代码只有500行，测试中发现，车牌定位算法的参数受图像分辨率、色偏、车距影响（test目录下的车牌的像素都比较小，其他图片很可能因为像素等问题识别不了，识别其他像素的车牌需要修改config文件里面的参数，此项目仅是抛砖引玉，提供一个思路）。
##### 有任何疑问请邮件至 wzh191920@sina.com

##### 界面效果：
![](https://github.com/wzh191920/License-Plate-Recognition/blob/master/Screenshots/3.png)
![](https://github.com/wzh191920/License-Plate-Recognition/blob/master/Screenshots/5.png)
