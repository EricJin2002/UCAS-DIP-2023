# 图像拼接

## 环境配置

需要安装 `python` 以及 `juptyer notebook` 环境。

用到了 `numpy`，`opencv`，`matplotlib` 这三个库。

由于用到了 `cv2.xfeatures2d.SIFT_create()`，所以特别注意需要安装的是 `opencv-contrib-python`，并且版本应小于3.4.2（之后的版本移除了non-free模块）。

```bash
pip install opencv-contrib-python==3.4.2.17
```

测试机器上安装的版本如下：
- `python=3.7.7`
- `opencv-contrib-python=3.4.2.17`
- `numpy=1.21.5`
- `matplotlib-inline=0.1.6`

## 文件结构

```plain
实验代码
├── cache               # 存储中间结果，以便复用
│   ├── left_kp1.pkl    # 左图特征点
│   ├── left_kp2.pkl
│   ├── left_kp3.pkl
│   ├── left_kp4.pkl
│   ├── matches1.pkl    # 左右图特征点匹配
│   ├── matches2.pkl
│   ├── matches3.pkl
│   ├── matches4.pkl
│   ├── right_kp1.pkl   # 右图特征点
│   ├── right_kp2.pkl
│   ├── right_kp3.pkl
│   └── right_kp4.pkl
├── demo.ipynb          # 图像拼接算法演示
├── images              # 测试图片
│   ├── left1.jpg
│   ├── left2.jpg
│   ├── left3.jpg
│   ├── left4.jpg
│   ├── right1.jpg
│   ├── right2.jpg
│   ├── right3.jpg
│   └── right4.jpg
├── img_stitching.py    # 图像拼接算法实现
├── output              # 输出结果
│   ├── left1_kp.jpg    # 左图特征点可视化
│   ├── left2_kp.jpg
│   ├── left3_kp.jpg
│   ├── left4_kp.jpg
│   ├── match1.jpg      # 左右图特征点匹配可视化
│   ├── match2.jpg
│   ├── match3.jpg
│   ├── match4.jpg
│   ├── right1_kp.jpg   # 右图特征点可视化
│   ├── right2_kp.jpg
│   ├── right3_kp.jpg
│   ├── right4_kp.jpg
│   ├── stitch1.jpg     # 拼接结果
│   ├── stitch2.jpg
│   ├── stitch3.jpg
│   └── stitch4.jpg
├── README.md
├── report.md           # 实验报告 markdown 源文件
└── report.pdf          # 实验报告 pdf 版本
```

## 代码使用说明

使用方式参见 `demo.ipynb`。

主体函数为：

```python
img_stitching.stitch_img(left, right, save_suffix, save_path="output", cache_path="cache", reversed=False)
```

输入参数：
- `left`：左图数据
- `right`：右图数据
- `save_suffix`：保存文件名后缀（用于区分不同图片）
- `save_path`：特征点及其匹配可视化图像、拼接结果保存路径
- `cache_path`：中间结果缓存路径
- `reversed`：`False` 表示将右图拼接到左图，`True` 表示将左图拼接到右图

返回值：
- `img_out`：拼接结果