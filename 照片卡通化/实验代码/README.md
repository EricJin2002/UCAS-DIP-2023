# 照片卡通化

## 环境配置

需要安装 `python` 以及 `juptyer notebook` 环境。

用到了 `numpy`，`opencv`，`matplotlib` 这三个库，无特定版本要求。

测试机器上安装的版本如下：
- `python=3.7.7`
- `opencv-contrib-python=3.4.2`
- `numpy=1.21.5`
- `matplotlib=3.4.3`

## 文件结构

```
实验代码
├── cartoonize.py           # 卡通化算法实现
├── demo.ipynb              # 卡通化算法演示
├── images                  # 测试图片
│   ├── 10.jpg
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   ├── 5.jpg
│   ├── 6.jpg
│   ├── 7.jpg
│   ├── 8.jpg
│   └── 9.jpg
├── output                  # 卡通化结果
│   ├── 10_cartoonized.jpg
│   ├── 1_cartoonized.jpg
│   ├── 2_cartoonized.jpg
│   ├── 3_cartoonized.jpg
│   ├── 4_cartoonized.jpg
│   ├── 5_cartoonized.jpg
│   ├── 6_cartoonized.jpg
│   ├── 7_cartoonized.jpg
│   ├── 8_cartoonized.jpg
│   └── 9_cartoonized.jpg
├── README.md
├── report.md               # 实验报告 markdown 源文件
└── report.pdf              # 实验报告 pdf 版本
```

## 代码使用说明

卡通化的主体函数位于`cartoonize.py`文件内：

```python
def cartoonize(img, k=16, plot=True)
```

其中 `img` 为输入图片，`k` 为颜色量化数，`plot` 控制是否绘制结果，返回值为卡通化后的图片。

具体使用方法见 `demo.ipynb`。

