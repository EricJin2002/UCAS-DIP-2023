# 图像去雾

## 环境配置

需要安装 `python` 以及 `juptyer notebook` 环境。

用到了 `numpy`，`opencv`，`matplotlib` 这三个库，无特定版本要求。

测试机器上安装的版本如下：
- `python=3.7.12`
- `opencv=4.6.0`
- `numpy=1.21.5`
- `matplotlib-inline=0.1.6`

## 文件结构

```plain
实验代码
├── demo.ipynb          # 去雾算法演示
├── doc
│   └── ...             # 实验报告内嵌图
├── haze_removal.py     # 去雾算法实现
├── Org images
│   └── ...             # 待去雾的图片及参考结果
├── output
│   └── ...             # demo.ipynb 输出的去雾结果
├── README.md
├── report.md           # 实验报告 markdown 源文件
└── report.pdf          # 实验报告 pdf 版本
```

## 代码使用说明

使用方式参考 `demo.ipynb`。