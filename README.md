## FPTree
FP-growth（频繁模式增长）关联规则学习是一种用于从大型数据库中挖掘频繁项集的方法。我使用Python实现了这个算法并实现FPtree的可视化。

## installation
- python 3.7
- 下载安装[graphviz](https://www.graphviz.org/download/)，并将bin目录添加到环境变量PATH中。
- 安装依赖包如下
```
pip install pygraphviz-1.5-cp37-cp37m-win_amd64.whl
pip install -r requirements.txt
```
如需安装其他版本的pygraphviz，可以到[https://github.com/CristiFati/Prebuilt-Binaries/tree/master/PyGraphviz](https://github.com/CristiFati/Prebuilt-Binaries/tree/master/PyGraphviz)下载对应版本的whl文件，并安装。
## Demo
[基于fp-tree实现购物篮可视化分析](demo/基于fp-tree实现购物篮分析【可视化】.md)
