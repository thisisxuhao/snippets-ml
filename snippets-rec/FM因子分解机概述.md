# FM因子分解机概述

## 简介

- FM是推荐算法领域一个有里程碑意义的算法
- FM算法又有延伸

$$
\hat{y}(x):=\underbrace{w_{i=1}^n w_i x_i}_{\text {线性回归 }}+\underbrace{\sum_{i=1}^n \sum_{j=i+1}^n w_{i j} x_i x_j}_{\text {交叉项 (组合待征) }}
$$

  	

维度高时会导致参数爆炸，同时很多维度是不相关的，学习起来比较困难。

将wij矩阵W近似看为两个n*k阶矩阵的乘积：


$$
\hat{\mathbf{W}}=\mathbf{V} \mathbf{V}^T=\left(\begin{array}{c}
\mathbf{v}_1 \\
\mathbf{v}_2 \\
\vdots \\
\mathbf{v}_n
\end{array}\right)\left(\begin{array}{llll}
\mathbf{v}_1^T & \mathbf{v}_2^T & \cdots & \mathbf{v}_n^T
\end{array}\right)
$$

$$
\begin{aligned}
交叉相部分= \\
& \sum_{i=1}^{n-1} \sum_{j=i+1}^n\left\langle\mathbf{v}_i, \mathbf{v}_j\right\rangle x_i x_j \\
= & \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n\left\langle\mathbf{v}_i, \mathbf{v}_j\right\rangle x_i x_j-\frac{1}{2} \sum_{i=1}^n\left\langle\mathbf{v}_i, \mathbf{v}_i\right\rangle x_i x_i \\
= & \frac{1}{2}\left(\sum_{i=1}^n \sum_{j=1}^n \sum_{f=1}^k v_{i, f} v_{j, f} x_i x_j-\sum_{i=1}^n \sum_{f=1}^k v_{i, f} v_{i, f} x_i x_i\right) \\
= & \frac{1}{2} \sum_{f=1}^k\left(\left(\sum_{i=1}^n v_{i, f} x_i\right)\left(\sum_{j=1}^n v_{j, f} x_j\right)-\sum_{i=1}^n v_{i, f}^2 ||x_i||_{2}\right) \\
= & \frac{1}{2} \sum_{f=1}^k\left(||\sum_{i=1}^n v_{i, f} x_i||_{2}-\sum_{i=1}^n v_{i, f}^2 ||x_i||_{2}\right)
\end{aligned}
$$



 ## FM求解

### 梯度下降求解

最优化求解可以使用SGC, ALS, MCMC等方法。以最简单的梯度下降为例，y关于v的梯度为：

$$
\frac{\partial}{\partial v_{i, f}} \hat{y}(\mathbf{x})= \left ( x_i \sum_{j=1}^n v_{j, f} x_j \right )-v_{i, f} ||x_i||_{2}
$$

此外y关于wi的梯度就是xi。

假定损失函数为$MSE$, 学习率为$alpha$，则参数求解的梯度下降迭代过程为：


$$
\begin{aligned}
& w_0 = w_0 - \alpha * \frac{\partial L(w_0)}{\partial w_0} \\
& w_i = w_i - \alpha * \frac{\partial L(w_i)}{\partial w_i} \\
&v_{i,f} =v_{i,f} - \alpha * \frac{\partial L(v_{i,f})}{\partial v_{i,f}} \\
\end{aligned}
$$

梯度求解方式如下：     

$$
\begin{aligned}
\frac{\partial L(w_0)}{\partial w_0}
& =  \frac{\partial \sum_{j=1}^n (\hat y_j- y_j)^2}{\partial w_0} \\
&=\sum_{j=1}^n 2*(\hat y_j - y_j)* \frac{\partial \hat y_j}{\partial w_0} \\
&=\sum_{j=1}^n 2*(\hat y_j - y_j)*1 \\
\end{aligned}
$$


$$
\begin{aligned}
\frac{\partial L(w_i)}{\partial w_i}
& =  \frac{\partial \sum_{j=1}^n (\hat y_j- y_j)^2}{\partial w_i} \\
&=\sum_{j=1}^n 2*(\hat y_j - y_j)* \frac{\partial \hat y_j}{\partial w_i} \\
&=\sum_{j=1}^n 2*(\hat y_j - y_j)*x_{ji} \\
\end{aligned}
$$


$$
\begin{aligned}
\frac{\partial L(v_{i,f})}{\partial v_{i,f}}
& =\frac{\partial \sum_{j=1}^n (\hat y_j- y_j)^2}{\partial v_{i,f}} \\
&=\sum_{j=1}^n 2*(\hat y_j - y_j) * \frac{\partial}{\partial v_{i, f}} \hat{y}(\mathbf{x}) \\
&= \sum_{j=1}^n 2*(\hat y_j-y_j) * x_i \left( \sum_{h=1}^n v_{h, f} x_j-v_{h, f} ||x_h||_{2} \right)
\end{aligned}
$$



### 求解DEMO

我实现了一个求解demo代码[fm.py](fm/fm.py)

## FM的意义

- FM与SVM
- FM与矩阵分解
- FM与因子分析




## FM的扩展

- FFM
- DeepFM



## FM的API

- xlearn
- pytorch
- 




## 参考

- [原文论文](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.393.8529&rep=rep1&type=pdf)
- [libFM](http://www.libfm.org/)
- [因子机深入理解](https://tracholar.github.io/machine-learning/2017/03/10/factorization-machine.html)
- [xlearn](https://xlearn-doc-cn.readthedocs.io/en/latest/install/index.html)
- [一文看懂推荐系统：排序08：Factorization Machines（FM）因子分解机，一个特殊的案例就是MF，矩阵分解为uv的乘积](https://blog.csdn.net/weixin_46838716/article/details/126554031)
- [paddle-rec](https://gitee.com/paddlepaddle/PaddleRec)

