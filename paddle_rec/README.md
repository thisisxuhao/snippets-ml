## 基于paddle推荐算法

- 对PaddleRec[https://gitee.com/paddlepaddle/PaddleRec]中demo的一个消化学习   

$$
\hat{y}(x):=\underbrace{w_{i=1}^n w_i x_i}_{\text {线性回归 }}+\underbrace{\sum_{i=1}^n \sum_{j=i+1}^n w_{i j} x_i x_j}_{\text {交双烦 (组合待征) }}
$$

​		维度高时会导致参数爆炸，同时很多维度是不相关的，学习起来比较困难。

​		将wij矩阵W近似看为两个n*k阶矩阵的乘积：
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



​			最优化求解可以使用SGC, ALS, MCMC等方法。以最简单的梯度下降为例，v的梯度下降求解方式为：

​			
$$
\frac{\partial}{\partial v_{i, f}} \hat{y}(\mathbf{x})=  x_i \sum_{j=1}^n v_{j, f} x_j-v_{i, f} ||x_i||_{2}
$$
​			此外wi的梯度就是xi（常数项xi=1）。



## 参考

- [原文论文](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.393.8529&rep=rep1&type=pdf)
- [libFM](http://www.libfm.org/)
- 

https://gitee.com/paddlepaddle/PaddleRec

