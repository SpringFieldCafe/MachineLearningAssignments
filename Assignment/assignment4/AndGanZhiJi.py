#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import print_function  # 兼容旧版print写法
from functools import reduce  # 用于连续累加


class VectorOp(object):  # 向量运算工具类

    @staticmethod
    def dot(x, y):  # 计算内积
        return reduce(lambda a, b: a + b, VectorOp.element_multiply(x, y), 0.0)

    @staticmethod
    def element_multiply(x, y):  # 两个向量对应位置相乘
        return list(map(lambda x_y: x_y[0] * x_y[1], zip(x, y)))

    @staticmethod
    def element_add(x, y):  # 两个向量对应位置相加
        return list(map(lambda x_y: x_y[0] + x_y[1], zip(x, y)))

    @staticmethod
    def scala_multiply(v, s):  # 向量乘以学习步长
        return map(lambda e: e * s, v)


class Perceptron(object):  # 感知机模型
    def __init__(self, input_num, activator):
        self.activator = activator  # 保存激活函数
        self.weights = [0.0] * input_num  # 初始权重全置0
        self.bias = 0.0  # 初始偏置为0

    def __str__(self):  # 便于打印当前参数
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):  # 按wx+b得到预测值
        return self.activator(
            VectorOp.dot(input_vec, self.weights) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):  # 多轮训练
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):  # 训练一遍样本
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            output = self.predict(input_vec)  # 先用当前参数预测
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):  # 按误差修正参数
        delta = label - output  # 目标值与输出的差
        self.weights = VectorOp.element_add(
            self.weights, VectorOp.scala_multiply(input_vec, rate * delta))
        self.bias += rate * delta  # 同步更新偏置


def f(x):  # 阶跃函数
    return 1 if x > 0 else 0


def get_training_dataset():  # 与运算训练集
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    return input_vecs, labels


def train_and_perceptron():  # 训练一个二输入感知机
    p = Perceptron(2, f)
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p


if __name__ == '__main__':
    and_perception = train_and_perceptron()  # 得到训练后的模型
    print(and_perception)  # 查看权重和偏置
    print('1 and 1 = %d' % and_perception.predict([1, 1]))  # 测试1与1
    print('0 and 0 = %d' % and_perception.predict([0, 0]))  # 测试0与0
    print('1 and 0 = %d' % and_perception.predict([1, 0]))  # 测试1与0
    print('0 and 1 = %d' % and_perception.predict([0, 1]))  # 测试0与1
