#!/usr/bin/env python
# @File       : sampler.py
# @Path       : layers/sampler.py
# @Author     : 刘赋平
# @Date       : 2026-08-09 09:40:13
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/9 | 刘赋平 | v1.0.0 | 初始化创建
"""
GraphSAGE Neighbor Sampler

对应论文:
Sample and Aggregate

功能:
给定节点，随机采样邻居
"""
import random
class NeighborSampler:
    def __init__(
            self,
            adj_lists,
            num_samples
    ):
        self.adj_lists = adj_lists
        # 每个节点采样多少邻居
        self.num_samples = num_samples
    def sample(self, nodes):
        """
        nodes:
        一批节点
        example:
        [0,1,2]
        return:
        {
          0:[633,1862],
          1:[...]
        }
        """
        sampled_neighbors = {}
        for node in nodes:
            neighbors = list(
                self.adj_lists[node]
            )
            # 没有邻居
            if len(neighbors)==0:
                sampled_neighbors[node]=[]
            # 邻居少于采样数量
            elif len(neighbors)<=self.num_samples:
                sampled_neighbors[node]=neighbors
            # 随机采样
            else:
                sampled_neighbors[node]=random.sample(
                    neighbors,
                    self.num_samples
                )
        return sampled_neighbors