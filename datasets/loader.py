#!/usr/bin/env python
# @File       : loader.py.py
# @Path       : datasets/loader.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 15:30:08
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建
from .citation import load_citation


def load_dataset(name):

    citation_datasets = [
        "Cora",
        "Citeseer",
        "PubMed"
    ]


    if name in citation_datasets:

        return load_citation(name)


    raise ValueError(
        f"Dataset {name} not supported"
    )