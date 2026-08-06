#!/usr/bin/env python
# @File       : seed.py
# @Path       : utils/seed.py
# @Author     : 刘赋平
# @Date       : 2026-08-06 15:36:10
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/8/6 | 刘赋平 | v1.0.0 | 初始化创建
import random
import numpy as np
import torch


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
