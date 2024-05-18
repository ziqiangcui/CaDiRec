# __init__.py

# 可以在 __init__.py 中添加一些初始化代码

# 导入其他模块或子包
from . import modules
from . import sasrec
from . import sasrec_new

# 定义导入时执行的初始化函数或变量
# print("Initializing my package...")

# 定义 __all__ 变量，指定在使用 `from package import *` 导入时应导入的模块或变量
__all__ = ['modules', 'sasrec', "sasrec_new"]