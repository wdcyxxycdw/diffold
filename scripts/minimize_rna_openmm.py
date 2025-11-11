#!/usr/bin/env python3
"""
RNA 结构 Amber 能量最小化脚本
使用 rhofold 的 AmberRelaxation 类进行结构优化
"""
import sys
import os
import logging

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rhofold.relax.relax import AmberRelaxation

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # 解析命令行参数
    if len(sys.argv) > 1:
        in_pdb = sys.argv[1]
    else:
        in_pdb = os.path.join(os.path.dirname(__file__), "8uyg_A_best.pdb")
    
    if len(sys.argv) > 2:
        out_pdb = sys.argv[2]
    else:
        base_name = os.path.splitext(in_pdb)[0]
        out_pdb = f"{base_name}_relaxed.pdb"
    
    if len(sys.argv) > 3:
        max_iterations = int(sys.argv[3])
    else:
        max_iterations = 4000
    
    # 检查输入文件是否存在
    if not os.path.exists(in_pdb):
        logger.error(f"输入文件不存在: {in_pdb}")
        sys.exit(1)
    
    logger.info(f"输入文件: {in_pdb}")
    logger.info(f"输出文件: {out_pdb}")
    logger.info(f"最大迭代次数: {max_iterations}")
    
    # 创建 AmberRelaxation 实例
    # 使用 rhofold 现有的 Amber 弛豫功能
    # 会自动添加氢原子、溶剂，进行能量最小化，然后移除氢原子
    amber_relax = AmberRelaxation(
        max_iterations=max_iterations,
        use_gpu=False,  # 如果有 GPU 可以设置为 True
        logger=logger
    )
    
    # 执行能量最小化
    logger.info("开始 Amber 能量最小化...")
    amber_relax.process(in_pdb, out_pdb)
    
    logger.info(f"✓ 完成！最小化后的结构已保存到: {out_pdb}")

if __name__ == '__main__':
    main()
