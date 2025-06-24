#!/usr/bin/env python3
"""
字体安装和管理工具
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import requests
import os
import zipfile
from pathlib import Path

def download_chinese_font():
    """下载并安装中文字体"""
    print("📥 下载中文字体...")
    
    # 创建字体目录
    font_dir = Path.home() / '.matplotlib' / 'fonts'
    font_dir.mkdir(parents=True, exist_ok=True)
    
    # 字体URL (使用开源字体)
    font_urls = {
        'SourceHanSansCN-Regular.otf': 'https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSansCN.zip',
        'NotoSansCJK-Regular.ttc': 'https://github.com/googlefonts/noto-cjk/releases/download/Sans2.004/08_NotoSansCJK-Regular.ttc'
    }
    
    try:
        # 检查是否已经有字体文件
        existing_fonts = list(font_dir.glob('*.otf')) + list(font_dir.glob('*.ttf')) + list(font_dir.glob('*.ttc'))
        if existing_fonts:
            print(f"✅ 找到现有字体文件: {len(existing_fonts)} 个")
            return True
        
        print("⚠️  字体下载功能需要网络连接，跳过自动下载")
        print("💡 建议手动安装中文字体或使用系统自带字体")
        return False
        
    except Exception as e:
        print(f"❌ 字体下载失败: {e}")
        return False

def list_available_fonts():
    """列出可用的字体"""
    print("\n📋 可用字体列表:")
    
    # 获取所有字体
    all_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 查找可能的中文字体
    chinese_keywords = ['Han', 'CJK', 'Chinese', 'SimHei', 'SimSun', 'YaHei', 'KaiTi', 'Noto', 'Source']
    chinese_fonts = []
    
    for font in all_fonts:
        for keyword in chinese_keywords:
            if keyword.lower() in font.lower():
                chinese_fonts.append(font)
                break
    
    if chinese_fonts:
        print("🎨 可能支持中文的字体:")
        for i, font in enumerate(set(chinese_fonts)[:10], 1):
            print(f"  {i}. {font}")
    else:
        print("⚠️  未找到明确的中文字体")
    
    print(f"\n📊 总字体数量: {len(all_fonts)}")
    print(f"🔍 可能的中文字体: {len(set(chinese_fonts))}")

def test_font_rendering():
    """测试字体渲染效果"""
    print("\n🧪 测试字体渲染...")
    
    # 测试字体列表
    test_fonts = [
        'SimHei',
        'Microsoft YaHei', 
        'SimSun',
        'Source Han Sans CN',
        'Noto Sans CJK SC',
        'DejaVu Sans'
    ]
    
    # 测试文本
    test_text = "实验结果可视化仪表板"
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    fig.suptitle('Font Rendering Test / 字体渲染测试', fontsize=14)
    
    axes = axes.flatten()
    
    for i, font in enumerate(test_fonts):
        if i >= len(axes):
            break
            
        ax = axes[i]
        try:
            # 设置字体
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            
            ax.text(0.5, 0.5, test_text, fontsize=12, ha='center', va='center')
            ax.text(0.5, 0.3, f'Font: {font}', fontsize=10, ha='center', va='center', style='italic')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f'Test {i+1}')
            ax.axis('off')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Font Error\n{font}', fontsize=10, ha='center', va='center', color='red')
            ax.set_xlim(0, 1) 
            ax.set_ylim(0, 1)
            ax.set_title(f'Error {i+1}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('font_test.png', dpi=200, bbox_inches='tight')
    print("✅ 字体测试图已保存: font_test.png")
    plt.show()

def get_best_chinese_font():
    """获取最佳的中文字体"""
    # 按优先级排序的字体列表
    preferred_fonts = [
        'Source Han Sans CN',
        'Noto Sans CJK SC', 
        'Microsoft YaHei',
        'SimHei',
        'SimSun',
        'WenQuanYi Micro Hei',
        'AR PL UKai CN',
        'DejaVu Sans'  # 后备字体
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in preferred_fonts:
        if font in available_fonts:
            return font
    
    return 'DejaVu Sans'  # 默认后备

def main():
    print("🔧 中文字体配置工具")
    print("=" * 50)
    
    # 列出可用字体
    list_available_fonts()
    
    # 获取最佳字体
    best_font = get_best_chinese_font()
    print(f"\n🎯 推荐字体: {best_font}")
    
    # 测试字体渲染
    test_font_rendering()
    
    print(f"\n✅ 字体配置完成！")
    print(f"📝 在代码中使用: plt.rcParams['font.sans-serif'] = ['{best_font}']")

if __name__ == "__main__":
    main()
