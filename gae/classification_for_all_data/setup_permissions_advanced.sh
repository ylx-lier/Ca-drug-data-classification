#!/bin/bash
# setup_permissions_advanced.sh - 高级权限设置脚本，支持交互式查看

echo "🔧 设置脚本文件执行权限..."

# 脚本文件列表
SCRIPTS=(
    "main.py"
    "start_tensorboard.sh"
    "view_results.py"
    "runall.sh"
)

# 遍历并设置权限
for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        chmod +x "$script"
        echo "✅ $script - 权限已设置"
    else
        echo "⚠️  $script - 文件不存在"
    fi
done

echo ""
echo "📋 当前脚本文件权限:"

# 获取文件数量
sh_count=$(ls -1 *.sh 2>/dev/null | wc -l)
py_count=$(ls -1 *.py 2>/dev/null | wc -l)
total_count=$((sh_count + py_count))

if [ "$total_count" -eq 0 ]; then
    echo "  无脚本文件"
    exit 0
fi

# 定义显示函数
show_files() {
    local file_type="$1"
    local pattern="$2"
    local count="$3"
    
    if [ "$count" -gt 0 ]; then
        echo "${file_type} (共${count}个):"
        ls -la $pattern 2>/dev/null
        echo ""
    fi
}

# 根据文件数量决定显示方式
if [ "$total_count" -le 20 ]; then
    # 文件少，直接全部显示
    echo "所有脚本文件:"
    ls -la *.sh *.py 2>/dev/null | grep -E '\.(sh|py)$'
else
    # 文件多，提供选择
    echo "检测到${total_count}个脚本文件 (Shell: ${sh_count}, Python: ${py_count})"
    echo ""
    echo "选择查看方式:"
    echo "1) 查看所有文件 (可能很长)"
    echo "2) 只查看可执行文件"
    echo "3) 分类查看前10个"
    echo "4) 只显示统计信息"
    echo ""
    
    read -p "请选择 (1-4, 默认3): " choice
    choice=${choice:-3}
    
    case $choice in
        1)
            echo ""
            echo "所有脚本文件:"
            ls -la *.sh *.py 2>/dev/null | grep -E '\.(sh|py)$'
            ;;
        2)
            echo ""
            echo "可执行脚本文件:"
            ls -la *.sh *.py 2>/dev/null | grep '^-rwx'
            ;;
        3)
            echo ""
            if [ "$sh_count" -gt 0 ]; then
                echo "Shell脚本文件 (前10个):"
                ls -la *.sh 2>/dev/null | head -10
                [ "$sh_count" -gt 10 ] && echo "  ... 还有$((sh_count - 10))个.sh文件"
                echo ""
            fi
            
            if [ "$py_count" -gt 0 ]; then
                echo "Python脚本文件 (前10个):"
                ls -la *.py 2>/dev/null | head -10
                [ "$py_count" -gt 10 ] && echo "  ... 还有$((py_count - 10))个.py文件"
            fi
            ;;
        4)
            echo ""
            echo "只显示统计信息（跳过文件列表）"
            ;;
        *)
            echo "无效选择，显示前10个文件"
            show_files "Shell脚本文件" "*.sh" "$sh_count"
            show_files "Python脚本文件" "*.py" "$py_count"
            ;;
    esac
fi

# 显示执行权限统计
executable_count=$(ls -la *.sh *.py 2>/dev/null | grep -c '^-rwx')
echo ""
echo "📊 权限统计: ${executable_count}/${total_count} 个脚本可执行"

# 列出不可执行的重要脚本
non_executable=$(ls -la *.sh *.py 2>/dev/null | grep -v '^-rwx' | grep -E '\.(sh|py)$' | awk '{print $9}')
if [ -n "$non_executable" ]; then
    echo ""
    echo "⚠️  不可执行的脚本文件:"
    echo "$non_executable" | while read file; do
        echo "   $file"
    done
fi

echo ""
echo "💡 常用命令:"
echo "   查看全部权限: ls -la *.sh *.py"
echo "   查看可执行文件: ls -la *.sh *.py | grep '^-rwx'"
echo "   设置所有脚本可执行: chmod +x *.sh *.py"

echo ""
echo "✨ 权限设置完成！现在可以直接执行："
echo "  ./main.py"
echo "  ./view_results.py"
echo "  ./start_tensorboard.sh"
