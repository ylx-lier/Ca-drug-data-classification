#!/bin/bash
# setup_permissions.sh - 一键设置所有脚本文件权限

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

# 智能显示：如果文件少就全显示，文件多就分组显示
sh_count=$(ls -1 *.sh 2>/dev/null | wc -l)
py_count=$(ls -1 *.py 2>/dev/null | wc -l)
total_count=$((sh_count + py_count))

if [ "$total_count" -eq 0 ]; then
    echo "  无脚本文件"
elif [ "$total_count" -le 15 ]; then
    # 文件少，直接全部显示
    echo "所有脚本文件:"
    ls -la *.sh *.py 2>/dev/null | grep -E '\.(sh|py)$'
else
    # 文件多，分组显示并提供统计
    if [ "$sh_count" -gt 0 ]; then
        echo "Shell脚本文件 (共${sh_count}个):"
        if [ "$sh_count" -le 10 ]; then
            ls -la *.sh 2>/dev/null
        else
            ls -la *.sh 2>/dev/null | head -10
            echo "  ... 还有$((sh_count - 10))个.sh文件"
        fi
    fi
    
    echo ""
    if [ "$py_count" -gt 0 ]; then
        echo "Python脚本文件 (共${py_count}个):"
        if [ "$py_count" -le 10 ]; then
            ls -la *.py 2>/dev/null
        else
            ls -la *.py 2>/dev/null | head -10
            echo "  ... 还有$((py_count - 10))个.py文件"
        fi
    fi
fi

# 显示执行权限统计
if [ "$total_count" -gt 0 ]; then
    executable_count=$(ls -la *.sh *.py 2>/dev/null | grep -c '^-rwx')
    echo ""
    echo "📊 权限统计: ${executable_count}/${total_count} 个脚本可执行"
    
    if [ "$total_count" -gt 15 ]; then
        echo ""
        echo "💡 查看全部文件权限: ls -la *.sh *.py"
        echo "💡 查看可执行文件: ls -la *.sh *.py | grep '^-rwx'"
    fi
fi

echo ""
echo "✨ 权限设置完成！现在可以直接执行："
echo "  ./main.py"
echo "  ./view_results.py"
echo "  ./start_tensorboard.sh"
