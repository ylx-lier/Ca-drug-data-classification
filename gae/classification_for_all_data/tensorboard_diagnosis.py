#!/usr/bin/env python3
"""
TensorBoard连接诊断工具
"""

import subprocess
import requests
import socket
import time
import os
import sys

def check_tensorboard_process():
    """检查TensorBoard进程"""
    print("=== TensorBoard进程检查 ===")
    
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        tensorboard_lines = [line for line in result.stdout.split('\n') if 'tensorboard' in line and 'grep' not in line]
        
        if tensorboard_lines:
            print("✅ TensorBoard进程正在运行:")
            for line in tensorboard_lines:
                print(f"  {line}")
            return True
        else:
            print("❌ TensorBoard进程未运行")
            return False
    except Exception as e:
        print(f"❌ 检查进程失败: {e}")
        return False

def check_port():
    """检查端口状态"""
    print("\n=== 端口状态检查 ===")
    
    try:
        # 检查端口是否被监听
        result = subprocess.run(['netstat', '-tulpn'], capture_output=True, text=True)
        port_6006 = [line for line in result.stdout.split('\n') if ':6006' in line]
        
        if port_6006:
            print("✅ 端口6006正在监听:")
            for line in port_6006:
                print(f"  {line}")
            return True
        else:
            print("❌ 端口6006未监听")
            return False
    except Exception as e:
        print(f"❌ 检查端口失败: {e}")
        return False

def check_local_connection():
    """检查本地连接"""
    print("\n=== 本地连接检查 ===")
    
    urls_to_test = [
        'http://localhost:6006',
        'http://127.0.0.1:6006',
        'http://0.0.0.0:6006'
    ]
    
    for url in urls_to_test:
        try:
            print(f"测试 {url} ...", end=' ')
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("✅ 连接成功")
                return True
            else:
                print(f"❌ HTTP {response.status_code}")
        except requests.exceptions.ConnectError:
            print("❌ 连接被拒绝")
        except requests.exceptions.Timeout:
            print("❌ 连接超时")
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    return False

def check_firewall():
    """检查防火墙状态"""
    print("\n=== 防火墙检查 ===")
    
    try:
        # 检查iptables
        result = subprocess.run(['iptables', '-L', '-n'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            port_rules = [line for line in lines if '6006' in line]
            if port_rules:
                print("⚠️  发现6006端口的防火墙规则:")
                for rule in port_rules:
                    print(f"  {rule}")
            else:
                print("✅ 没有发现6006端口的阻止规则")
        else:
            print("ℹ️  无法检查iptables (可能需要root权限)")
    except FileNotFoundError:
        print("ℹ️  iptables命令不存在")
    except Exception as e:
        print(f"❌ 防火墙检查失败: {e}")

def get_network_info():
    """获取网络信息"""
    print("\n=== 网络信息 ===")
    
    try:
        # 获取主机名
        hostname = socket.gethostname()
        print(f"主机名: {hostname}")
        
        # 获取IP地址
        ip_result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        if ip_result.returncode == 0:
            ips = ip_result.stdout.strip().split()
            print(f"IP地址: {', '.join(ips)}")
            
            print("\n可能的访问地址:")
            for ip in ips:
                print(f"  http://{ip}:6006")
        
    except Exception as e:
        print(f"❌ 获取网络信息失败: {e}")

def test_with_curl():
    """使用curl测试连接"""
    print("\n=== cURL连接测试 ===")
    
    urls = ['http://localhost:6006', 'http://127.0.0.1:6006']
    
    for url in urls:
        try:
            print(f"cURL测试 {url} ...", end=' ')
            result = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', url], 
                                 capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                http_code = result.stdout.strip()
                if http_code == '200':
                    print("✅ cURL连接成功")
                    return True
                else:
                    print(f"❌ HTTP {http_code}")
            else:
                print("❌ cURL失败")
        except subprocess.TimeoutExpired:
            print("❌ cURL超时")
        except Exception as e:
            print(f"❌ cURL错误: {e}")
    
    return False

def suggest_solutions():
    """建议解决方案"""
    print("\n=== 解决方案建议 ===")
    
    print("1. 重启TensorBoard:")
    print("   pkill -f tensorboard")
    print("   tensorboard --logdir=../../results/exp114/tensorboard --port=6006 --host=0.0.0.0")
    print()
    
    print("2. 尝试不同端口:")
    print("   tensorboard --logdir=../../results/exp114/tensorboard --port=6007 --host=0.0.0.0")
    print()
    
    print("3. 使用本地访问:")
    print("   tensorboard --logdir=../../results/exp114/tensorboard --port=6006 --host=127.0.0.1")
    print()
    
    print("4. 检查浏览器:")
    print("   - 清除浏览器缓存")
    print("   - 尝试无痕模式")
    print("   - 尝试不同浏览器")
    print()
    
    print("5. 使用VS Code内置Simple Browser:")
    print("   Ctrl+Shift+P -> 'Simple Browser' -> 输入 http://localhost:6006")

def restart_tensorboard():
    """重启TensorBoard"""
    print("\n=== 重启TensorBoard ===")
    
    try:
        # 停止现有TensorBoard
        print("停止现有TensorBoard进程...")
        subprocess.run(['pkill', '-f', 'tensorboard'], capture_output=True)
        time.sleep(2)
        
        # 重新启动
        print("重新启动TensorBoard...")
        logdir = "../../results/exp114/tensorboard"
        if os.path.exists(logdir):
            cmd = ['tensorboard', '--logdir', logdir, '--port', '6006', '--host', '0.0.0.0']
            print(f"执行命令: {' '.join(cmd)}")
            
            # 在后台启动
            with open('/dev/null', 'w') as devnull:
                subprocess.Popen(cmd, stdout=devnull, stderr=devnull)
            
            time.sleep(3)
            print("✅ TensorBoard已重启")
            return True
        else:
            print(f"❌ 日志目录不存在: {logdir}")
            return False
            
    except Exception as e:
        print(f"❌ 重启失败: {e}")
        return False

def main():
    print("🔧 TensorBoard连接诊断工具")
    print("=" * 50)
    
    # 基础检查
    process_ok = check_tensorboard_process()
    port_ok = check_port()
    
    if not process_ok:
        print("\n⚠️  TensorBoard未运行，尝试重启...")
        if restart_tensorboard():
            print("请重新运行诊断工具检查状态")
        return
    
    # 连接测试
    local_ok = check_local_connection()
    curl_ok = test_with_curl()
    
    # 系统检查
    check_firewall()
    get_network_info()
    
    # 建议解决方案
    if not local_ok and not curl_ok:
        print("\n❌ 本地连接失败")
        suggest_solutions()
    else:
        print("\n✅ 连接测试通过，问题可能在浏览器端")
        print("\n建议:")
        print("1. 清除浏览器缓存并刷新")
        print("2. 尝试无痕模式")
        print("3. 检查浏览器控制台错误信息")
        print("4. 使用VS Code内置Simple Browser")

if __name__ == "__main__":
    main()
