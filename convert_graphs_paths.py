import os
import shutil

def convert_graphs_paths(folder_path, backup=True):
    """
    转换指定文件夹中所有txt文件里的"graphs\"为"graphs/"
    
    参数:
        folder_path (str): 要处理的文件夹路径
        backup (bool): 是否备份原文件，默认为True
    """
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 只处理txt文件
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            print(f"处理文件: {file_path}")
            
            try:
                # 备份原文件
                if backup:
                    backup_path = f"{file_path}.bak"
                    shutil.copy2(file_path, backup_path)
                    print(f"已创建备份: {backup_path}")
                
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # 替换路径格式
                modified_content = content.replace('graphs\\', 'graphs/')
                
                # 如果内容有变化才写入
                if modified_content != content:
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(modified_content)
                    print(f"已更新文件: {file_path}")
                else:
                    print(f"文件无需更新: {file_path}")
            
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")
    
        print("处理完成")

if  __name__ == "__main__":
    # 在这里指定要处理的文件夹路径
    target_folder = input("请输入要处理的文件夹路径: ")
    convert_graphs_paths(target_folder, backup=False)