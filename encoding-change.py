# 导入codecs模块，用于处理不同编码的文件
import codecs

g="gb2312"
u="utf-8"

# 设置path变量，指定要转换的文件的路径
path = "E:\\novel\龙族3小说绘版.txt"

# 选择输入的编码格式，可以是utf-8, ansi或者gb2312
input_encoding = g

# 选择输出的编码格式，可以是utf-8, ansi或者gb2312
output_encoding = u

# 用codecs模块打开文件，指定输入的编码格式
with codecs.open(path, "r", encoding=input_encoding) as input_file:
    # 读取文件内容
    content = input_file.read()
    # 用codecs模块创建一个新的文件，指定输出的编码格式
    with codecs.open(path + "_converted.txt", "w", encoding=output_encoding) as output_file:
        # 写入转换后的内容
        output_file.write(content)

# 打印完成提示
print("转换完成！")