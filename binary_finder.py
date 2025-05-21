import subprocess as sp
def find_binaries(fw_path):
    # 获取所有文件
    # cmd = f"find '{fw_path}' "
    # 忽略链接类型的文件
    cmd = f"find '{fw_path}' -type f ! -xtype l -ls -exec file {{}} \; | grep ELF |cut -d: -f1 | grep -v 'lib' | awk -F':' '{{print $1}}'"
    # print(cmd)  # find '../firmware/Tenda/analyzed/_US_AC6V1.0BR_V15.03.05.16_multi_TD01.bin.extracted/squashfs-root'
    p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    # print("p", p) # <subprocess.Popen object at 0x7f09cc95a550>
    # 用于从刚刚创建的进程（p）中获取输出（stdout）和错误输出（stderr）。并将它们分别赋值给变量 o 和 e。此时，p 进程将会被阻塞，直到读取完所有的输出数据。
    o, e = p.communicate()
    if o:
        # changed to o.decode() for python3
        return o.decode().split('\n')

    return []


# fw_path = '/home/a123456/Desktop/GetRawFeatures/firmware/DIR-880_ARM/squashfs-root'
# fw_path = '/home/a123456/Desktop/GetRawFeatures/firmware/netgear/R7900-V1_ARM/squashfs-root'
fw_path = '/home/a123456/Desktop/GetRawFeatures/firmware/tenda/_US_AC15V1.0BR_V15.03.05.19_multi_TD01.bin.extracted'
executables = find_binaries(fw_path)
print(executables)
print(" the number of executables:", len(executables))