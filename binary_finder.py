import subprocess as sp
def find_binaries(fw_path):

    # cmd = f"find '{fw_path}' "

    cmd = f"find '{fw_path}' -type f ! -xtype l -ls -exec file {{}} \; | grep ELF |cut -d: -f1 | grep -v 'lib' | awk -F':' '{{print $1}}'"

    p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)

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
