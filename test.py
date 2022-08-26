import subprocess
process = subprocess.Popen(['cat', '/proc/device-tree/model'],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
output, error = process.communicate(timeout=3)
if process.returncode != 0: 
   print("bitcoin failed %d %s %s" % (process.returncode, output, error))
print(output)







