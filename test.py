import os
from ctypes import cdll

def find_files(filename, search_path):
   result = []

# Wlaking top-down from the root
   for root, dir, files in os.walk(search_path):
      if filename in files:
         result.append(os.path.join(root, filename))
   return result

dlls = find_files("MLCore.dll", "./")

if(len(dlls) == 0):
   print("DLL not found. Exiting.")
   exit()

dllPath = dlls[0]
print("Using '" + dllPath + "'.")
libc = cdll.LoadLibrary(dllPath)
print("X = " + str(libc.infos()))
