import subprocess
import time
import sys
import os

print("Program loading, this may take up to a minute...")
time.sleep(5)  # Optional pause so message is seen

# Locate the main executable
exe_path = os.path.join(os.path.dirname(sys.executable), "alita/alita.exe")

# Start the main GUI app
subprocess.Popen([exe_path], shell=True)