import os
import time
import subprocess

subprocess.run(["python3 text_ewsn.py"], shell=True)
subprocess.run(['python3 walking.py'], shell=True)
subprocess.run(["python3 arrow.py"], shell=True)
subprocess.run(["python3 walking_text_2.py"], shell=True)

for i in range(3):
	subprocess.run(["python3 area.py"], shell=True)
	subprocess.run(["python3 text_abcd.py"], shell=True)
	subprocess.run(["python3 milk_misson_2.py"], shell=True)
	subprocess.run(["python3 center.py"], shell=True)
	if i == 2:
		break
subprocess.run(["python3 after_misson_walking.py"], shell=True)
subprocess.run(["python3 exit_2.py"], shell=True)

