import subprocess

# Execute the 'ulimit -n' command and capture the output
output = subprocess.run("ulimit -n", stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)

# Decode the output and print the result
ulimit_value = output.stdout.decode().strip()
print("Current ulimit -n value:", ulimit_value)