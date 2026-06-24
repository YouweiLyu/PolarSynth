import subprocess

def run(args):
    
    output = subprocess.run(args, 
        stdout = subprocess.PIPE, stderr = subprocess.PIPE, text=True,
    )
    
    if output.returncode:
        print(output.stdout)
        print(output.stderr)
        exit()
