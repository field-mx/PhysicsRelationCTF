"""Detached one-job queue; waits only for a specified prior experiment PID."""
import argparse
import os
from pathlib import Path
import time

p=argparse.ArgumentParser();p.add_argument('--after',type=int,required=True)
p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
while True:
    try:
        state=Path(f'/proc/{a.after}/stat').read_text().split(') ',1)[1].split()[0]
        if state=='Z':break
    except FileNotFoundError:break
    time.sleep(5)
command=a.command[1:] if a.command and a.command[0]=='--' else a.command
os.execvp(command[0],command)
