import os

with open('classInd.txt', 'r') as f:
    raw_lines = f.readlines()

content = [line.strip().split(' ')[1] for line in raw_lines]
with open('category_ucf101.txt', 'w') as f:
    for i in range(len(content)):
        f.write(content[i]+'\n')
