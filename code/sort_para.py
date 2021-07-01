import re

text = open("optuna.txt", "r").read()
_re = re.compile('finished with value: (.*?) and parameters: {(.*?)}')
lst = re.findall(_re,text)
for para in lst:
    para = list(para)
    para[0] = float(para[0])
lst.sort()
for para in lst:
    print(para)