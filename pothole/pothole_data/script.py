import os
files = os.listdir('.')
files.remove('script.py')
files.remove('index.txt')
files.remove('lables.txt')
files.sort()
fi = open('index.txt','w')
fl = open('lables.txt','w')
for f in files:
    f = f.strip()
    l = 0
    fi.write(f+'\n');
    if f[0] == 'p' and f[1] == 't':
	fl.write(f[0]+'\n')
    elif f[0] == 'r':
	fl.write('o\n')
    elif f[0] == 'g':
	fl.write('o\n')
    elif f[0] == 'p':
	fl.write('o\n')
    else:
	fl.write('o\n')
fi.close()
fl.close()
