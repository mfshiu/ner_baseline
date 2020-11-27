def main1():
    infile = open('fruits.dat', 'r')
    lines = infile.readlines()
    infile.close()
    for line in lines:
        print(line)

def main2():
    with open('fruits.dat', 'r') as infile
        lines = infile.readlines()
    for line in lines:
        print(line)