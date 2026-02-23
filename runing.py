
1. excel generation

2. reading parameters frome xcel 
data =pd.read_csv()

for i, r in data.rows:
    index = i
    p1 = r['p1'][index]
    p2 = r['p2'][index]
    p3 = r['p3'][index]
    acc = r['acc'][index]
    if acc != -1:
        continue # do not to train again

    cmd = f"python model.py {p1} {p2} {p3} " + str(index)
    os.system(cmd)
#finish