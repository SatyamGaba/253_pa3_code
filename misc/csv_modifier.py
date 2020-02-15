import csv

Files = ["train.csv", "val.csv", "test.csv"]
for file in Files:
    data = []
    with open(str(file),"r") as csv_file:
            reader = csv.reader(csv_file)
            next(reader) # skip heading line
            l = next(reader)
            for l in reader:
                data.append(["./.."+str(l[0]), "./.."+str(l[1])])

    csv_mod = file.split('.')[0]+"_mod.csv"
    with open(csv_mod,'w',newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["images", "labels"])
        for d in data:
            writer.writerow([d[0], d[1]])