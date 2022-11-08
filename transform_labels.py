import os


def loadDatadet(infile,k):
    f=open(infile,'r')
    sourceInLine=f.readlines()
    dataset=[]
    for line in sourceInLine:
        temp1=line.strip('\n')
        temp2=temp1.split( )
        dataset.append(temp2)
    for i in range(0,len(dataset)):
        for j in range(k):
            dataset[i].append(float(dataset[i][j]))
        del(dataset[i][0:k])
    f.close()
    return dataset

origin_dataset_path = '/Users/apple/Downloads/yolov5-face-master-2/data/test_data/val/labels'
output_dir_path = '/Users/apple/Downloads/yolov5-face-master-2/data/test_data/val/n_labels'


k=9
if not os.path.exists(output_dir_path):
    os.mkdir(output_dir_path)
dataset = os.listdir(origin_dataset_path)
for file in dataset:

    inputfile = origin_dataset_path + "/" +file
    input = open(inputfile,'r')
    if len(input.read()) == 0:
        outputfile = output_dir_path + "/" + file
        output = open(outputfile, 'w')
        output.close()

    else:
        one_file_data = loadDatadet(inputfile,k)
        outputfile = output_dir_path + "/" + file
        output = open(outputfile, 'w')

        for line in one_file_data:
            names = line[0]
            tempx = line[1:9:2]
            tempy = line[2:9:2]
            xmax = max(tempx)
            ymax = max(tempy)
            xmin = min(tempx)
            ymin = min(tempy)
            xcenter = (xmax+xmin)/2
            ycenter = (ymax+ymin)/2
            w = xmax - xmin
            h = ymax - ymin
            output.write(str(names) + " " + str(round(xcenter,6)) + " " + str(round(ycenter,6)) + " " + str(round(w,6)) + " " + str(round(h,6)))
            for i in range(1,9):
                output.write(" " + str(line[i]))
            #output.write(" ")
            output.write("\n")
        output.close()

