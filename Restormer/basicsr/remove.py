import os


for root, fir, files in os.walk("/data/124-1/datasets/RSR_256"):
    for file in files: 
        if file[-3:] == "png":
            os.remove(os.path.join(root, file))
            print("removing", file)