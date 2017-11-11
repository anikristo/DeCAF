# Datasets
To download the datasets in this folder:
```bash
scp -r <username>@ssh.cs.brown.edu:/home/akristo/course/cs1470/project/datasets/ .
```

### **[Caltech 101 dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)** - _Object Recognition_

- 1 directory per category
- 101 Categories
- 40-800 images per category. (Usually 50)
- Image sizes: ~ 300x200px

### **[Office dataset](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)** - _Domain Adaptation_

- 3 sub-datasets: Amazon, DSLR, Webcam
- 31 object categories
- Image sizes: ~ 1000x1000px

### **[Birds dataset](http://www.vision.caltech.edu/visipedia/CUB-200.html)** - _Subcategory Recognition_

- 200 bird categories
- 6033 images
- Directory organization:
  - `images/`: The images organized in subdirectories based on species.
  - `attributes/`: Attribute data from MTurk workers. See README.txt in directory for more info.
  - `lists/`:
      - `classes.txt` : list of categories (species)
      - `files.txt`   : list of all image files (including - subdirectories)
      - `train.txt`   : list of all images used for training
      - `test.txt`    : list of all images used for testing

### **SUN-397** - _Scene Recognition_

TODO (ani) - This is 39GB, we need to think a smart way to handle this.
