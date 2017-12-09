# Datasets
The data is not included in git due to its enormous size. To download the datasets, use the links in the items below.

### **[Caltech 101 dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)** - _Object Recognition_

- 1 directory per category
- 101 Categories
- 40-800 images per category. (Usually 50)
- Image sizes: ~ 300x200px

### **[Office dataset](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)** - _Domain Adaptation_

- 3 sub-datasets: Amazon, DSLR, Webcam
- 31 object categories
- Image sizes: ~ 1000x1000px

### **[Birds dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)** - _Subcategory Recognition_

- Version 2011
- 200 bird categories
- 6033 images
- Directory organization:
    - images/
        The images organized in subdirectories based on species. See
        IMAGES AND CLASS LABELS section below for more info.
    - parts/
        15 part locations per image. See PART LOCATIONS section below
        for more info.
    - attributes/
        322 binary attribute labels from MTurk workers. See ATTRIBUTE LABELS
        section below for more info.

#### IMAGES AND CLASS LABELS:

Images are contained in the directory images/, with 200 subdirectories (one for each bird species)

##### List of image files (images.txt)
The list of image file names is contained in the file images.txt, with each line corresponding to one image:

<image_id> <image_name>



##### Train/test split (train_test_split.txt)
The suggested train/test split is contained in the file train_test_split.txt, with each line corresponding to one image:

<image_id> <is_training_image>

where <image_id> corresponds to the ID in images.txt, and a value of 1 or 0 for <is_training_image> denotes that the file is in the training or test set, respectively.



##### List of class names (classes.txt)
The list of class names (bird species) is contained in the file classes.txt, with each line corresponding to one class:

<class_id> <class_name>



##### Image class labels (image_class_labels.txt)
The ground truth class labels (bird species labels) for each image are contained in the file image_class_labels.txt, with each line corresponding to one image:

<image_id> <class_id>

where <image_id> and <class_id> correspond to the IDs in images.txt and classes.txt, respectively.


### **SUN-397** - _Scene Recognition_

TODO (ani) - This is 39GB, we need to think a smart way to handle this.
