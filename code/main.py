# -*- coding: utf-8 -*-

import time

from task.task import ObjectRecognitionTask

if __name__ == "__main__":
    obejct_recognition_task = ObjectRecognitionTask()
    obejct_recognition_task.train()
    obejct_recognition_task.test()