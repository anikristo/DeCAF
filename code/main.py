# -*- coding: utf-8 -*-

import time

from task.task import ObjectRecognitionTask, DomainAdaptationTask

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
#    obejct_recognition_task = ObjectRecognitionTask()
#    obejct_recognition_task.train()
#    obejct_recognition_task.test()
    
    domains = {0: "amazon", 1: "dslr", 2: "webcam"}
    domain_adaptation_task = DomainAdaptationTask(origin_domain = "amazon", target_domain = "webcam", combo = "ST")
    domain_adaptation_task.train()
    domain_adaptation_task.test()
    
    domain_adaptation_task = DomainAdaptationTask(origin_domain = "dslr", target_domain = "webcam", combo = "ST")
    domain_adaptation_task.train()
    domain_adaptation_task.test()
    
    