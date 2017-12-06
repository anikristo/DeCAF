# -*- coding: utf-8 -*-

import time

from task.task import ObjectRecognitionTask, DomainAdaptationTask, SubcategoryRecognitionTask, SceneObjectRecognitionTask

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    
    #SCENE RECOGNITION - CALTECH
    obejct_recognition_task = ObjectRecognitionTask()
    obejct_recognition_task.train()
    obejct_recognition_task.test()
    
    #DOMAIN ADAPTATION - OFFICE
    domain_adaptation_task = DomainAdaptationTask(origin_domain = "amazon", target_domain = "webcam", combo = "T")
    domain_adaptation_task.train()
    domain_adaptation_task.test()
    
    domain_adaptation_task = DomainAdaptationTask(origin_domain = "dslr", target_domain = "webcam", combo = "T")
    domain_adaptation_task.train()
    domain_adaptation_task.test()
    
    domain_adaptation_task = DomainAdaptationTask(origin_domain = "amazon", target_domain = "webcam", combo = "S")
    domain_adaptation_task.train()
    domain_adaptation_task.test()
    
    domain_adaptation_task = DomainAdaptationTask(origin_domain = "dslr", target_domain = "webcam", combo = "S")
    domain_adaptation_task.train()
    domain_adaptation_task.test()
    
    domain_adaptation_task = DomainAdaptationTask(origin_domain = "amazon", target_domain = "webcam", combo = "ST")
    domain_adaptation_task.train()
    domain_adaptation_task.test()
    
    domain_adaptation_task = DomainAdaptationTask(origin_domain = "dslr", target_domain = "webcam", combo = "ST")
    domain_adaptation_task.train()
    domain_adaptation_task.test()
    
    #SUBCATEGORY RECOGNITION - BIRDS
    subcategory_recognition_task = SubcategoryRecognitionTask()
    subcategory_recognition_task.train()
    subcategory_recognition_task.test()
    
    #SCENE RECOGNITION - SUN    
    scene_recognition_task = SceneObjectRecognitionTask()
    scene_recognition_task.train()
    scene_recognition_task.test()
    
