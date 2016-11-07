# -*- coding: utf-8 -*-
import numpy as np

def read_cvst_ori():
    data = []
    data_file = open('cvst2.txt')
    data_file.readline()
    
    for line in data_file.readlines():
        
        line = line.strip()
        vec = line.split('\t')
        print(len(vec))
        
        for index,elem in enumerate(vec):
            #==================PAGE1==================
            if index == 3:#age
                vec[3] = int(elem) /100.
            if index == 4:#sex
                vec[4] = float(elem)
            if index == 5:#perinatalstage
                if int(elem) == -1:vec[5] = 0.0
                elif int(elem) == 1 :vec[5] = 1.0
                else: vec[index] = 0.0
            if (index >5 and index <26):# medical1-13,medicine,contraceptive,faceinfections,otitismedia,mastoiditis,meningitis,nasosinusitis
                if elem == 'NULL':vec[index]=0.0
                elif elem == 'on':vec[index]=1.0    
                else: vec[index] = 0.0
            #==================PAGE2==================
            if index == 26:#headache
                if int(elem) == 1:vec[index] = 1.0
                elif int(elem) == -1:vec[index] = 0.0
            if (index > 26 and index < 30):#headachepro1,2,3
                if elem == 'NULL':vec[index] = 0.0
                elif elem == 'on':vec[index] = 1.0
                else: vec[index] = 0.0
            if index == 30:#vomit
                if int(elem) == -1:vec[index] = 0.0
                elif int(elem) ==  1:vec[index] = 1.0
                else: vec[index] = 0.0
            if index == 31:#vomitpro
                if elem == 'NULL':vec[index] = 0.0
                elif int(elem) == 1:vec[index] = 1.0
                else: vec[index] = 0.0
            if index == 32:#awareness
                vec[index] = (float(elem) +1)/4.
            if index == 33:#epilepsy
                if int(elem) == -1:vec[index] = 0.0
                elif int(elem) == 1:vec[index] = 1.0
                else: vec[index] = 0.0
            if index == 34:#epilepsypro
                if elem == 'NULL':vec[index] = 0.0
                elif int(elem) == 0:vec[index] = 0.5
                elif int(elem) == 1:vec[index] = 1.0
                else: vec[index] = 0.0
            if index == 35:#temperature
                if int(elem) == -1:vec[index] = 0.0
                elif int(elem) == 1:vec[index] = 0.5
                elif int(elem) == 2:vec[index] = 1.0
                else: vec[index] = 0.0
            if index == 36:#visiondisorder
                vec[index] = (float(elem) + 1)/4.
            if index == 37:#facialparalysis
                vec[index] = (float(elem) + 1)/2.
            if index == 38:#facialparalysispro
                if elem == 'NULL':vec[index] = 0.0
                elif int(elem) == 0:vec[index] = 0.5
                elif int(elem) == 1:vec[index] = 1.0
                else: vec[index] = 0.0
            if index == 39:#tinnitus
                vec[index] = (float(elem) + 1)/2.
            if index == 40:#tinnituspro
                if elem == 'NULL':vec[index] = 0.0
                elif int(elem) == 0:vec[index] = 0.5
                elif int(elem) == 1:vec[index] = 1.0
                else: vec[index] = 0.0
            if index == 41:#neckdiscomfort
                if int(elem) == -1:vec[index] = 0.0
                elif int(elem) ==  1:vec[index] = 1.0
            if index == 42:#neckdiscomfortpro
                if elem == 'NULL':vec[index] = 0.0
                elif int(elem) == 0:vec[index] = 0.5
                elif int(elem) == 1:vec[index] = 1.0
            if index == 43:#eyebilges
                vec[index] = (float(elem) + 1)/2.
            if index == 44:#dyskinesia
                vec[index] = (float(elem) + 1)/2.
            if index == 45:#dyskinesiapro
                if elem == 'NULL':vec[index] = 0.0
                elif int(elem) == 0:vec[index] = 0.33
                elif int(elem) == 1:vec[index] = 0.66
                elif int(elem) == 2:vec[index] = 1.0
            if index == 46:#papilledema
                vec[index] = (float(elem)+1)/2.
            #==================PAGE3==================
            if index == 47:#wbc
                if elem == 'NULL':vec[index] = 0.0;continue
                if elem == '':vec[index] = 0.0;continue
                vec[index] = (float(elem)) /10.
            if index == 48:#rbc
                if elem == 'NULL':vec[index] = 0.0;continue
                if elem == '':vec[index] = 0.0;continue
                vec[index] = (float(elem)) / 5.
            if index == 49:#hgb
                if elem == 'NULL':vec[index] = 0.0;continue
                if elem == '':vec[index] = 0.0;continue
                vec[index] = (float(elem)) / 150.
            if index == 50:#plt
                if elem == 'NULL':vec[index] = 0.0;continue
                if elem == '':vec[index] = 0.0;continue
                vec[index] = (float(elem)) / 300.
            if index == 51:#rdw
                if elem == 'NULL':vec[index] = 0.0;continue
                if elem == '':vec[index] = 0.0;continue
                vec[index] = (float(elem)) / 15.
            if index == 52:#fib
                if elem == 'NULL':vec[index] = 0.0;continue
                if elem == '':vec[index] = 0.0;continue
                vec[index] = (float(elem)) / 4.
            if index == 53:#ddimer
                if elem == 'NULL':vec[index] = 0.0;continue
                if elem == '':vec[index] = 0.0;continue
                vec[index] = (float(elem)) / 0.5
            if index == 54:#tg
                if elem == 'NULL':vec[index] = 0.0;continue
                if elem == '':vec[index] = 0.0;continue
                vec[index] = (float(elem)) / 2.25
            if index == 55:#ox_ldl            
                if elem == 'NULL':vec[index] = 0.0;continue
                if elem == '':vec[index] = 0.0;continue
                vec[index] = (float(elem)) / 3.36
            if index == 56:#hdl
                if elem == 'NULL':vec[index] = 0.0;continue
                if elem == '':vec[index] = 0.0;continue
                vec[index] = (float(elem)) / 3.
            if index == 57:#tcho
                if elem == 'NULL':vec[index] = 0.0;continue
                if elem == '':vec[index] = 0.0;continue
                vec[index] = (float(elem)) / 5.17
            if index == 58:#plp
                if elem == 'NULL':vec[index] = 0.0;continue
                if elem == '':vec[index] = 0.0;continue
                vec[index] = (float(elem)) / 180.
            if index == 59:#csfroutine
                if int(elem) == -1:vec[index] = 0.0
                elif int(elem) == 1:vec[index] = 1.0
            if index == 60:#csfsugar
                if elem == 'NULL':vec[index] = 0.0;continue
                if elem == '':vec[index] = 0.0;continue
                vec[index] = (float(elem)) / 80.
            if index == 61:#csfchlorine
                if elem == 'NULL':vec[index] = 0.0;continue
                if elem == '':vec[index] = 0.0;continue
                vec[index] = (float(elem)) / 130.
            if index == 62:#csfprotein
                if elem == 'NULL':vec[index] = 0.0;continue
                if elem == '':vec[index] = 0.0;continue
                vec[index] = (float(elem)) / 40.
        # PAGE4 NULL
        data.append(vec[3:63])
    return data
        
def denoising_simple():
    
    pass
    
def one_hot_encoder(label,class_count):
    mat = np.asarray(np.zeros(class_count),dtype='float64').reshape(1,class_count)
    for i in range(class_count):
        if i == label:
            mat[0][i] = 1
    return mat.flatten()

if __name__ == '__main__':
    data = read_cvst_ori()
    data = np.asarray(data)
    print data
    print data.shape
