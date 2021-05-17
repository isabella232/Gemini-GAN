import os 

def open_txt(path):
    data = []
    with open(path, 'r') as f:
        data = [line.strip() for line in f]
    return data

test = open_txt("./test.txt")
valid = open_txt("./valid.txt")
train = open_txt("./train.txt")

flag_test_ok = False
flag_valid_ok = False
for train_data in train:

    if train_data in test:
        print("Error!")
    else:
        flag_test_ok = True
    
    if train_data in valid:
        print("Error!")
    else:
        flag_valid_ok = True


valid_test_ok = True

for valid_data in valid:

    if valid_data in test:
        print("Error!")
    else:
        valid_test_ok = True

print(" \n ... Train size " + str(len(train)))
print(" \n ... Valid size " + str(len(valid)))
print(" \n ... Test size " + str(len(test)))

if flag_test_ok == True:
    print(" \n ... Test ok")

if flag_valid_ok == True:
    print(" \n ... Valid ok")

if valid_test_ok == True:
    print(" \n ... Valid != Test")




