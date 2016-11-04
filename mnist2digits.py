from __future__ import division

# ---------------- Loading the MNIST dataset ----------------------------------

import gzip
import cPickle

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
valid_x, valid_y =valid_set
test_x, test_y = test_set
batch_size=50
cond_salida=6
muestreo=50 #cada cuantos batches muestreo tasa_error

import nn4t

def runnetwork(net,data,labels):
    res = []
    coste=0
    error=0
    for x, y in zip(data, labels):
        salida = net.output(x)
        salida = nn4t.softmax(salida);
        # print max(nn4t.softmax(x))
        res.append([salida.tolist().index(max(salida)), y])
        if y <> salida.tolist().index(max(salida)):
            error += 1
        y_coded = nn4t.one_hot(y.astype(int), 10);
        coste=coste + (np.dot(salida-y_coded,(salida-y_coded).reshape(len(salida),1)))[0][0]/2
    tasa_error = error / len(res)
    coste=coste/len(data)

    return tasa_error,coste
# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

# plt.imshow(train_x[57].reshape((28, 28)), cmap = cm.Greys_r)
# plt.show() # Let's see a sample


# ---------------- Creating the training set of only two digits, 1 and 8 ------

# samples = [] # Our training set, initially empty
# labels = [] # Our labels set, initially empty
# j = 0
# for label in train_y:
#     if label == 1:
#         samples.append(train_x[j])
#         labels.append([1]) # Label '1' for the set of ones
#     if label == 8:
#         samples.append(train_x[j])
#         labels.append([-1]) # Label '-1' for the set of eights
#     j += 1


# ---------------- Training the neural network --------------------------------

import neurolab as nl
import numpy as np

# intervals = [[0.0,1.0] for i in range(784)] #intervals of input values
# net = nl.net.newff(intervals,[4, 1],[nl.trans.TanSig(), nl.trans.TanSig()])
# error = net.train( samples , labels , epochs=10, show=1, goal=0.01)
train_y_oh = nn4t.one_hot(train_y[:].astype(int), 10);
configuraciones=[5,10,20,50]
epochs=20
punto_salida=[]

for num_hidden_layers in configuraciones:

    net = nn4t.Net(layers=[784, num_hidden_layers, 10]);

    train=[]
    test=[]
    valid=[]
    te_min=1
    cs=0
    muestreo = 50

    for i in range(epochs):
        print "Epoch: ", i
        if i==2: muestreo=500
        for j in range(int(len(train_x)/batch_size)):
            net.train(train_x[j*batch_size:(j+1)*batch_size], train_y_oh[j*batch_size:(j+1)*batch_size], lr=0.1)

            if ((j+1)%muestreo)==0:

                res_train = ['e'] + [num_hidden_layers] + [i+((j+1)*batch_size/len(train_x))] \
                            + [row for row in runnetwork(net, train_x[:10000], train_y[:10000])]
                train.append(res_train)
                res_validacion = ['v'] + [num_hidden_layers] + [i+(j+1)*batch_size/len(train_x)] \
                                 + [row for row in runnetwork(net, valid_x, valid_y)]
                valid.append(res_validacion)

                res_test = ['t'] + [num_hidden_layers] + [i+(j+1)*batch_size/len(train_x)] \
                           + [row for row in runnetwork(net, test_x, test_y)]
                test.append(res_test)


                if te_min>res_validacion[3]:
                    cs=0
                    te_min=res_validacion[3]
                    #Guardar pesos
                if cs==cond_salida:
                    print cs
                    punto_salida.append(res_validacion[3])
                cs += 1

        if (len(train_x)%batch_size)>0:
            net.train(train_x[(j+1) * batch_size:], train_y_oh[(j+1) * batch_size:], lr=0.1)

            res_train=['e']+ [num_hidden_layers]+[i+1]+[row for row in runnetwork(net,train_x, train_y)]
            train.append(res_train)
            res_validacion=['v']+ [num_hidden_layers]+[i+1]+[row for row in runnetwork(net, valid_x, valid_y)]
            valid.append(res_validacion)

            res_test = ['t'] + [num_hidden_layers] + [i + 1] + [row for row in runnetwork(net, test_x, test_y)]
            test.append(res_test)
            # tasa_error, coste=runnetwork(test_x, test_y)
            # res_test=['t']+ [i]+[tasa_error]+[coste]
            # test.append(res_test)



    #print punto_salida
    #print train
    plt.subplot(len(configuraciones),1,configuraciones.index(num_hidden_layers)+1)
    plt.plot([row[2] for row in train],[row[3] for row in train],'r.-',label='train set');
    plt.plot([row[2] for row in valid],[row[3] for row in valid],'b^-',label='validation set');
    plt.plot([row[2] for row in test],[row[3] for row in test],'go-',label='test set');
    plt.axis([0, epochs, 0, 1])
    plt.grid(True)
    plt.legend(loc='upper right', shadow=True,fontsize='xx-small')
    plt.title('Accuracy(%) con '+str(num_hidden_layers)+ ' neuronas')
    plt.xlabel('epochs')
    plt.ylabel('Acc(%)')
    #print [row[3] for row in valid]


plt.show()

