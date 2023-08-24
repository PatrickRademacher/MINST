import pickle
import numpy as np
import math
import matplotlib.pyplot as pyplt
import matplotlib.cm as cm
import seaborn as sn
import pandas as pd


with open('all_of_dat_data.pkl', 'rb') as fp :
        gimme_dict = pickle.load(fp)

morethanaquiz_set  = gimme_dict['test_images']/255
choochoo_set  = gimme_dict['train_images']/255
labels_de_test = gimme_dict['test_labels']
labels_de_choochoo = gimme_dict['train_labels']
#initialize global variables that remain constant throughout code
epochs = 50
bias = 1.0
eta = 0.1
n1 = 20
n2 = 50
n3 = 100
nvalues = [n1, n2, n3]
momentum = 0.9
yay_colors = [(0.3, 0, 0.4), (0.3, 0.5, 0.9), (0.3, 0.7, 0.6), (0.6, 1.0, 0.6), (1, 0.5, 0.5), (0.8, 0.1, 1), 'red', 'green', 'cyan', 'black', 'blue', 'orange', (0.7, 0.5, 0.9), (0.9, 0.7, 0.6), (0.5, 1.0, 0.3), '#47a56c']




#sigmoid function
def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z

def get_accuracy(the_set_we_using, the_labels_we_using, wjis, wkjs, n):
    number_of_images = len(the_set_we_using)
    initial_accuracy = 0.0
    for i in range(number_of_images):
        xis = np.append(the_set_we_using[i].flatten(), bias) #flatten input nodes
        #np.append(xis, 1)
        this_is_right_based_on_label = the_labels_we_using[i] #get the correct value using label
        hjs = np.append((sigmoid(np.dot(wjis, xis))), bias) #add bias to hidden layer
        oks = sigmoid(np.dot(wkjs, hjs)) 
        i_guess_ill_guess = np.argmax(oks) #program takes highest value of outputs
        if i_guess_ill_guess == this_is_right_based_on_label: #compute accuracy
            initial_accuracy += 1
        
    final_accuracy = initial_accuracy/number_of_images
    return final_accuracy


#function to generate confusion matrix at the end
def get_confusion(wjis, wkjs, n):
    confusion_matrix = np.zeros([10, 10])
    woohoo = np.arange(0, 10000)
    np.random.shuffle(woohoo)
    for i in range(10000):
        proper_index = woohoo[i]
        xis = np.append(morethanaquiz_set[proper_index].flatten(), bias)
        this_is_right_based_on_label = int(labels_de_test[proper_index])
        hjs = np.append((sigmoid(np.dot(wjis, xis))), bias)
        oks = sigmoid(np.dot(wkjs, hjs))
        i_guess_ill_guess = int(np.argmax(oks))
        confusion_matrix[i_guess_ill_guess][this_is_right_based_on_label] += 1    
    confusion_matrix = confusion_matrix/10000
    return confusion_matrix      
        
#training function
def lets_train_exclamationpoint(choo_choo_set, n, wjis, wkjs, alpha):
    accuracy_training_array = []
    accuracy_testing_array = []
    numero_de_imagens = len(choo_choo_set)
    indices = np.arange(numero_de_imagens)
    cap_deltas_wkjs = np.zeros([10, n + 1])
    cap_deltas_wjis = np.zeros([n, 785])
    confusion_matrix = np.zeros([10, 10])

    for q in range(epochs):
        print("epoch = " + str(q))
        np.random.shuffle(indices)
        for i in range(numero_de_imagens):
            indx = indices[i]
            xis = np.append(choo_choo_set[indx].flatten(), bias)
            delta_ks = np.zeros(10)
            delta_js = np.zeros(n + 1)
            targets = np.zeros(10)
            lab = int(labels_de_choochoo[indx])
            targets = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
            targets[lab] = 0.9
            hjs = np.append((sigmoid(np.dot(wjis, xis))), bias) #append bias to hidden layer; take dot product of inputs with weights to hidden
            oks = sigmoid(np.dot(wkjs, hjs)) #sigmoid and dot of hidden layer and weights to outputs
            delta_ks = oks * (1 - oks) * (targets - oks)
            # for c in range(10):
            #     delta_k[c] = oks[c] * (1 - oks[c]) * (targets[c] - oks[c])
            delta_js = hjs*(1-hjs)*np.dot(np.transpose(wkjs), delta_ks) #transpose since back-prop
            cap_deltas_wkjs = (eta * np.outer(delta_ks, hjs)) + (alpha*cap_deltas_wkjs)
            wkjs+=cap_deltas_wkjs
            cap_deltas_wjis = (eta * np.outer(delta_js[:-1], xis)) + (alpha*cap_deltas_wjis)
            wjis += cap_deltas_wjis
        accuracy_training_array.append(get_accuracy(choo_choo_set, labels_de_choochoo, wjis, wkjs, n))
        accuracy_testing_array.append(get_accuracy(morethanaquiz_set, labels_de_test, wjis, wkjs, n))
    matrix_of_confusion = get_confusion(wjis, wkjs, n)
    return accuracy_training_array, accuracy_testing_array, matrix_of_confusion
                  

def experiment_report_numero_uno():
    
    #Experiment Report 1
    #initialize weight matrices for hidden layer
    wjis_uno = np.random.random((n1,785))/10-.05
    wjis_dos = np.random.random((n2, 785))/10-.05
    wjis_tres = np.random.random((n3,785))/10-.05
    #initialize weight matrices for output layer
    wkjs_uno = np.random.random((10, n1 + 1))/10-.05
    wkjs_dos = np.random.random((10, n2 + 1))/10-.05
    wkjs_tres = np.random.random((10, n3 + 1))/10-.05
    # run experiment 1
    print("doing 20 layers")
    train_results_uno, test_results_uno, confusion_uno = lets_train_exclamationpoint(choochoo_set, n1, wjis_uno, wkjs_uno, momentum)
    print("doing 50 layers")
    train_results_dos, test_results_dos, confusion_dos = lets_train_exclamationpoint(choochoo_set, n2, wjis_dos, wkjs_dos, momentum)
    print("doing 100 layers")    
    train_results_tres, test_results_tres, confusion_tres = lets_train_exclamationpoint(choochoo_set, n3, wjis_tres, wkjs_tres, momentum)
    all_train_results = [train_results_uno, train_results_dos, train_results_tres]
    all_test_results = [test_results_uno, test_results_dos, test_results_tres]    
    all_confusion = [confusion_uno, confusion_dos, confusion_tres]
    all_results = [all_train_results, all_test_results, all_confusion]
    make_confusion_matrix(all_confusion, 1, nvalues, [0.9, 0.9, 0.9])
    for i in range(3):
        plotting_zee_data(all_results, "Experiment 1: Results for " + str(nvalues[i]) + " Hidden Layers", yay_colors[i], 0)
    return all_results
    #Experiment Report 2

def experiment_report_numero_dos():
    the_only_wjis_needed = np.random.random((n3,785))/10-.05
    the_only_wkjs_needed = np.random.random((10,n3 + 1))/10-.05
    momentum_uno = 0.0
    momentum_dos = 0.25
    momentum_tres = 0.50
    train_results_uno, test_results_uno, confusion_uno = lets_train_exclamationpoint(choochoo_set, n3, the_only_wjis_needed, the_only_wkjs_needed, momentum_uno)
    train_results_dos, test_results_dos, confusion_dos = lets_train_exclamationpoint(choochoo_set, n3, the_only_wjis_needed, the_only_wkjs_needed, momentum_dos)
    train_results_tres, test_results_tres, confusion_tres = lets_train_exclamationpoint(choochoo_set, n3, the_only_wjis_needed, the_only_wkjs_needed, momentum_tres)
    all_train_results = [train_results_uno, train_results_dos, train_results_tres]
    all_test_results = [test_results_uno, test_results_dos, test_results_tres]
    all_confusion = [confusion_uno, confusion_dos, confusion_tres]
    all_results = [all_train_results, all_test_results, all_confusion]
    plotting_zee_data(all_train_results, "Experiment 2: Training Results for " + str(nvalues[2]) + " Hidden Layers.", yay_colors[3], 1)
    plotting_zee_data(all_test_results, "Experiment 2: Test Results for " + str(nvalues[2]) + " Hidden Layers.", yay_colors[3], 2)
    make_confusion_matrix(all_confusion, 2, [n3, n3, n3], [0.0, 0.25, 0.50])
    return all_results

def experiment_report_numero_tres():
    #Experiment Report #3
    quarter_choo = np.zeros([15000, 28, 28])
    half_choo = np.zeros([30000, 28, 28])
    for i in range(30000):
        if i < 15000:
            quarter_choo[i] = np.copy(choochoo_set[i])
        half_choo[i] = np.copy(choochoo_set[i])
    the_only_wjis_needed = np.random.random((n3,785))/10-.05
    the_only_wkjs_needed = np.random.random((10,n3 + 1))/10-.05
    train_results_uno, test_results_uno, confusion_uno = lets_train_exclamationpoint(quarter_choo, n3, the_only_wjis_needed, the_only_wkjs_needed, momentum)
    train_results_dos, test_results_dos, confusion_dos = lets_train_exclamationpoint(half_choo, n3, the_only_wjis_needed, the_only_wkjs_needed, momentum)
    all_train_results = [train_results_uno, train_results_dos]
    all_test_results = [test_results_uno, test_results_dos]
    all_confusion = [confusion_uno, confusion_dos, confusion_tres]
    all_results = [all_train_results, all_test_results, all_confusion]
    plotting_zee_data(all_train_results, "Experiment 3: Training Results for " + str(nvalues[2]) + " Hidden Layers", yay_colors[12], 3)
    plotting_zee_data(all_test_results, "Experiment 3: Test Results for " + str(nvalues[2]) + " Hidden Layers", yay_colors[14], 3)
    return all_results

def plotting_zee_data(resultz, title_for_graph, graph_color, num):
    
    
    if num == 0:
        t = np.linspace(0.0, 2.0, 201)
        fig = pyplt.figure()
        txt = ['Training ', 'Test']
        pyplt.xlabel('Epoch')
        pyplt.ylabel('Accuracy')
        pyplt.title(title_for_graph)
        pyplt.scatter(np.arange(epochs), resultz[0][yay_colors.index(graph_color)], color=graph_color)
        pyplt.scatter(np.arange(epochs), resultz[1][yay_colors.index(graph_color)], color= yay_colors[yay_colors.index(graph_color) + 3])
        fig.text(.2, .05, txt[0], ha='left', color=graph_color)
        fig.text(.2, .03, txt[1], ha='left', color=yay_colors[yay_colors.index(graph_color) + 3])
        pyplt.show() 
        
    elif num == 1:
        fig, ax = pyplt.subplots(figsize=(9.2, 5))
        t = np.linspace(0.0, 2.0, 201)
        fig = pyplt.figure()
        txt = ['Momentum = 0 ', 'Momentum = 0.25', 'Momentum = 0.50']
        pyplt.xlabel('Epoch')
        pyplt.ylabel('Accuracy')
        pyplt.title(title_for_graph)
        pyplt.scatter(np.arange(epochs), resultz[0], color=yay_colors[6])
        pyplt.scatter(np.arange(epochs), resultz[1], color=yay_colors[7])
        pyplt.scatter(np.arange(epochs), resultz[2], color=yay_colors[8])
        cool_colors = [yay_colors[6], yay_colors[7], yay_colors[8]]
        fig.text(.2, .05, txt[0], ha='left', color=cool_colors[0])
        fig.text(.2, .03, txt[1], ha='left', color=cool_colors[1])
        fig.text(.2, .01, txt[2], ha='left', color=cool_colors[2])
        pyplt.show()
        
    elif num == 2:
        t = np.linspace(0.0, 2.0, 201)
        txt = ['Momentum = 0 ', 'Momentum = 0.25', 'Momentum = 0.50']
        pyplt.xlabel('Epoch')
        pyplt.ylabel('Accuracy')
        pyplt.title(title_for_graph)
        pyplt.scatter(np.arange(epochs), resultz[0], color=yay_colors[9])
        pyplt.scatter(np.arange(epochs), resultz[1], color=yay_colors[10])
        pyplt.scatter(np.arange(epochs), resultz[2], color=yay_colors[11])
        cool_colors = [yay_colors[9], yay_colors[10], yay_colors[11]]
        fig.text(.2, .05, txt[0], ha='left', color=cool_colors[0])
        fig.text(.2, .03, txt[1], ha='left', color=cool_colors[1])
        fig.text(.2, .01, txt[2], ha='left', color=cool_colors[2])
        pyplt.show()
        
    else:
        fig = pyplt.figure()
        txt = ['', '', '', 'Using a quarter of the training samples', 'Using half of the training samples']
        t = np.linspace(0.0, 2.0, 201)
        pyplt.xlabel('Epoch')
        pyplt.ylabel('Accuracy')
        pyplt.title(title_for_graph)
        pyplt.scatter(np.arange(epochs), resultz[0], color=graph_color)
        fig.text(.2, .05, txt[3], ha='left', color = graph_color)
        where_am_i = yay_colors.index(graph_color) + 1
        fig.text(.2, .03, txt[4], ha='left', color=yay_colors[where_am_i])
        pyplt.scatter(np.arange(epochs), resultz[1], color=yay_colors[where_am_i])


        pyplt.show() 
        
def make_confusion_matrix(data, experiment_num, num_hidden, mom):
    num_matrices = len(data)
    matrix_to_plot = np.zeros([3, 10, 10])
    for m in range(3):
        matrix_to_plot[m] = data[m]
        df_cm = pd.DataFrame(matrix_to_plot[m], index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
        pyplt.figure(figsize = (10,7))
        fig, ax = pyplt.subplots(figsize=(9.2, 5))
        pyplt.title(('Confusion Matrix for Experiment' + str(experiment_num) + ', with ' + str(num_hidden[m]) + ' Hidden Layers and Momentum = ' + str(mom[m])))
        pyplt.xlabel('Actual')
        pyplt.ylabel('Guess/Prediction')
        sn.heatmap(df_cm, annot=True)
        pyplt.show()
        


    


if __name__ == "__main__":
    
    experiment_uno_results = experiment_report_numero_uno()
    #experiment_dos_results = experiment_report_numero_dos()
    #experiment_tres_results = experiment_report_numero_tres()
    f = open("outputttty_for_exp1.txt","w+")
    
    for q in range(len(experiment_uno_results)):
        f.write("This is part " + str(q) + " of experiment 1\n\n" + str(experiment_uno_results[q]) + "\n\n\n")
    f.close()
    

    
