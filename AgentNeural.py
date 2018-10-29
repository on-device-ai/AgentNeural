#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:16:25 2018

@author: yilintung
"""

import sys
import numpy as np
import pandas as pd 

from pathlib import Path

from autokeras.image_supervised import ImageClassifier
from autokeras.image_supervised import load_image_dataset

from keras.layers import Input, Dense

from keras.layers import Activation
from keras.models import Model
from keras.models import load_model
from keras.utils import to_categorical

def true_false_list_generator(step=0.0001,true_value_start=1.0,true_value_threshold=0.9,false_value_start=0.0,false_value_threshold=0.1) :

    true_value_list = list()
    false_value_list = list()

    true_value = true_value_start
    while true_value >= true_value_threshold :
        true_value_list.append(true_value)
        true_value -= step

    false_value = false_value_start
    while false_value <= false_value_threshold :
        false_value_list.append(false_value)
        false_value += step
        
    return true_value_list,false_value_list

if __name__ == '__main__':
    
    agent_filename = 'traffic_light.agent'
    
    agent_percept = 'image'
    agent_func_interpert_input = 'autokeras'
    agent_skip_autokeras = False
    train_csv_file_path = None
    train_images_path = None
    test_csv_file_path = None
    test_images_path = None
    autokeras_timeout = 12 * 60 * 60
    output_keras_mode = 'my_model.h5'
    keras_num_classes = 0
    keras_output_activation = 'softmax'
    state_list = list()
    action_list = list()
    rule_list = list()
    
    true_list,false_list = true_false_list_generator()
    
    with open(agent_filename) as fp:
        line = fp.readline()
        while line:
            tokens = line.split('=')
            command = tokens[0].replace(' ', '').replace('\t', '').replace('\n', '')
            if command == 'percept' :
                parameter = tokens[1].replace(' ', '').replace('\t', '').replace('\n', '')
                agent_percept = parameter
            elif command == 'interpert_input' :
                parameter = tokens[1].replace(' ', '').replace('\t', '').replace('\n', '')
                agent_func_interpert_input = parameter
            elif command == 'autokeras_train_data':
                parameter = tokens[1].replace('\n', '')
                parameter_tokens = parameter.split(',')
                parameter_sub1 = parameter_tokens[0].replace(' ', '').replace('\t', '')
                parameter_sub2 = parameter_tokens[1].replace(' ', '').replace('\t', '')
                train_csv_file_path = parameter_sub1
                train_images_path = parameter_sub2
            elif command == 'autokeras_test_data':
                parameter = tokens[1].replace('\n', '')
                parameter_tokens = parameter.split(',')
                parameter_sub1 = parameter_tokens[0].replace(' ', '').replace('\t', '')
                parameter_sub2 = parameter_tokens[1].replace(' ', '').replace('\t', '')
                test_csv_file_path = parameter_sub1
                test_images_path = parameter_sub2
            elif command == 'autokeras_timeout':
                parameter = tokens[1].replace(' ', '').replace('\t', '').replace('\n', '')
                autokeras_timeout = int(parameter)
            elif command == 'autokeras_output_keras_model':
                parameter = tokens[1].replace(' ', '').replace('\t', '').replace('\n', '')
                output_keras_mode = parameter
            elif command == 'keras_model_num_classes':
                parameter = tokens[1].replace(' ', '').replace('\t', '').replace('\n', '')
                keras_num_classes = int(parameter)
            elif command == 'keras_model_output_activation':
                parameter = tokens[1].replace(' ', '').replace('\t', '').replace('\n', '')
                keras_output_activation = parameter
            elif command == 'state' :
                parameter = tokens[1].replace('\n', '')
                parameter_tokens = parameter.split(',')
                for token in parameter_tokens:
                    state_list.append(token.replace(' ', '').replace('\t', ''))
            elif command == 'action':
                parameter = tokens[1].replace('\n', '')
                parameter_tokens = parameter.split(',')
                for token in parameter_tokens:
                    action_list.append(token.replace(' ', '').replace('\t', ''))
                if len(action_list) > 1 :
                    sys.exit("Current only support one action!!")
            elif command == 'rule':
                parameter = tokens[1].replace('\n', '')
                parameter_tokens = parameter.split('->')
                state_tokens  = parameter_tokens[0]
                action_tokens = parameter_tokens[1]
                
                state_in_list = list()
                
                states_token = state_tokens.split('&')
                for state_token in states_token :
                    state_count = 0
                    state_index = -1
                    state_value_list = state_token.split(' ')
                    state_value_list = [x for x in state_value_list if x]
                    for state in state_list :
                        if state_value_list[0].replace('\t', '') == state :
                            state_index = state_count
                            break;
                        state_count += 1
                    if state_index < 0 :
                        continue
                    if state_value_list[2].replace('\t', '') == 'true' :
                        state_in_list.append([state_index,true_list])
                    elif state_value_list[2].replace('\t', '') == 'false' :
                        state_in_list.append([state_index,false_list])
                        
                action_value_list = action_tokens.split(' ')
                action_value_list = [x for x in action_value_list if x]
                if action_value_list[0].replace('\t', '') == action_list[0] :
                    rule_value_list = list()
                    rule_state_list = list()
                    
                    state_count = 0
                    while state_count < len(state_in_list) :
                        for in_list in state_in_list :
                            if in_list[0] == state_count :
                                rule_state_list.append(in_list[1])
                        state_count += 1
                    rule_value_list.append(rule_state_list)
                        
                    if action_value_list[2].replace('\t', '') == 'true' :
                        rule_value_list.append(1.0)
                    elif action_value_list[2].replace('\t', '') == 'false' :
                        rule_value_list.append(0.0)
                    else :
                        sys.exit("Invalid rule!!")
                        
                rule_list.append(rule_value_list)            
                    
            line = fp.readline()

    my_file = Path(output_keras_mode)
    if agent_skip_autokeras is True and my_file.is_file() is True :
        print('Skip running AutoKeras ...\n')
    else : 
        if test_csv_file_path is not None and test_images_path is not None :
            x_train, y_train = load_image_dataset(csv_file_path=train_csv_file_path,images_path=train_images_path)
        else :
            sys.exit("No train data input!!")
        
        if test_csv_file_path is not None and test_images_path is not None : 
            x_test, y_test = load_image_dataset(csv_file_path=test_csv_file_path,images_path=test_images_path)
            do_evaluate = True
        else :
            do_evaluate = False

        if agent_percept == 'image' and agent_func_interpert_input == 'autokeras' :
            clf = ImageClassifier(verbose=True)
            clf.fit(x_train, y_train, time_limit=autokeras_timeout)
            clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
        else :
            sys.exit("No valid percept type or INTERPERT-INPUT function!!")
    
        if do_evaluate is True : 
            y = clf.evaluate(x_test, y_test)
            print('test data evaluate = ' + str(y))

        clf.load_searcher().load_best_model().produce_keras_model().save('my_model.h5')
        
    keras_model = load_model('my_model.h5')

    if keras_output_activation == 'softmax' :    
        x = keras_model.output
        x = Activation('softmax', name='activation_add')(x)
        keras_model = Model(inputs=keras_model.input, outputs=x)
        keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    elif keras_output_activation == 'sigmoid' :
        x = keras_model.output
        x = Activation('sigmoid', name='activation_add')(x)
        keras_model = Model(inputs=keras_model.input, outputs=x)
        keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    keras_model.summary()
     
    if test_csv_file_path is not None and test_images_path is not None :
        x_train, y_train = load_image_dataset(csv_file_path=train_csv_file_path,images_path=train_images_path)
        y_train = to_categorical(y_train.astype(int), keras_num_classes)
    else :
        sys.exit("No train data input!!")
        
    if test_csv_file_path is not None and test_images_path is not None : 
        x_test, y_test = load_image_dataset(csv_file_path=test_csv_file_path,images_path=test_images_path)
        y_test_label = y_test.astype(int)
        y_test = to_categorical(y_test.astype(int), keras_num_classes)
        do_evaluate = True
    else :
        do_evaluate = False

    keras_model.fit(x_train, y_train,batch_size=32,epochs=10,verbose=1,validation_split=0.2)
    
    if do_evaluate is True :
        score = keras_model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        prediction = keras_model.predict(x_test) #print(prediction)
        prediction = prediction.argmax(axis=-1)
        
        print("[Info] Display Confusion Matrix:")  
        print("%s\n" % pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict'])) 
    
    for layer in keras_model.layers:
        layer.trainable = False
    keras_mode_total_layer = len(keras_model.layers)
    
    data_list  = list()
    label_list = list()
    
    for rule in rule_list :
        rule_count = 0
        para1_count = 0
        para2_count = 0
        
        while rule_count < len(rule[0][0]):
            if para2_count >= len(rule[0][1]) :
                para2_count = 0
    
            data_list.append([rule[0][0][para1_count],rule[0][1][para2_count]])
            label_list.append(rule[1]);
    
            rule_count += 1
            para1_count += 1
            para2_count +=1
    
    data = np.array(data_list)
    labels = np.array(label_list)

    input_layer_node = len(state_list)
    output_layer_node = len(action_list) 
    hidden_layer_node = 2 * (round((input_layer_node + output_layer_node) ** 0.5))
    
    inputs = Input(shape=(input_layer_node,))

    x = Dense(hidden_layer_node, activation='relu')(inputs)
    output = Dense(1, activation='sigmoid')(x)  

    rule_model = Model(inputs=inputs, outputs=output)
    rule_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    rule_model.summary()

    rule_model.fit(data, labels, batch_size = 32, epochs=50)

    layer_count = 0;
    for layer in rule_model.layers :
        if layer_count == 1 :
            rule_model_hidden_layer_weights = layer.get_weights()
            
        elif layer_count == 2 :
            rule_model_output_layer_weights = layer.get_weights()
        
        layer_count += 1
    
    x = keras_model.output
    x = Dense(hidden_layer_node, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)  
    keras_model = Model(inputs=keras_model.input, outputs=output)
    keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    keras_model.summary()
    
    layer_count = 0;
    for layer in keras_model.layers :
        if layer_count == keras_mode_total_layer:
            layer.set_weights(rule_model_hidden_layer_weights)
        elif layer_count == keras_mode_total_layer+1 :
            layer.set_weights(rule_model_output_layer_weights)
        
        layer_count +=1

    if do_evaluate is True :        
        prediction = keras_model.predict(x_test)
        print(prediction)
        actions = ['Go' if x >= 0.5 else 'Stop' for x in prediction]
        print(actions)