from copy import deepcopy
import numpy as np
from scipy.stats import truncnorm
import pickle
if 'google.colab' in str(get_ipython()):
    from MgAlloyPublic.model_paths import models
else:
    from model_paths import models



class MgDatapoint:
    def __init__(self, settings):
        self.categorical_inputs = settings.categorical_inputs
        self.categorical_inputs_info = settings.categorical_inputs_info
        self.range_based_inputs = settings.range_based_inputs
        self.mg_balance = True

    def formatForInput(self):
        ht = [1 if [i+1] in [*self.categorical_inputs.values()] else 0 for i in range(6)]
        
        
        if sum([*self.range_based_inputs.values()]) != 100:
            self.mg_balance = False
            

        my_input = [100 - sum([*self.range_based_inputs.values()][1:])] + [*self.range_based_inputs.values()][1:] + ht  
                   # [*self.range_based_inputs.values()] 
        
        
        return np.reshape(my_input, (1, -1))

    def print(self):
        for key, value in self.categorical_inputs.items():
            print(f"{key}: {self.categorical_inputs_info[key]['tag'][self.categorical_inputs_info[key]['span'].index(value)]}")
        #print(f"Mg%: {round(self.getMg(), 2)}")
        for key, value in self.range_based_inputs.items():
            if value:
                print(f"{key}: {value}")

    #def getMg(self):
        #return 100 - sum(sum(row) for row in list(self.range_based_inputs.values())[1:])
                


class scanSettings:
    def __init__(self, mode):
        self.mode = mode

        if self.mode == 'DoS':
            self.loss_type = 'Linear'
            self.max_steps = 1
            self.targets = {
                'DoS': 10
            }
            self.categorical_inputs = {
                'Heat Treatment': [1]
            }
            self.categorical_inputs_info = {
                'Heat Treatment': {'span': [1, 2, 3, 4, 5, 6], 'tag': ['Extruded', 'ECAP',
                                                                       'Cast_Slow', 'Cast_Fast', 'Cast_HT', 'Wrought']}}
            
            self.range_based_inputs = dict.fromkeys(
                ['Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag', 'Ho', 'Mn',
       'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga', 'Be', 'Fe',
       'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'], [0])


        if self.mode == 'Mechanical':
            self.loss_type = 'Percentage'
            self.max_steps = 1
            self.targets = {
                'elongation%': 6,
                'tensile strength(MPa)': 250
            }
            self.categorical_inputs = {
                'Heat Treatment': [1]
            }
            self.categorical_inputs_info = {
                'Heat Treatment': {'span': [1, 2, 3, 4, 5, 6], 'tag': ['Extruded', 'ECAP', 'Cast_Slow', 'Cast_Fast', 'Cast_HT', 'Wrought']}}
            
#             self.range_based_inputs = dict(zip(
#                 ['Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag', 'Ho', 'Mn',
#                  'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga', 'Be', 'Fe',
#                  'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'],
#                 [[0.827], [0.0026], [0], [0], [0], [0], [0.065], [0.0945], [0],
#                  [0], [0], [0], [0], [0], [0], [0.0076],
#                  [0], [0], [0], [0], [0], [0], [0], 
#                  [0], [0], [0], [0], [0], [0.0032], [0], [0]]))
            
            self.range_based_inputs = dict(zip(
                ['Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag', 'Ho', 'Mn',
                 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga', 'Be', 'Fe',
                 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'],
                [[100], [0] , [0] , [0] , [0] , [0] ,
                    [0] , [0] , [0] , [0] , [0],
                    [0] , [0] , [0] , [0] , [0], 
                    [0] , [0] , [0] , [0] , [0], 
                    [0] , [0] , [0] , [0] , [0], 
                    [0] , [0] , [0] , [0] , [0]]))
        
            self.range_based_inputs['Mg'] = [100 - sum(sum(row) for row in list(self.range_based_inputs.values())[1:-1])]
            
        
        
            


class optimiser:
    def __init__(self, settings):
        self.step_batch_size = 100
        self.step_final_std = 0.01
        self.finetune_max_rounds = 3
        self.finetune_batch_size = 10
        self.mode = settings.mode
        self.loss_type = settings.loss_type
        self.targets = settings.targets
        self.max_steps = settings.max_steps
        self.categorical_inputs = settings.categorical_inputs
        self.range_based_inputs = settings.range_based_inputs
        self.settings = settings
        self.models = models

        self.run()

    def calculateLoss(self, datapoint):
        if self.mode == 'DoS':
            return self.models['elongation'].predict(datapoint.formatForInput())[0]
        elif self.mode == 'Mechanical':
            return self.models['elongation'].predict(datapoint.formatForInput())[0]

    def printResults(self, best_datapoint):
        if self.mode == 'DoS':
            print('data point:',best_datapoint.formatForInput()) 
            #print('predicted %f Elongation' % (1.25*self.models['elongation'].predict(best_datapoint.formatForInput())[0]))
            print('predicted %f Yield Strength' % (1.25*self.models['yield'].predict(best_datapoint.formatForInput())[0]))
            #print('predicted %f Tensile Strength' % (1.25*self.models['tensile'].predict(best_datapoint.formatForInput())[0]))
        elif self.mode == 'Mechanical':
            final_alloy  = dict(zip(
                ['Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag', 'Ho', 'Mn',
                 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga', 'Be', 'Fe',
                 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi', 'Extruded', 'ECAP', 'Cast_Slow', 'Cast_Fast', 'Cast_HT', 'Wrought'],
                best_datapoint.formatForInput().reshape(-1,)))
            
            if not best_datapoint.mg_balance:
                print('\033[1m'+'\033[91m'+ "Mg content has been balanced to "+ str(final_alloy['Mg']) + " %" +'\033[0m')
            
            print('Chemical composition: ')
            for index, key in enumerate(final_alloy):
                print(key+ ":" + str(final_alloy[key]), end="  ")
                if (index+1)%10 ==0:
                    print("")
                  
                
            print('\nPredicted %f Elongation' % (self.models['elongation'].predict(best_datapoint.formatForInput())[0]))
            print('Predicted %f Yield Strength' % (self.models['yield'].predict(best_datapoint.formatForInput())[0]))
            print('Predicted %f Tensile Strength' % (self.models['tensile'].predict(best_datapoint.formatForInput())[0]))

    def run(self):
        best_loss = None
        best_datapoint = MgDatapoint(self.settings)
        for i in range(self.max_steps):
            loss, datapoint = self.calculateStep(best_datapoint, i, 'all')
            if best_loss is None or loss < best_loss:
                best_datapoint = datapoint
                best_loss = loss

        for i in range(self.finetune_max_rounds):
            for key in [*self.categorical_inputs.keys(), *self.range_based_inputs.keys()]:
                loss, datapoint = self.calculateStep(best_datapoint, i, key)
                if loss < best_loss:
                    best_datapoint = datapoint
                    best_loss = loss
            else:
                break
        print('==========Scan Finished==========')
        self.printResults(best_datapoint)

    def calculateStep(self, best_datapoint, step_number, target_var):
        if target_var == 'all':
            batch_size = self.step_batch_size
        else:
            batch_size = self.finetune_batch_size
        loss = [0] * batch_size
        datapoints = []
        std = self.step_final_std * (self.max_steps / float(step_number + 1))
        for i in range(batch_size):
            datapoints.append(deepcopy(best_datapoint))
            for key in self.categorical_inputs.keys():
                if target_var == key or target_var == 'all':
                    datapoints[i].categorical_inputs[key] = np.random.choice(self.categorical_inputs[key])
            for key in self.range_based_inputs.keys():
                if target_var == key or target_var == 'all':
                    if max(self.range_based_inputs[key]) != min(self.range_based_inputs[key]):
                        a = (min(self.range_based_inputs[key]) - np.mean(best_datapoint.range_based_inputs[key])) / std
                        b = (max(self.range_based_inputs[key]) - np.mean(best_datapoint.range_based_inputs[key])) / std
                        datapoints[i].range_based_inputs[key] = round(
                            float(truncnorm.rvs(a, b, loc=np.mean(best_datapoint.range_based_inputs[key]), scale=std)),
                            2)
                    else:
                        datapoints[i].range_based_inputs[key] = min(self.range_based_inputs[key])
            loss[i] = self.calculateLoss(datapoints[i])
        return min(loss), datapoints[loss.index(min(loss))]
