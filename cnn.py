"""
Created on Mon Jun 24 14:27:49 2019

@author: Pedro Salamoni
"""

#Parameters
data_path = '/home/vinicius/Documents/MEUS_CODIGOS/example.csv'
labels = ['avePet', 'bandeja', 'biofarm', 'bolaCravo', 'brinqBolaCGuizo', 'brinqBolaCorda', 'brinqCoxinha', 'brinqHamburguer', 'brinqPeluciaBanana', 'brinqPeluciaBerinjela', 'brinqPeluciaCCorda', 'brinqPeluciaChinelo', 'brinqPeluciaDinossauro', 'brinqPeluciaDogFeliz', 'brinqPeluciaDoguinho', 'brinqPeluciaDonaGirafa', 'brinqPeluciaGirafa', 'brinqPeluciaGuepardo', 'brinqPeluciaLeonino', 'brinqPeluciaLion', 'brinqPeluciaPolvo', 'brinqPeluciaTigreBranco', 'brinqPeluciaVaquinha', 'caixaTransporteFalcon', 'caixaTransportePreta', 'catNip', 'colarElisabetano', 'coleiraChickCat', 'comedPlast', 'condicionadorSavana', 'cordaTrançadaCBola', 'cordosso', 'cortador', 'cremeDentalMenta', 'cremeDentalMorango', 'cremeDentalNeutro', 'cremeDentalTuttiFrutti', 'dedeira', 'defumadoBovino', 'dinoBallTrex', 'duraball', 'durabone', 'durafish', 'escovaDuplaBlack', 'femurSuino', 'focinheiraPlastico', 'galinhaSonora', 'guiaRetratilCoruja', 'halterCravinho', 'luvaMagica', 'mamadeira', 'nutriconAcidificante', 'nutriconAlcalinizante', 'nutriconAntiAlgas', 'NutriconAntiCloro', 'nutriconBottom', 'nutriconCondicionadorDeAgua', 'nutriconNectarBeijaflor', 'nutriconNutriballs', 'nutriconNutriBird', 'nutriconNutriroedores', 'nutriconPremiumFlakes', 'nutriconStickFood', 'nutriconTestePh', 'ossoCanelinhaDefumada', 'ossoCoxa', 'ossoNo', 'ossoPalito', 'ossoPalitoCarne', 'ossoPalitoFrango', 'ossoPalitoLeite', 'ossoPalitoMenta', 'ossoPalitoMiudos', 'ossoRodelinhaRecheada', 'pa', 'penteAntiPulgas', 'pipiPode', 'porcoEspinho', 'porcoSonoro', 'portaCaixa', 'portaLixoOsso', 'portaLixoRefil', 'poteRatinho', 'rasqueadeiraAutoLimpante', 'rasqueadeiraBamboo', 'rasqueadeiraConfort', 'rasqueadeiraLuxo', 'rasqueadeiraMadeira', 'rasqueadeiraPlastico', 'rasqueadeiraPremium', 'rasqueadeiraQuality', 'rasqueadeiraWhite', 'resfriadorDeLaminas', 'serragem', 'shampooClareador', 'shampooNeutralizadorOdores', 'shampooNeutro', 'talcoBanho', 'tapete', 'varinhaMouse']
test_rate = 0.5
n_training_data = None #Number of data from training set to be used in order to train ai | To use all data, insert None
n_testing_data = 1000
epochs = 10


################################################################################################################################

def freememo():
    import gc
    
    gc.collect()

def savemodel(model,evaluation):
    import datetime

    model_name = str('teste1')    
    #model_name = str(datetime.datetime.now()).replace('-','').replace(':','').replace(' ','_')
    evaluation = str(evaluation[1])
    model_name = 'Models/model_' + model_name[:model_name.rfind('.')] + '_' + evaluation[evaluation.rfind('.')+1:]
    
    model_yaml = model.to_yaml()
    with open(model_name + "_structure.yaml","w") as yaml_file:
        yaml_file.write(model_yaml)
        
    model.save_weights(model_name + "_weight.h5")
    
    model.save(model_name + "_all.h5")
    print("\n Model Saved \n")
    return (model_name, model_name + "_weight.h5")
    
def importdata(data_path):
    import pandas as pd
    
    my_data = pd.read_csv(data_path, sep=',',header=None)
    path, res = my_data[0], my_data[1]
    print("\n Metadata Imported \n")
    
    return path,res

def preprocessdata(path,res,test_rate):
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    path_train, path_test, res_train, res_test = train_test_split(path, res, test_size = test_rate)
    res_train = np.array(res_train)
    res_test = np.array(res_test)
    path_train = np.array(path_train)
    path_test = np.array(path_test)
    
    print("\n Metadata Processed \n")
    
    return path_train,path_test,res_train,res_test

def getdata(paths,n_data,res):
    from PIL import Image
    import numpy as np
    
    if n_data!=None:
        if len(res)==0:
            if n_data>len(paths):
                n_data = len(paths)
        else:
            if n_data>min(len(paths),len(res)):
                n_data = min(len(paths),len(res))

    im = Image.open(paths[0])
    x = np.array(im)[None,:,:,:]

    for i,path in enumerate(paths[1:n_data]):
        print(i,path)
        im = Image.open(path)
        new = np.array(im)[None,:,:,:]
        x = np.append(x,new, axis=0)
        
    x = x / 255.0
    if len(res)==0:
        print("\n Data Imported \n")
        return x
    else:
        y = res[:n_data]
        print("\n Data Imported \n")
        return x,y

def createai(output_size):
    from tensorflow import keras
    
    model = keras.Sequential()
    model.add( keras.layers.Conv2D(64,3) )
    model.add( keras.layers.MaxPooling2D() )
    model.add( keras.layers.Conv2D(64,3) )
    model.add( keras.layers.MaxPooling2D() )
    model.add( keras.layers.Flatten() )
    model.add( keras.layers.Dense( 128, activation=keras.activations.relu ))
    model.add( keras.layers.Dense( output_size , activation=keras.activations.softmax ))
    
    model.compile( optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    
    return model
    
def trainai(model,x_train,y_train,epochs):
    
    model.fit(x = x_train, y = y_train, epochs = epochs)

def evaluateai(model,x,y):

    evaluation = model.evaluate(x,y)            

    return evaluation
    
def main(path_train, res_train, path_test, res_test):
    from tensorflow import keras
    from tensorflow.keras.models import model_from_yaml

  
    x_train,y_train = getdata(path_train,n_training_data,res_train)
    trainai(model,x_train,y_train,epochs)
    del x_train,y_train
    freememo()

    x_test,y_test = getdata(path_test,n_testing_data,res_test)
    evaluation = evaluateai(model,x_test,y_test)
    del x_test,y_test
    freememo()
    loaded_model, loaded_weights = savemodel(model,evaluation)

    """for i in range (20):
        # load YAML and create model
        yaml_file = open('/home/vinicius/Documents/MEUS_CODIGOS/Models/model_20190813_224951_6112977_structure.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        loaded_model.load_weights("/home/vinicius/Documents/MEUS_CODIGOS/Models/model_20190813_224951_6112977_weight.h5")
        print("Loaded model from disk")
        
        loaded_model = createai(len(labels))
        
        x_train,y_train = getdata(path_train,n_training_data,res_train)
        
        trainai(loaded_model,x_train,y_train,epochs)
        del x_train,y_train
        freememo()
        
        x_test,y_test = getdata(path_test,n_testing_data,res_test)
        
        evaluation = evaluateai(loaded_model,x_test,y_test)
        del x_test,y_test
        freememo()
        
        loaded_model, loaded_weights = savemodel(loaded_model,evaluation)"""



paths,res = importdata(data_path)    
path_train,path_test,res_train,res_test = preprocessdata(paths,res,test_rate)
model = createai(len(labels))
for i in range (32):
    path_train_part = path_train[i*1000:i*1000+1000]
    res_train_part = res_train[i*1000:i*1000+1000]
    main(path_train_part, res_train_part, path_test, res_test)    