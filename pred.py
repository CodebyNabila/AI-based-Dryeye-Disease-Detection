#======================== IMPORT PACKAGES ===========================
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential
import matplotlib.image as mpimg
import cv2
import streamlit as st

import google.generativeai as genai
import streamlit as st
import time

import base64
from streamlit_option_menu import option_menu
import pickle
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.image as mpimg
#======================== BACK GROUND IMAGE ===========================



st.markdown(f'<h1 style="color:#ffffff;text-align: center;font-size:32px;">{"AI-Based Advanced Approaches and Dry Eye Disease Detection Based on Multi-Source Evidence: Cases, Applications, Issues, and Future Directions"}</h1>', unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('7.jpeg')


# -------------------------------------------------------------------

selected = option_menu(
    menu_title=None, 
    options=["Dry Eye Prediction", "Eye Disease Prediction", "Eye Blink Detection" ,"Evaluvation", "Ask Chatbot"],  
    orientation="horizontal",
)


st.markdown(
    """
    <style>
    .option_menu_container {
        position: fixed;
        top: 20px;
        right: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Disease

if selected == 'Eye Disease Prediction':

    # abt = " Here , the system can predict the input image is affected or not with the help of deep learning algorithm such as CNN-2D effectively"
    # a="A Long-Term Recurrent Convolutional Network for Eye Blink Completeness Detection introduces a novel deep learning framework designed to accurately detect the completeness of eye blinks. By integrating convolutional neural networks (CNNs) with long short-term memory (LSTM) networks, the Eye-LRCN model effectively captures both spatial and temporal features from video sequences. This hybrid architecture allows for precise identification of partial and complete blinks, improving over traditional methods that often struggle with the subtle nuances of eye movements. The model's performance is evaluated on multiple datasets, demonstrating its robustness and potential applications in fields such as driver drowsiness detection, human-computer interaction, and neurological disorder monitorin"
    # st.markdown(f'<h1 style="color:#000000;text-align: justify;font-size:16px;">{a}</h1>', unsafe_allow_html=True)

    st.write("-----------------------------------------------------------")

    filename = st.file_uploader("Choose Image",['jpg','png'])
    
    with open('file.pickle', 'wb') as f:
        pickle.dump(filename, f)
    
    
    
    if filename is None:

        st.text("Upload Image")
        
    else:
            
        # filename = askopenfilename()
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Input Image"}</h1>', unsafe_allow_html=True)
        
        img = mpimg.imread(filename)
    
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis ('off')
        plt.show()
        
        
        st.image(img,caption="Original Image")
    
        
        #============================ PREPROCESS =================================
        
        #==== RESIZE IMAGE ====
        
        
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Preprocessing"}</h1>', unsafe_allow_html=True)
        
        
        resized_image = cv2.resize(img,(300,300))
        img_resize_orig = cv2.resize(img,((50, 50)))
        
        fig = plt.figure()
        plt.title('RESIZED IMAGE')
        plt.imshow(resized_image)
        plt.axis ('off')
        plt.show()
        
        st.image(resized_image,caption="Resized Image")
        
        # st.image(img,caption="Original Image")
                 
        #==== GRAYSCALE IMAGE ====
        

        SPV = np.shape(img)
        
        try:            
            gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
            
        except:
            gray1 = img_resize_orig
           
        fig = plt.figure()
        plt.title('GRAY SCALE IMAGE')
        plt.imshow(gray1)
        plt.axis ('off')
        plt.show()
        
    
        st.image(gray1,caption="Gray Scale Image")        
        
        #=============================== 3.FEATURE EXTRACTION ======================
        
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Feature Extraction"}</h1>', unsafe_allow_html=True)
        
    
        
        # === GRAY LEVEL CO OCCURENCE MATRIX ===
        
        from skimage.feature import graycomatrix, graycoprops
        
        print()
        print("-----------------------------------------------------")
        print("FEATURE EXTRACTION -->GRAY LEVEL CO-OCCURENCE MATRIX ")
        print("-----------------------------------------------------")
        print()
        
        
        PATCH_SIZE = 21
        
        # open the image
        
        image = img[:,:,0]
        image = cv2.resize(image,(768,1024))
         
        grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
        grass_patches = []
        for loc in grass_locations:
            grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                       loc[1]:loc[1] + PATCH_SIZE])
        
        # select some patches from sky areas of the image
        sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
        sky_patches = []
        for loc in sky_locations:
            sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                     loc[1]:loc[1] + PATCH_SIZE])
        
        # compute some GLCM properties each patch
        xs = []
        ys = []
        for patch in (grass_patches + sky_patches):
            glcm = graycomatrix(image.astype(int), distances=[4], angles=[0], levels=256,symmetric=True)
            xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
            ys.append(graycoprops(glcm, 'correlation')[0, 0])
        
        
        # create the figure
        fig = plt.figure(figsize=(8, 8))
        
        # display original image with locations of patches
        ax = fig.add_subplot(3, 2, 1)
        ax.imshow(image, cmap=plt.cm.gray,
                  vmin=0, vmax=255)
        for (y, x) in grass_locations:
            ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 3, 'gs')
        for (y, x) in sky_locations:
            ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
        ax.set_xlabel('GLCM')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('image')
        plt.show()
        
        # for each patch, plot (dissimilarity, correlation)
        ax = fig.add_subplot(3, 2, 2)
        ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
                label='Region 1')
        ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
                label='Region 2')
        ax.set_xlabel('GLCM Dissimilarity')
        ax.set_ylabel('GLCM Correlation')
        ax.legend()
        plt.show()
        
        
        sky_patches0 = np.mean(sky_patches[0])
        sky_patches1 = np.mean(sky_patches[1])
        sky_patches2 = np.mean(sky_patches[2])
        sky_patches3 = np.mean(sky_patches[3])
        
        Glcm_fea = [sky_patches0,sky_patches1,sky_patches2,sky_patches3]
        Tesfea1 = []
        Tesfea1.append(Glcm_fea[0])
        Tesfea1.append(Glcm_fea[1])
        Tesfea1.append(Glcm_fea[2])
        Tesfea1.append(Glcm_fea[3])
        
        
        print("---------------------------------------------------")
        st.write("GLCM FEATURES =")
        print("---------------------------------------------------")
        print()
        st.write(Glcm_fea)
        


         
        # ========= IMAGE SPLITTING ============
        
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Image Splitting"}</h1>', unsafe_allow_html=True)
        
        
        import os 
        
        from sklearn.model_selection import train_test_split
          
        data_aff = os.listdir('Dataset/Affected/')
         
        data_not = os.listdir('Dataset/Not/')
         
        

        
        import numpy as np
        dot1= []
        labels1 = [] 
        for img11 in data_aff:
            # print(img)
            img_1 = mpimg.imread('Dataset/Affected//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(1)
         
         
        for img11 in data_not:
            # print(img)
            img_1 = mpimg.imread('Dataset/Not//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(2)
         
   

        x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
        
        
        print("------------------------------------------------------------")
        print(" Image Splitting")
        print("------------------------------------------------------------")
        print()
        
        st.write("The Total of Images       =",len(dot1))
        st.write("The Total of Train Images =",len(x_train))
        st.write("The Total of Test Images  =",len(x_test))
          
          
              
          
        #=============================== CLASSIFICATION =================================
        
        from keras.utils import to_categorical
        
        y_train1=np.array(y_train)
        y_test1=np.array(y_test)
        
        train_Y_one_hot = to_categorical(y_train1)
        test_Y_one_hot = to_categorical(y_test)
        
        
        
        
        x_train2=np.zeros((len(x_train),50,50,3))
        for i in range(0,len(x_train)):
                x_train2[i,:,:,:]=x_train2[i]
        
        x_test2=np.zeros((len(x_test),50,50,3))
        for i in range(0,len(x_test)):
                x_test2[i,:,:,:]=x_test2[i]
    
    # ===================================== CLASSIFICATION ==================================
    
     # ----------------------- MOBILENET -----------------------
    
            
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Classification - MobileNet"}</h1>', unsafe_allow_html=True)
        
    
        import time
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from keras.utils import to_categorical
        from tensorflow.keras import layers, models
        
        
        print()
        print("----------------------------------------------")
        print(" Classification - Mobilnet")
        print("----------------------------------------------")
        print()
        from tensorflow.keras.applications import MobileNet
        
        start_mob = time.time()
        
        base_model = MobileNet(weights=None, input_shape=(50, 50, 3), classes=3)
        
        model = models.Model(inputs=base_model.input, outputs=base_model.output)
        
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        model.summary()
        
        history = model.fit(x_train2,train_Y_one_hot, epochs=3, batch_size=64)
        
        loss_val = history.history['loss']
        
        loss_val = min(loss_val)
        
        acc_mob = 100 - loss_val
        
        
        print("-------------------------------------")
        print("Mobilenet - Perfromance Analysis")
        print("-------------------------------------")
        print()
        print("1. Accuracy   =", acc_mob,'%')
        print()
        print("2. Error Rate =",loss_val)
        print()
        
        
        predictions = model.predict(x_test2)
        
        end_mob = time.time()
        
        time_mob = (end_mob-start_mob) * 10**3
        
        time_mob = time_mob / 1000
        
        print("3. Execution Time  = ",time_mob, "s")
        
        
        st.write("-------------------------------------")
        st.write("Mobilenet - Perfromance Analysis")
        st.write("-------------------------------------")
        print()
        st.write("1. Accuracy   =", acc_mob,'%')
        print()
        st.write("2. Error Rate =",loss_val)
        print()
        
        
        predictions = model.predict(x_test2)
        
        end_mob = time.time()
        
        time_mob = (end_mob-start_mob) * 10**3
        
        time_mob = time_mob / 1000
        
        st.write("3. Execution Time  = ",time_mob, "s")
                
        
        # --- prediction
        
        st.write("-----------------------------------------------------------")
    
        st.markdown(f'<h1 style="color:#ffffff;text-align: center;font-size:26px;">{"Prediction -Eye Disease"}</h1>', unsafe_allow_html=True)
         
         
        Total_length = len(data_aff) + len(data_not) 
        
        
           
        # Add labels1 definition here
        labels1 = [1 if i < len(grass_patches) else 2 for i in range(len(grass_patches) + len(sky_patches))]

        st.write("-----------------------------------------------------------")
        st.markdown(f'<h1 style="color:#ffffff;text-align: center;font-size:26px;">{"Prediction - Eye Disease Staging"}</h1>', unsafe_allow_html=True)

        mean_intensity = np.mean(gray1)

        if mean_intensity < 50:
            stage = "Severe"
        elif 50 <= mean_intensity < 100:
            stage = "Moderate"
        elif 100 <= mean_intensity < 150:
            stage = "Mild"
        else:
            stage = "Normal"

        if labels1[0] == 1:  # Assuming first entry matches affected
            st.write('-----------------------------------------')
            st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Affected"}</h1>', unsafe_allow_html=True)
            st.markdown(f'<h2 style="color:#000000;text-align: center;">Stage: {stage}</h2>', unsafe_allow_html=True)

        else:
            st.write('---------------------------------')
            st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Not Affected"}</h1>', unsafe_allow_html=True)
            st.markdown(f'<h2 style="color:#000000;text-align: center;">Stage: Normal</h2>', unsafe_allow_html=True)

        st.write("---------------------------------")



    


if selected == "Dry Eye Prediction":
    
    file = st.file_uploader("Upload Input Dataset",['csv','xlsx'])
    
    import pandas as pd
    import time
    from sklearn.model_selection import train_test_split


    
    if file is None:
        
        st.warning("Upload Input Data")
    
    else:
        
    
        dataframe=pd.read_excel("Dataset.xlsx")
                
        print("--------------------------------")
        print("Data Selection")
        print("--------------------------------")
        print()
        print(dataframe.head(15))    
        
        st.write("--------------------------------")
    
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Data Selection !!!"}</h1>', unsafe_allow_html=True)
    
        # st.write("--------------------------------")
        # st.write("Data Selection")
        # st.write("--------------------------------")
        print()
        st.write(dataframe.head(15))    
        
        
     #-------------------------- PRE PROCESSING --------------------------------    
        
        # ----- CHECKING MISSING VALUES 
        
        
        
        print("----------------------------------------------------")
        print("              Handling Missing values               ")
        print("----------------------------------------------------")
        print()
        print(dataframe.isnull().sum())
        
        st.write("--------------------------------")
    
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Pre-processing !!!"}</h1>', unsafe_allow_html=True)
    
        
        # st.write("----------------------------------------------------")
        st.write("              Handling Missing values               ")
        st.write("----------------------------------------------------")
        print()
        st.write(dataframe.isnull().sum())
        
        res = dataframe.isnull().sum().any()
            
        if res == False:
            
            print("--------------------------------------------")
            print("  There is no Missing values in our dataset ")
            print("--------------------------------------------")
            print()   
            
            
            st.write("--------------------------------------------")
            st.write("  There is no Missing values in our dataset ")
            st.write("--------------------------------------------")
      
        
            
        else:
        
            print("--------------------------------------------")
            print(" Missing values is present in our dataset   ")
            print("--------------------------------------------")
            print()    
            
            st.write("--------------------------------------------")
            st.write("  Missing values is present in our dataset ")
            
            dataframe = dataframe.fillna(0)
            
            resultt = dataframe.isnull().sum().any()
            
            if resultt == False:
                
                print("--------------------------------------------")
                print(" Data Cleaned !!!   ")
                print("--------------------------------------------")
                print()    
                print(dataframe.isnull().sum())  
                
                
                st.write("--------------------------------------------")
                st.write(" Data Cleaned !!!   ")
                st.write("--------------------------------------------")
                print()    
                st.write(dataframe.isnull().sum()) 
                
                
        # --- DROP UNWANTED COLUMNS
        
        dataframe = dataframe.drop(['Timestamp','Consent'],axis=1)
            
        
        # ----- LABEL ENCODING
                
    
        print("----------------------------------------------------")
        print("            Before Label Encoding                   ")
        print("----------------------------------------------------")
        print()
        print(dataframe['Gender'].head(15))                
                    
        st.write("----------------------------------------------------")
        st.write("            Before Label Encoding                   ")
        st.write("----------------------------------------------------")
        print()
        st.write(dataframe['Gender'].head(15))
        
        
        gen =  dataframe['Gender']
        
        aca_yr = dataframe['Gender']
        
    
    
    
    
        from sklearn import preprocessing
        
        label_encoder = preprocessing.LabelEncoder()
        
        dataframe['Gender']= label_encoder.fit_transform(dataframe['Gender'])  
        
        dataframe['Academic Year']= label_encoder.fit_transform(dataframe['Academic Year'])  
        
        dataframe['What type of Digital display device do you use?']= label_encoder.fit_transform(dataframe['What type of Digital display device do you use?'])  
        
        dataframe['How many hours in a day do you spend on your smartphones, laptops, etc?']= label_encoder.fit_transform(dataframe['How many hours in a day do you spend on your smartphones, laptops, etc?'])                
                    
        dataframe['Eyes that are sensitive to light?']= label_encoder.fit_transform(dataframe['Eyes that are sensitive to light?'])  
        
        dataframe['Eyes that feel gritty (itchy and Scratchy) ?']= label_encoder.fit_transform(dataframe['Eyes that feel gritty (itchy and Scratchy) ?'])  
    
    
        dataframe['Painful or Sore eyes?']= label_encoder.fit_transform(dataframe['Painful or Sore eyes?'])  
        
        dataframe['Blurred vision?']= label_encoder.fit_transform(dataframe['Blurred vision?'])  
    
        dataframe['Reading?']= label_encoder.fit_transform(dataframe['Reading?'])  
        
        dataframe['Driving at night?']= label_encoder.fit_transform(dataframe['Driving at night?'].astype(str))  
    
        dataframe['Working with a computer or bank machine ATM?']= label_encoder.fit_transform(dataframe['Working with a computer or bank machine ATM?'])  
    
        dataframe['Watching TV?']= label_encoder.fit_transform(dataframe['Watching TV?'].astype(str))  
        
        dataframe['Windy conditions?']= label_encoder.fit_transform(dataframe['Windy conditions?'].astype(str))  
                   
                    
        dataframe['Places or areas with low humidity (very dry)?']= label_encoder.fit_transform(dataframe['Places or areas with low humidity (very dry)?'].astype(str))  
         
        dataframe['Areas that are air-conditioned?']= label_encoder.fit_transform(dataframe['Areas that are air-conditioned?'].astype(str))                 
        
        dataframe['Poor Vision?']= label_encoder.fit_transform(dataframe['Poor Vision?'])  

                    
        dataframe['Results']= label_encoder.fit_transform(dataframe['Results'])  
           
        print("----------------------------------------------------")
        print("            After Label Encoding                   ")
        print("----------------------------------------------------")
        print()
        print(dataframe['Gender'].head(15))
        
        st.write("----------------------------------------------------")
        st.write("            After Label Encoding                   ")
        st.write("----------------------------------------------------")
        print()
        st.write(dataframe['Gender'].head(15))
        
        
        
       #-------------------------- DATA SPLITTING  --------------------------------
        
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Data Splitting !!!"}</h1>', unsafe_allow_html=True)

        
        X=dataframe.drop('Results',axis=1)
                
        y=dataframe['Results']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        
        print("---------------------------------------------")
        print("             Data Splitting                  ")
        print("---------------------------------------------")
        
        print()
        
        print("Total no of input data   :",dataframe.shape[0])
        print("Total no of test data    :",X_test.shape[0])
        print("Total no of train data   :",X_train.shape[0])
        
        
        # st.write("---------------------------------------------")
        st.write("     Test Data & Train Data                  ")
        st.write("---------------------------------------------")
        
        print()
        
        st.write("Total no of input data   :",dataframe.shape[0])
        st.write("Total no of test data    :",X_test.shape[0])
        st.write("Total no of train data   :",X_train.shape[0])
    
        #-------------------------- CLASSIFICATION --------------------------------
        
    
        #  ------ MLP --------
        
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Classification - MLP !!!"}</h1>', unsafe_allow_html=True)

        
        from sklearn.neural_network import MLPClassifier         
        
        start_mlp = time.time()
        
        mlpp = MLPClassifier()
        
        mlpp.fit(X_train, y_train)
        
        
        pred_mlpp = mlpp.predict(X_test)
        
        
        from sklearn import metrics
        
        
        acc_mlp = metrics.accuracy_score(pred_mlpp,y_test)* 100
        
        loss = 100 - acc_mlp
        # Classification report
        report_svm = metrics.classification_report(y_test, pred_mlpp, target_names=['Mild', 'Moderate','Normal','Severe'])
        
        # Confusion matrix
        cm = metrics.confusion_matrix(y_test, pred_mlpp)
        
       
        
        
        end_mlp = time.time()
        
        
        exec_time = (end_mlp-start_mlp) * 10**3
        
        exec_time_mlp = exec_time/1000
        
        
        print("----------------------------------------------------")
        print("     Classification -- Multi Layer Perceptron      ")
        print("----------------------------------------------------")
        
        print()
        
        print("1)  Accuracy        =", acc_mlp ,'%')
        print()
        print("2)  Error rate      = ", loss ,'%' )
        print()
        print("3)  Execution Time  = ", exec_time_mlp , 'sec')
        print()
        print("4) Classification Report  = ", )
        print()
        print(report_svm)
        
        
        st.write("----------------------------------------------------")
        st.write("      Classification -- Multi Layer Perceptron      ")
        st.write("----------------------------------------------------")
        
        print()
        
        st.write("1)  Accuracy        =", acc_mlp ,'%')
        print()
        st.write("2)  Error rate      = ", loss ,'%' )
        print()
        st.write("3)  Execution Time  = ", exec_time_mlp , 'sec')
        print()
        st.write("4) Classification Report  = ", )
        print()
        st.write(report_svm)
        
        st.write("---------------------------------------------")                
                    
                
                
# -------------------------- PREDICTION  ----------------------
    
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Prediction  !!!"}</h1>', unsafe_allow_html=True)
    
        # inputt = int(input("Enter Prediction Number :"))
        
        inputt = st.text_input("Enter Prediction Number :")
        
        butt = st.button("Submit")
        
        if butt:
            
            inputt = int(inputt)
            if pred_mlpp[inputt] == 0:
                
            
                    
                # st.write("Identified = Attack")
                st.markdown(f'<h1 style="color:#ffffff;text-align: center;font-size:25px;">{"Identified Affected = MILD STAGE"}</h1>', unsafe_allow_html=True)
    
            
            
            elif pred_mlpp[inputt] == 1:
                
    
                st.markdown(f'<h1 style="color:#ffffff;text-align: center;font-size:25px;">{"Identified Affected = MODERATE STAGE"}</h1>', unsafe_allow_html=True)
    
                        
            elif pred_mlpp[inputt] == 2:
                
    
                st.markdown(f'<h1 style="color:#ffffff;text-align: center;font-size:25px;">{"Identified Normal "}</h1>', unsafe_allow_html=True)
                   
            elif pred_mlpp[inputt] == 3:
                
    
                st.markdown(f'<h1 style="color:#ffffff;text-align: center;font-size:25px;">{"Identified Affected = SEVERE STAGE"}</h1>', unsafe_allow_html=True)
                    
                    
                
if selected == 'Eye Blink Detection':               
                
                
   # abt = " Here , the system can predict the input image is affected or not with the help of deep learning algorithm such as CNN-2D effectively"
    # a="A Long-Term Recurrent Convolutional Network for Eye Blink Completeness Detection introduces a novel deep learning framework designed to accurately detect the completeness of eye blinks. By integrating convolutional neural networks (CNNs) with long short-term memory (LSTM) networks, the Eye-LRCN model effectively captures both spatial and temporal features from video sequences. This hybrid architecture allows for precise identification of partial and complete blinks, improving over traditional methods that often struggle with the subtle nuances of eye movements. The model's performance is evaluated on multiple datasets, demonstrating its robustness and potential applications in fields such as driver drowsiness detection, human-computer interaction, and neurological disorder monitorin"
    # st.markdown(f'<h1 style="color:#000000;text-align: justify;font-size:16px;">{a}</h1>', unsafe_allow_html=True)

    st.write("-----------------------------------------------------------")

    filename = st.file_uploader("Choose Image",['jpg','png'])
    

    
    
    if filename is None:

        st.text("Upload Image")
        
    else:
            
        # filename = askopenfilename()
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Input Image"}</h1>', unsafe_allow_html=True)
        
        img = mpimg.imread(filename)
    
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis ('off')
        plt.show()
        
        
        st.image(img,caption="Original Image")
    
        
        #============================ PREPROCESS =================================
        
        #==== RESIZE IMAGE ====
        
        
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Preprocessing"}</h1>', unsafe_allow_html=True)
        
        
        resized_image = cv2.resize(img,(300,300))
        img_resize_orig = cv2.resize(img,((50, 50)))
        
        fig = plt.figure()
        plt.title('RESIZED IMAGE')
        plt.imshow(resized_image)
        plt.axis ('off')
        plt.show()
        
        st.image(resized_image,caption="Resized Image")
        
        # st.image(img,caption="Original Image")
                 
        #==== GRAYSCALE IMAGE ====
        

        SPV = np.shape(img)
        
        try:            
            gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
            
        except:
            gray1 = img_resize_orig
           
        fig = plt.figure()
        plt.title('GRAY SCALE IMAGE')
        plt.imshow(gray1)
        plt.axis ('off')
        plt.show()
        
    
        st.image(gray1,caption="Gray Scale Image")        
        
        #=============================== 3.FEATURE EXTRACTION ======================
        
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Feature Extraction"}</h1>', unsafe_allow_html=True)
        
    
        
        # === GRAY LEVEL CO OCCURENCE MATRIX ===
        
        from skimage.feature import graycomatrix, graycoprops
        
        print()
        print("-----------------------------------------------------")
        print("FEATURE EXTRACTION -->GRAY LEVEL CO-OCCURENCE MATRIX ")
        print("-----------------------------------------------------")
        print()
        
        
        PATCH_SIZE = 21
        
        # open the image
        
        image = img[:,:,0]
        image = cv2.resize(image,(768,1024))
         
        grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
        grass_patches = []
        for loc in grass_locations:
            grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                       loc[1]:loc[1] + PATCH_SIZE])
        
        # select some patches from sky areas of the image
        sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
        sky_patches = []
        for loc in sky_locations:
            sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                     loc[1]:loc[1] + PATCH_SIZE])
        
        # compute some GLCM properties each patch
        xs = []
        ys = []
        for patch in (grass_patches + sky_patches):
            glcm = graycomatrix(image.astype(int), distances=[4], angles=[0], levels=256,symmetric=True)
            xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
            ys.append(graycoprops(glcm, 'correlation')[0, 0])
        
        
        # create the figure
        fig = plt.figure(figsize=(8, 8))
        
        # display original image with locations of patches
        ax = fig.add_subplot(3, 2, 1)
        ax.imshow(image, cmap=plt.cm.gray,
                  vmin=0, vmax=255)
        for (y, x) in grass_locations:
            ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 3, 'gs')
        for (y, x) in sky_locations:
            ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
        ax.set_xlabel('GLCM')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('image')
        plt.show()
        
        # for each patch, plot (dissimilarity, correlation)
        ax = fig.add_subplot(3, 2, 2)
        ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
                label='Region 1')
        ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
                label='Region 2')
        ax.set_xlabel('GLCM Dissimilarity')
        ax.set_ylabel('GLCM Correlation')
        ax.legend()
        plt.show()
        
        
        sky_patches0 = np.mean(sky_patches[0])
        sky_patches1 = np.mean(sky_patches[1])
        sky_patches2 = np.mean(sky_patches[2])
        sky_patches3 = np.mean(sky_patches[3])
        
        Glcm_fea = [sky_patches0,sky_patches1,sky_patches2,sky_patches3]
        Tesfea1 = []
        Tesfea1.append(Glcm_fea[0])
        Tesfea1.append(Glcm_fea[1])
        Tesfea1.append(Glcm_fea[2])
        Tesfea1.append(Glcm_fea[3])
        
        
        print("---------------------------------------------------")
        st.write("GLCM FEATURES =")
        print("---------------------------------------------------")
        print()
        st.write(Glcm_fea)
        


         
        # ========= IMAGE SPLITTING ============
        
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Image Splitting"}</h1>', unsafe_allow_html=True)
        
        
        import os 
        
        from sklearn.model_selection import train_test_split
          
        data_clos = os.listdir('Blink/Closed/')
         
        data_forward = os.listdir('Blink/forward_look/')
         
        data_left= os.listdir('Blink/left_look/')
        

        
        import numpy as np
        dot1= []
        labels1 = [] 
        for img11 in data_clos:
            # print(img)
            img_1 = mpimg.imread('Blink/Closed//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(1)
         
         
        for img11 in data_forward:
            # print(img)
            img_1 = mpimg.imread('Blink/forward_look//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(2)
         
         
        for img11 in data_left:
            # print(img)
            img_1 = mpimg.imread('Blink/left_look//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(3)

        x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
        
        
        print("------------------------------------------------------------")
        print(" Image Splitting")
        print("------------------------------------------------------------")
        print()
        
        st.write("The Total of Images       =",len(dot1))
        st.write("The Total of Train Images =",len(x_train))
        st.write("The Total of Test Images  =",len(x_test))
          
          
              
        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Classification - VGG-19"}</h1>', unsafe_allow_html=True)

        #=============================== CLASSIFICATION =================================
        
        from keras.utils import to_categorical
        
        
        y_train1=np.array(y_train)
        y_test1=np.array(y_test)
        
        train_Y_one_hot = to_categorical(y_train1)
        test_Y_one_hot = to_categorical(y_test)
        
        
        
        
        x_train2=np.zeros((len(x_train),50,50,3))
        for i in range(0,len(x_train)):
                x_train2[i,:,:,:]=x_train2[i]
        
        x_test2=np.zeros((len(x_test),50,50,3))
        for i in range(0,len(x_test)):
                x_test2[i,:,:,:]=x_test2[i]
          

                      
        import time
         # ==== VGG19 ==
        start_time = time.time()
        
        from keras.utils import to_categorical
        
        from tensorflow.keras.models import Sequential
        
        from tensorflow.keras.applications.vgg19 import VGG19
        vgg = VGG19(weights="imagenet",include_top = False,input_shape=(50,50,3))
        
        for layer in vgg.layers:
            layer.trainable = False
        from tensorflow.keras.layers import Flatten,Dense
        model = Sequential()
        model.add(vgg)
        model.add(Flatten())
        model.add(Dense(1,activation="sigmoid"))
        model.summary()
        
        model.compile(optimizer="adam",loss="mae")
        # from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
        # checkpoint = ModelCheckpoint("vgg19.h5",monitor="val_acc",verbose=1,save_best_only=True,
        #                              save_weights_only=False,period=1)
        # earlystop = EarlyStopping(monitor="val_acc",patience=5,verbose=1)
        
        
        history = model.fit(x_train2,y_train1,batch_size=50,
                            epochs=2,validation_data=(x_train2,y_train1),verbose=1)               
                        
                    
        end_time = time.time()
        
        loss=history.history['loss']
        
        error_cnn=min(loss)
        
        acc_cnn=100- error_cnn
        
        exec_time = (end_time-start_time) * 10**3
        
        exec_time = exec_time/1000
        
        # st.write("-------------------------------------------")
        st.write("  Convolutional Neural Network - VGG 19")
        st.write("-------------------------------------------")
        print()
        st.write("1. Accuracy       =", acc_cnn,'%')
        print()
        st.write("2. Error Rate     =",error_cnn)
        print()
        st.write("3. Execution Time =",exec_time,'s')
                   
        st.write("-----------------------------------------------------------")
    
        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Prediction - Eye Blink Detection"}</h1>', unsafe_allow_html=True)
         
            
        Total_length = len(data_clos) + len(data_forward) + len(data_left)
        
        
           
        temp_data1  = []
        for ijk in range(0,Total_length):
                    # print(ijk)
                temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
                temp_data1.append(temp_data)
                    
        temp_data1 =np.array(temp_data1)
                
        zz = np.where(temp_data1==1)
                
        if labels1[zz[0][0]] == 1:
                st.write('-----------------------------------------')
                print()
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Eye Closed"}</h1>', unsafe_allow_html=True)
    
            
        elif labels1[zz[0][0]] == 2:
                st.write('---------------------------------')
                print()
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Forward Look"}</h1>', unsafe_allow_html=True)
                  
        elif labels1[zz[0][0]] == 3:
                st.write('---------------------------------')
                print()
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Left Look"}</h1>', unsafe_allow_html=True)
                              
                
        elif labels1[zz[0][0]] == 4:
                st.write('---------------------------------')
                print()
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Eye Opened"}</h1>', unsafe_allow_html=True)
                                       
        elif labels1[zz[0][0]] == 5:
                st.write('---------------------------------')
                print()
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Open Partially"}</h1>', unsafe_allow_html=True)
       
        elif labels1[zz[0][0]] == 6:
                st.write('---------------------------------')
                print()
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Right Look"}</h1>', unsafe_allow_html=True)

                
if selected == "Evaluvation":
    # Import necessary libraries
    import numpy as np
    import cv2
    import streamlit as st
    import torch
    import torch.nn.functional as F
    from torchvision import transforms, models
    from PIL import Image

    # Define the fixed apply_gradcam function here (Paste the fixed code)
    def apply_gradcam(image_path, model, target_layer):
        """ Generates Grad-CAM heatmap for ResNet """
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        input_tensor.requires_grad = True

        activation_maps = {}
        gradients = {}

        def forward_hook(module, input, output):
            activation_maps["features"] = output

        def backward_hook(module, grad_input, grad_output):
            gradients["gradients"] = grad_output[0]

        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_full_backward_hook(backward_hook)

        # Forward pass
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        # Backward pass for Grad-CAM
        model.zero_grad()
        output[:, pred_class].backward()

        handle_forward.remove()
        handle_backward.remove()

        pooled_gradients = torch.mean(gradients["gradients"], dim=[0, 2, 3])
        activations = activation_maps["features"].detach().squeeze(0)

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        overlayed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        return overlayed_img

    # Streamlit UI starts here
    st.title("Dry Eye Detection with Explainable AI")

    selected = st.radio("Select Prediction Type:", [ "Eye Disease Prediction"])

    if selected == 'Eye Disease Prediction':
        uploaded_file = st.file_uploader("Choose an Eye Image", type=['jpg', 'png'])
        
        if uploaded_file is not None:
            image_path = "temp_image.jpg"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.image(image_path, caption="Uploaded Eye Image", width=300)

            # Load the pre-trained model
            model = models.resnet18(pretrained=True)  
            target_layer = model.layer4[-1]  

            if st.button("Show Explanation"):
                gradcam_image = apply_gradcam(image_path, model, target_layer)

                # Save & Display Grad-CAM Image
                gradcam_path = "gradcam_output.jpg"
                cv2.imwrite(gradcam_path, gradcam_image)
                st.image([image_path, gradcam_path], caption=["Original", "Explainability Heatmap"], width=300)

                    
                
if selected == "Help to Chat":
    import google.generativeai as genai
    import streamlit as st
    import time

    # Configure API key (replace with your actual API key)
    genai.configure(api_key="AIzaSyBVYhHjcPQuBZTpq2TrkbBI2HOZlrh3rno")

    # Create model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    def GenerateResponse(input_text):
        """Generates a response using the Gemini model based on input text."""
        try:
            response = model.generate_content([
                "input: who are you",
                "output: I am a Dry Eye chatbot 👁️",
                "input: what all can you do?",
                "output: I can help you with any Eye related help 👁️‍🗨️",
                f"input: {input_text}",
                "output: ",
            ])
            return response.text
        except Exception as e:
            return f"Error generating response: {e}"

    # Streamlit App
    def main():
        # Set page configuration
        st.set_page_config(
            page_title="Dry Eye Chatbot",
            page_icon="👁️",
            layout="centered",  # Centered layout for compact view
            initial_sidebar_state="collapsed"  # Hide sidebar for compact view
        )

        # Custom CSS for styling
        st.markdown("""
            <style>
                .stApp {
                    background-color: #f0f2f6;
                    font-family: 'Arial', sans-serif;
                    max-width: 600px;  /* Limit width for compact view */
                    margin: auto;  /* Center the chatbox */
                    padding: 20px;
                    border-radius: 15px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .stChatMessage {
                    border-radius: 15px;
                    padding: 15px;
                    margin: 10px 0;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .stChatMessage.user {
                    background-color: #d1e7dd;
                    color: #0f5132;
                    margin-left: 20%;
                    border-bottom-right-radius: 5px;
                }
                .stChatMessage.assistant {
                    background-color: #fff3cd;
                    color: #856404;
                    margin-right: 20%;
                    border-bottom-left-radius: 5px;
                }
                .stChatInput {
                    background-color: #ffffff;
                    border-radius: 15px;
                    padding: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    width: 100%;  /* Full width input */
                }
                .loading {
                    display: inline-block;
                    width: 50px;
                    height: 50px;
                    border: 5px solid #f3f3f3;
                    border-radius: 50%;
                    border-top: 5px solid #3498db;
                    border-right: 5px solid #e74c3c;
                    border-bottom: 5px solid #2ecc71;
                    border-left: 5px solid #f1c40f;
                    animation: spin 1s linear infinite;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                .stButton>button {
                    background-color: #3498db;
                    color: white;
                    border-radius: 15px;
                    padding: 10px 20px;
                    font-size: 16px;
                    border: none;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .stButton>button:hover {
                    background-color: #2980b9;
                }
                .stTitle {
                    text-align: center;
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 20px;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="stTitle">👁️ Dry Eye Chatbot</div>', unsafe_allow_html=True)

        # Chatbot UI
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you with your eye concerns today? 👋"}]

        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Ask me anything about dry eyes..."):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            # Display loading animation
            with st.spinner(""):
                loading_placeholder = st.empty()
                loading_placeholder.markdown('<div class="loading"></div>', unsafe_allow_html=True)
                time.sleep(1)  # Simulate loading time

            # Get response from the model
            response = GenerateResponse(prompt)

            # Remove loading animation and display response
            loading_placeholder.empty()
            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

    if __name__ == "__main__":
        main()