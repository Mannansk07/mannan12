import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import numpy as np
import base64
from PIL import Image

# Set maximum image pixels to avoid decompression bomb warnings
Image.MAX_IMAGE_PIXELS = None

# Function to load image and encode it into base64
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Custom CSS for background image and styling
image_path = '../Images/background.png'  # Adjust path to your image
encoded_image = get_base64_image(image_path)

st.markdown(
    f"""
    <style>
    #   .stApp {{
    #     background-image: url('data:image/png;base64,{encoded_image}');
    #     background-repeat: no-repeat;
    #     background-attachment: fixed;
    #   }}
      .overlay {{
            background: rgba(0, 0, 0, 0.6);
            padding: 10px;
            border-radius: 10px;
        }}
      .title {{
            color: gold;
            font-size: 32px;
            font-weight: bold;
            text-align: center;
        }}
    </style>
    """,
    unsafe_allow_html=True
)



    
# Sidebar Menu
with st.sidebar:
    selected = option_menu(
        "MULTIPLE DISEASE PREDICTION SYSTEM",
        ["Home", "Disease Information", "Diabetes Prediction", "Heart Disease Prediction", "Pneumonia Prediction",
         "Parkinson's Prediction", "Lung Disease Prediction", "Liver Disease Prediction", 
         "Kidney Disease Prediction", "Dengue Prediction", "Feedback and Contact"],
        icons=["house", "info-circle", "person", "heart", "lungs", "activity", 
               "wind", "droplet", "tint", "bug", "envelope"],
        default_index=0
    )

# Load trained models
try:
    diabetes_model = pickle.load(open("../Models/diabetes.sav", 'rb'))
    heart_model = pickle.load(open("../Models/heart.sav", 'rb'))
    parkinson_model = pickle.load(open("../Models/parkinssons.sav", 'rb'))
    lung_model = pickle.load(open("../Models/lung cancer.sav", 'rb'))
    liver_model = pickle.load(open("../Models/liver_model.sav", 'rb'))
    kidney_model = pickle.load(open("../Models/kidney_disease.sav", 'rb')) 
    dengue_model = pickle.load(open("../Models/dengue_model.pkl",'rb'))
except Exception as e:
    st.error(f"Error loading models: {e}")

# Load the trained models
@st.cache_resource
def load_models():
    model_01 = tf.keras.models.load_model('../Models/model_01.h5')
    pneumonia_model = tf.keras.models.load_model('../Models/pneumonia.h5')
    return model_01, pneumonia_model

model_01, pneumonia_model = load_models()

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to match model input
    image = np.array(image) / 255.0   # Normalize pixel values
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)  # Convert grayscale to RGB
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to convert input data to numeric safely
def safe_float(value, default=0.0):
    try:
        return float(value)
    except ValueError:
        return default

# Home Page
if selected == "Home":
    st.markdown('<div class="overlay">', unsafe_allow_html=True)
    st.markdown('<div class="title">Welcome to the Multiple Disease Prediction System</div>', unsafe_allow_html=True)
    st.write(""" 
        This Web Application is designed to help users predict the likelihood of developing certain diseases based on their input features. With the use of trained and tested machine learning models, we provide predictions for Diabetes, Heart Disease, Pneumonia, Lung Cancer, Liver Disease, Kidney, and Parkinson’s Disease.

        Please select the disease prediction you want to check from the sidebar menu.
    """)

    # Display images side by side (using columns)
    col1, col2, col3 = st.columns(3)

    with col1:
         st.image("../Images/Diabetes.png", caption="Diabetes Prediction", use_container_width=True)

    with col2:
          st.image("../Images/Heart.png", caption="Heart Disease Prediction", use_container_width=True)

    with col3:
         st.image("../Images/Lung.png", caption="Lung Cancer Prediction", use_container_width=True)

    # Second row for more images
    col4, col5, col6 = st.columns(3)

    with col4:
         st.image("../Images/Liver.png", caption="Liver Disease Prediction", use_container_width=True)

    with col5:
         st.image("../Images/Parkinsons.png", caption="Parkinson's Prediction", use_container_width=True)

    with col6:
         st.image("../Images/Kidney.png", caption="Kidney Prediction", use_container_width=True)

    col7, col8 = st.columns(2)

    with col7:
        st.image("../Images/Dengue.png", caption="Dengue Prediction", use_container_width=True)
        
    with col8:
        st.image("../Images/penumonia.png", caption="Pneumonia Prediction", use_container_width=True)

    # Disclaimer Section
    st.markdown("---")
    st.markdown('<h3 style="color: gold;">Disclaimer</h3>', unsafe_allow_html=True)
    st.write(""" 
        This Web App may not provide accurate predictions at all times. When in doubt, please enter the values again and verify the predictions.
       **It is important to note that individuals with specific risk factors or concerns should consult with healthcare professionals for personalized advice and management.**
    """)
    st.markdown("---")

# Disease Information Page
if selected == "Disease Information":
    st.markdown('<div class="title" style="color: gold;">Disease Information</div>', unsafe_allow_html=True)

    # Create tabs for each disease
    tabs = st.tabs([" Heart Disease", " Diabetes", "Lung Disease", " Liver Disease", 
                    "Parkinson’s", " Kidney Disease","Dengue", "Pneumonia"])

    with tabs[0]:  # Heart Disease
        st.subheader("Heart Disease")
        st.write("Heart disease refers to various types of heart conditions. The most common is coronary artery disease (CAD).")
        st.image("../Images/Heart.png", width=400)
        st.markdown("### Causes")
        st.write("- High blood pressure\n- High cholesterol\n- Smoking\n- Obesity")
        st.markdown("### Symptoms")
        st.write("- Chest pain\n- Shortness of breath\n- Fatigue\n- Irregular heartbeat")
        st.markdown("### Prevention")
        st.write("- Exercise regularly\n- Eat a healthy diet\n- Control blood pressure and cholesterol\n- Quit smoking")

    with tabs[1]:  # Diabetes
        st.subheader("Diabetes")
        st.write("Diabetes occurs when blood sugar levels are too high.")
        st.image("../Images/Diabetes.png", width=400)
        st.markdown("### Causes")
        st.write("- Insulin resistance\n- Genetics\n- Obesity\n- Sedentary lifestyle")
        st.markdown("### Symptoms")
        st.write("- Increased thirst\n- Frequent urination\n- Unexplained weight loss")
        st.markdown("### Prevention")
        st.write("- Maintain a healthy weight\n- Eat fiber-rich foods\n- Monitor blood sugar levels")

    with tabs[2]:  # Lung Disease
        st.subheader("Lung Disease")
        st.write("Lung diseases are conditions that impair lung function, such as COPD and lung cancer.")
        st.image("../Images/Lung.png", width=400)
        st.markdown("### Causes")
        st.write("- Smoking\n- Air pollution\n- Infections")
        st.markdown("### Symptoms")
        st.write("- Chronic cough\n- Wheezing\n- Shortness of breath")
        st.markdown("### Prevention")
        st.write("- Avoid smoking\n- Exercise regularly\n- Wear masks in polluted areas")

    with tabs[3]:  # Liver Disease
        st.subheader("Liver Disease")
        st.write("Liver disease includes hepatitis, fatty liver, and cirrhosis.")
        st.image("../Images/Liver.png", width=400)
        st.markdown("### Causes")
        st.write("- Excessive alcohol consumption\n- Viral infections\n- Fatty liver")
        st.markdown("### Symptoms")
        st.write("- Jaundice\n- Abdominal pain\n- Swelling in legs")
        st.markdown("### Prevention")
        st.write("- Limit alcohol intake\n- Eat a balanced diet\n- Get vaccinated for hepatitis")

    with tabs[4]:  # Parkinson's Disease
        st.subheader("Parkinson's Disease")
        st.write("Parkinson’s disease is a progressive nervous system disorder affecting movement.")
        st.image("../Images/Parkinsons.png", width=400)
        st.markdown("### Causes")
        st.write("- Genetic mutations\n- Aging\n- Environmental toxins")
        st.markdown("### Symptoms")
        st.write("- Tremors\n- Slow movement\n- Stiffness")
        st.markdown("### Prevention")
        st.write("- Regular exercise\n- Healthy diet\n- Avoid toxins")

    with tabs[5]:  # Kidney Disease
        st.subheader("Kidney Disease")
        st.write("Kidney disease affects the ability of the kidneys to filter waste.")
        st.image("../Images/Kidney.png", width=400)
        st.markdown("### Causes")
        st.write("- Diabetes\n- High blood pressure\n- Infections")
        st.markdown("### Symptoms")
        st.write("- Swelling in legs\n- Fatigue\n- Frequent urination")
        st.markdown("### Prevention")
        st.write("- Drink enough water\n- Control blood sugar levels\n- Eat less salt")

    with tabs[6]:  # Dengue
        st.subheader("Dengue")
        st.write("Dengue is a mosquito-borne viral infection that causes flu-like illness and can develop into severe dengue, which can be fatal.")
        st.image("../Images/Dengue.png", width=400)
        st.markdown("### Causes")
        st.write("- Bite of an infected Aedes mosquito\n- Dengue virus transmission through mosquito bites\n- Lack of proper sanitation")
        st.markdown("### Symptoms")
        st.write("- High fever\n- Severe headaches\n- Joint and muscle pain\n- Skin rash\n- Nausea and vomiting")
        st.markdown("### Prevention")
        st.write("- Use mosquito repellent\n- Wear long-sleeved clothing\n- Avoid stagnant water near homes\n- Use mosquito nets while sleeping")

    with tabs[7]:  # Pneumonia
        st.subheader("Pneumonia")
        st.write("Pneumonia is an infection that inflames the air sacs in one or both lungs, causing difficulty in breathing.")
        st.image("../Images/penumonia.png", width=400)
        st.markdown("### Causes")
        st.write("- Bacterial infections (Streptococcus pneumoniae)\n- Viral infections (Influenza, COVID-19)\n- Fungal infections")
        st.markdown("### Symptoms")
        st.write("- Cough with phlegm or pus\n- Fever\n- Chills\n- Difficulty breathing\n- Chest pain")
        st.markdown("### Prevention")
        st.write("- Get vaccinated\n- Wash hands frequently\n- Avoid smoking\n- Maintain good hygiene\n- Seek early treatment for respiratory infections")

# Diabetes Prediction
if selected == "Diabetes Prediction":
    st.markdown('<div class="title" style="color: gold;">Diabetes Prediction using Machine Learning</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = safe_float(st.text_input("Number of Pregnancies"))
        SkinThickness = safe_float(st.text_input("Skin Thickness value"))
        BMI = safe_float(st.text_input("BMI"))
        DiabetesPedigreeFunctionvalue = safe_float(st.text_input("Diabetes Pedigree Function value"))

    with col2:
        BloodPressure = safe_float(st.text_input("BloodPressure"))
        Glucose = safe_float(st.text_input("Glucose Level"))
        Insulin = safe_float(st.text_input("Insulin Level"))
        AgeofthePerson = safe_float(st.text_input("Age of the Person"))

    if st.button("Predict Diabetes"):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, SkinThickness, Insulin, DiabetesPedigreeFunctionvalue, AgeofthePerson ,BMI ,BloodPressure]])
        result = "The person is diabetic" if diab_prediction[0] == 1 else "The person is not diabetic"
        st.success(result)

# Heart Disease Prediction
if selected == "Heart Disease Prediction":
    st.markdown('<div class="title" style="color: gold;">Heart Disease Prediction using Machine Learning</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = safe_float(st.text_input("Age"))
        trestbps = safe_float(st.text_input("Resting Blood Pressure"))
        restecg = safe_float(st.text_input("Resting Electrocardiographic Results"))
        oldpeak = safe_float(st.text_input("ST Depression Induced by Exercise"))

    with col2:
        sex = safe_float(st.text_input("Sex (1 = Male, 0 = Female)"))
        chol = safe_float(st.text_input("Serum Cholesterol in mg/dl"))
        thalach = safe_float(st.text_input("Maximum Heart Rate Achieved"))
        slope = safe_float(st.text_input("Slope of Peak Exercise ST Segment"))

    with col3:
        cp = safe_float(st.text_input("Chest Pain Types"))
        fbs = safe_float(st.text_input("Fasting Blood Sugar > 120 mg/dl"))
        exang = safe_float(st.text_input("Exercise Induced Angina"))
        ca = safe_float(st.text_input("Number of Major Vessels Colored by Fluoroscopy"))
        thal = safe_float(st.text_input("Thalassemia (0-3)"))  

    if st.button("Heart Disease Test Result"):
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        result = "The person has heart disease" if heart_prediction[0] == 1 else "The person does not have heart disease"
        st.success(result)

# Kidney Disease Prediction
if selected == "Kidney Disease Prediction":
    st.markdown('<div class="title" style="color: gold;">Kidney Disease Prediction using Machine Learning</div>', unsafe_allow_html=True)

    # Input features for Kidney Disease
    features = [
        "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar", 
        "Red Blood Cells", "Pus Cell", "Polynuclear Cells", "Cell Size", "Mucus", 
        "Blood Glucose Random", "Blood Urea", "Serum Creatinine", "Sodium", "Potassium", 
        "Hemoglobin", "Packed Cell Volume", "White Blood Cell Count", "Red Blood Cell Count", 
        "Hypertension", "Diabetes Mellitus", "Coronary Artery Disease", "Appetite", 
        "Pedal Edema", "Anemia"
    ]
    
    # Collecting input data for Kidney disease prediction
    inputs = [safe_float(st.text_input(feature)) for feature in features]

    if st.button("Kidney Disease Test Result"):
        kidney_prediction = kidney_model.predict([inputs])
        result = "The person may have kidney disease" if kidney_prediction[0] == 1 else "The person does not have kidney disease"
        st.success(result)

# Pneumonia Prediction Page
if selected == "Pneumonia Prediction":
    st.markdown('<div class="title" style="color: gold;">Pneumonia Detection using Deep Learning</div>', unsafe_allow_html=True)
    
    st.write("""
        **Upload a chest X-ray image** to determine if the patient has pneumonia or not.
        The model will analyze the image and provide a prediction.
    """)
    
    # File uploader for image input
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)
        
        # Preprocess the image for the model
        processed_image = preprocess_image(image)
        
        # Make a prediction
        with st.spinner("Analyzing the image..."):
            prediction = pneumonia_model.predict(processed_image)
            result = "Pneumonia Detected" if prediction[0][0] > 0.5 else "No Pneumonia Detected"
            confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
        
        # Display the result
        st.success(f"**Result:** {result}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")
        
        # Option to download the report
        if st.button("Generate PDF Report"):
            user_inputs = {
                "Uploaded Image": uploaded_file.name,
                "Prediction Result": result,
                "Confidence": f"{confidence * 100:.2f}%"
            }
            generate_pdf_report("Pneumonia Detection", result, user_inputs)

# Parkinson's Disease Prediction
if selected == "Parkinson's Prediction":
    st.markdown("<div class='title' style='color: gold;'>Parkinson's Disease Prediction using Machine Learning</div>", unsafe_allow_html=True)

    input_features = [
        "MDVP:Fo (Hz)", "MDVP:Fhi (Hz)", "MDVP:Flo (Hz)", "MDVP:Jitter (%)", "MDVP:Jitter (Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer (dB)",
        "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR",
        "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]
    
    inputs = [safe_float(st.text_input(feature)) for feature in input_features]

    if st.button("Parkinson's Test Result"):
        parkinson_prediction = parkinson_model.predict([inputs])
        result = "The person has Parkinson’s disease" if parkinson_prediction[0] == 1 else "The person does not have Parkinson’s disease"
        st.success(result)

# Lung Disease Prediction
if selected == "Lung Disease Prediction":
    st.markdown('<div class="title" style="color: gold;">Lung Disease Prediction using Machine Learning</div>', unsafe_allow_html=True)

    features = ["Age", "Gender (1 = Male, 0 = Female)", "Smoking (1 = Yes, 0 = No)",
                "Yellow Fingers", "Anxiety", "Chronic Disease", "Fatigue",
                "Allergy", "Wheezing", "Alcohol Consumption", "Coughing",
                "Shortness of Breath", "Swallowing Difficulty", "Chest Pain"]

    inputs = [safe_float(st.text_input(feature)) for feature in features]

    if st.button("Lung Disease Test Result"):
        lung_prediction = lung_model.predict([inputs])
        result = "The person may have lung disease" if lung_prediction[0] == 1 else "The person does not have lung disease"
        st.success(result)

# Liver Disease Prediction
if selected == "Liver Disease Prediction":
    st.markdown('<div class="title" style="color: gold;">Liver Disease Prediction using Machine Learning</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        Gender = safe_float(st.text_input("Gender (0 for Male, 1 for Female)"))  # Missing feature added
        Age = safe_float(st.text_input("Age"))
        Total_Bilirubin = safe_float(st.text_input("Total Bilirubin"))

    with col2:
        Direct_Bilirubin = safe_float(st.text_input("Direct Bilirubin"))
        Alkaline_Phosphotase = safe_float(st.text_input("Alkaline Phosphotase"))
        Alanine_Aminotransferase = safe_float(st.text_input("Alanine Aminotransferase"))

    with col3:
        Aspartate_Aminotransferase = safe_float(st.text_input("Aspartate Aminotransferase"))
        Total_Proteins = safe_float(st.text_input("Total Proteins"))
        Albumin = safe_float(st.text_input("Albumin"))
        Albumin_and_Globulin_Ratio = safe_float(st.text_input("Albumin and Globulin Ratio"))

    if st.button("Liver Disease Test Result"):
        # Validate inputs (ensure all values are entered)
        if None in [Gender, Age, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alanine_Aminotransferase, 
                    Aspartate_Aminotransferase, Total_Proteins, Albumin, Albumin_and_Globulin_Ratio]:
            st.error("Please enter valid numeric values for all fields.")
        else:
            # Convert input to NumPy array (ensures correct format)
            input_data = np.array([[Gender, Age, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, 
                                    Alanine_Aminotransferase, Aspartate_Aminotransferase, Total_Proteins, 
                                    Albumin, Albumin_and_Globulin_Ratio]], dtype=np.float32)

            # Ensure correct shape
            input_data = input_data.reshape(1, -1)

            # Predict
            liver_prediction = liver_model.predict(input_data)

            # Display result
            result = "The person may have liver disease" if liver_prediction[0] == 1 else "The person does not have liver disease"
            st.success(result)

# Dengue Prediction Page
if selected == "Dengue Prediction":
    st.markdown('<div class="title" style="color: gold;">Dengue Prediction using Machine Learning</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        gender = safe_float(st.text_input("Gender (1 = Male, 0 = Female)"))
        age = safe_float(st.text_input("Age"))
        hemoglobin = safe_float(st.text_input("Hemoglobin (g/dl)"))
        neutrophils = safe_float(st.text_input("Neutrophils (%)"))
        lymphocytes = safe_float(st.text_input("Lymphocytes (%)"))
        monocytes = safe_float(st.text_input("Monocytes (%)"))
        eosinophils = safe_float(st.text_input("Eosinophils (%)"))
        rbc = safe_float(st.text_input("Red Blood Cell Count (RBC)"))
        hct = safe_float(st.text_input("Hematocrit (HCT %)"))

    with col2:
        mcv = safe_float(st.text_input("Mean Corpuscular Volume (MCV fl)"))
        mch = safe_float(st.text_input("Mean Corpuscular Hemoglobin (MCH pg)"))
        mchc = safe_float(st.text_input("Mean Corpuscular Hemoglobin Concentration (MCHC g/dl)"))
        rdw_cv = safe_float(st.text_input("Red Cell Distribution Width (RDW-CV %)"))
        platelet_count = safe_float(st.text_input("Total Platelet Count (/cumm)"))
        mpv = safe_float(st.text_input("Mean Platelet Volume (MPV fl)"))
        pdw = safe_float(st.text_input("Platelet Distribution Width (PDW %)"))
        pct = safe_float(st.text_input("Plateletcrit (PCT %)"))
        wbc = safe_float(st.text_input("Total White Blood Cell Count (/cumm)"))

    if st.button("Predict Dengue"):
        input_features = np.array([[gender, age, hemoglobin, neutrophils, lymphocytes, monocytes, eosinophils,
                                    rbc, hct, mcv, mch, mchc, rdw_cv, platelet_count, mpv, pdw, pct, wbc]])

        dengue_prediction = dengue_model.predict(input_features)
        result = "The person is likely to have dengue" if dengue_prediction[0] == 1 else "The person is not likely to have dengue"
        
        st.success(result)

# Feedback and Contact Page
if selected == "Feedback and Contact":
    st.markdown('<div class="title" style="color: gold;">Feedback and Contact</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["💭 Feedback", "✉️ Contact"])
    
    with tabs[0]:  # Feedback tab
        st.markdown("""
            <h2 style='text-align: center; color: white; font-size: 2.5em;'>Your Feedback is Valuable!</h2>
        """, unsafe_allow_html=True)
        
        st.markdown("<p style='text-align: center; color: #ffffff; font-size: 1.2em;'>Please rate your overall experience in using our Web App</p>", unsafe_allow_html=True)
        
        # Creating a row of star rating buttons
        cols = st.columns(5)
        rating = 0
        
        # Custom CSS for star buttons
        st.markdown("""
            <style>
                div.stButton > button {
                    background-color: rgba(0,0,0,0.2);
                    color: gold;
                    border: 1px solid rgba(255,255,255,0.1);
                    padding: 15px 30px;
                    width: 100%;
                }
                div.stButton > button:hover {
                    background-color: rgba(0,0,0,0.4);
                    color: gold;
                    border: 1px solid rgba(255,255,255,0.2);
                }
            </style>
        """, unsafe_allow_html=True)
        
        with cols[0]:
            if st.button("⭐"):
                rating = 1
        with cols[1]:
            if st.button("⭐⭐"):
                rating = 2
        with cols[2]:
            if st.button("⭐⭐⭐"):
                rating = 3
        with cols[3]:
            if st.button("⭐⭐⭐⭐"):
                rating = 4
        with cols[4]:
            if st.button("⭐⭐⭐⭐⭐"):
                rating = 5
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #ffffff; font-size: 1.2em;'>Have questions or suggestions? We'd love to hear from you.</p>", unsafe_allow_html=True)
        
        # Custom CSS for textarea
        st.markdown("""
            <style>
                textarea {
                    background-color: rgba(0,0,0,0.2) !important;
                    color: white !important;
                    border: 1px solid rgba(255,255,255,0.1) !important;
                    border-radius: 5px !important;
                }
            </style>
        """, unsafe_allow_html=True)
        
        feedback = st.text_area("", placeholder="Type here...", height=200)
        
        # Center the submit button
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            submit_button = st.button("Submit", key="feedback_submit")
            if submit_button:
                if feedback:
                    st.success("Thank you for your feedback!")
                    # Here you can add code to handle the feedback submission
                else:
                    st.warning("Please enter your feedback before submitting.")
    
    with tabs[1]:  # Contact tab
        st.markdown("""
            <h2 style='text-align: center; color: white; font-size: 2em;'>Contact Us</h2>
        """, unsafe_allow_html=True)
        
        contact_form = """
        <form>
            <input type="text" placeholder="Your Name" style="width: 100%; padding: 12px; margin: 8px 0; background: rgba(0,0,0,0.2); color: white; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px;">
            <input type="email" placeholder="Your Email" style="width: 100%; padding: 12px; margin: 8px 0; background: rgba(0,0,0,0.2); color: white; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px;">
            <textarea placeholder="Your Message" style="width: 100%; height: 150px; padding: 12px; margin: 8px 0; background: rgba(0,0,0,0.2); color: white; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px;"></textarea>
        </form>
        """
        st.markdown(contact_form, unsafe_allow_html=True)
        st.balloons()

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: white;">
        <p>© 2025 Multiple Disease Prediction System | All Rights Reserved</p>
    </div>
""", unsafe_allow_html=True)