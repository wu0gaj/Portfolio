import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd

#Loading the supplementary data to look up Lego sets

df1 = pd.read_csv("C:/Users/miro/Desktop/wercia/Bootcamp/DL project/Rozpoznawanie zestawów/set/inventories.csv")
df2 = pd.read_csv("C:/Users/miro/Desktop/wercia/Bootcamp/DL project/Rozpoznawanie zestawów/set/inventory_parts.csv")
df3 = pd.read_csv("C:/Users/miro/Desktop/wercia/Bootcamp/DL project/Rozpoznawanie zestawów/set/sets.csv")
df4 = pd.read_csv("C:/Users/miro/Desktop/wercia/Bootcamp/DL project/Rozpoznawanie zestawów/set/parts.csv")


#Merge tables inventory and inventory_parts
attempt1 = pd.merge(df1,df2, left_on = 'id', right_on = 'inventory_id')

#Merge the above with table sets
attempt2 = pd.merge(attempt1,df3, left_on = 'set_num', right_on = 'set_num')

#Final shape of the table
final = attempt2.drop(["id", "version", "inventory_id", "color_id","is_spare", "theme_id"], axis=1)

#List of all parts (classes)

bricks = [10247,11090,11211,11212,11214,11458,11476,11477,14704,14719,14769,15068,15070,15100,15379,15392,15535,15573,15712,18651,18654,
18674,18677,20482,22388,22885,2357,"2412b",2420,24201,24246,2429,2430,2431,2432,2436,2445,2450,2454,2456,24866,25269,2540,26047,
2654,26601,26603,26604,2780,27925,28192,2877,3001,3002,3003,3004,3005,3008,3009,3010,30136,3020,3021,3022,3023,3024,3031,3032,
3034,3035,3037,30374,3039,3040,30413,30414,"3062b",3065,"3068b","3069b","3070b",32000,32013,32028,32054,32062,32064,32073,32123,32140,
32184,32278,32316,"3245c",32523,32524,32525,32526,32607,32952,33291,33909,34103,3460,35480,3622,3623,3660,3665,3666,3673,3700,
3701,3705,3710,3713,3749,3795,3832,3937,3941,3958,4032,40490,4070,4073,"4081b",4085,4162,41677,41740,41769,41770,42003,4274,
4286,43093,43722,43723,44728,4477,4519,4589,"4599b",4740,47457,48336,4865,48729,49668,50950,51739,53451,54200,59443,60470,60474,
60478,60479,60481,60483,60592,60601,6091,61252,6134,61409,61678,62462,63864,63868,63965,64644,6536,6541,6558,6632,6636,85080,
85861,85984,87079,87083,87087,87552,87580,87620,87994,88072,88323,92280,92946,93273,98138,98283,99206,99207,99563,99780,99781]


bricks4 = [10247,11090,11211,11212,11214,11458,11476,11477,14704,14719,14769,15068,15070,15100,15379,15392,15535,15573,15712,18651,18654,
18674,18677,20482,22388,22885,2357,2420,24201,24246,2429,2430,2431,2432,2436,2445,2450,2454,2456,24866,25269,2540,26047,
2654,26601,26603,26604,2780,27925,28192,2877,3001,3002,3003,3004,3005,3008,3009,3010,30136,3020,3021,3022,3023,3024,3031,3032,
3034,3035,3037,30374,3039,3040,30413,30414, 3065,32000,32013,32028,32054,32062,32064,32073,32123,32140,
32184,32278,32316,32523,32524,32525,32526,32607,32952,33291,33909,34103,3460,35480,3622,3623,3660,3665,3666,3673,3700,
3701,3705,3710,3713,3749,3795,3832,3937,3941,3958,4032,40490,4070,4073,4085,4162,41677,41740,41769,41770,42003,4274,
4286,43093,43722,43723,44728,4477,4519,4589,4740,47457,48336,4865,48729,49668,50950,51739,53451,54200,59443,60470,60474,
60478,60479,60481,60483,60592,60601,6091,61252,6134,61409,61678,62462,63864,63868,63965,64644,6536,6541,6558,6632,6636,85080,
85861,85984,87079,87083,87087,87552,87580,87620,87994,88072,88323,92280,92946,93273,98138,98283,99206,99207,99563,99780,99781]


def add_apostrophe(lst):
    return [f"{element}" for element in lst]

#classes = ['2357', '2412b', '2420', '2429','2430']

classes = add_apostrophe(bricks)

#classes = ['2357', '2412b', '2420', '2429', '2430', '2431', '2432', '2436', '2445', '2450', '2454', '2456', '2540', '2654', '2780', '2877', '3001', '3002', '3003', '3004']

# Load the trained CNN model
model = tf.keras.models.load_model('C:/Users/miro/Desktop/wercia/Bootcamp/DL project/saved_models/simple_model2.hdf5')

#Function to preprocess the image
def preprocess_image(image):
    # Resize the image to the required input shape of the CNN model
    image = image.resize((64, 64))
    # Convert the image to a NumPy array and normalize the pixel values
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    # Expand the dimensions of the image to match the CNN model's input shape
    image = tf.expand_dims(image, axis=0)
    return image

# Streamlit app
st.image("streamlit3.png")

st.text("")
st.text("")

st.subheader("Welcome!")
st.text("Check your brick! is an app created to help you identify your LEGO parts.\nJust upload the picture of your brick below and we will let you know the serial\nnumber as well as suggest some sets where you can find your piece.")

# Upload image file
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Perform inference using the CNN model
    predictions = model.predict(processed_image)
    predicted_class = classes[predictions.argmax()]

    df5 = df4[df4["part_num"] == predicted_class]
    df6 = df5.iloc[0]["name"]

    # Display the predicted Lego brick part
    st.write("Your brick serial number is: ", predicted_class, " - ", df6)
   
    #Display a table with suggested Lego sets
    suggestions = final[final['part_num'] == predicted_class]

    st.write("You can find this part in the following sets:")
    st.dataframe(suggestions, column_config={"set_num": "Set",
                                             "part_num": "Part",
                                             "quantity": "Quantity",
                                             "name": "Part name",
                                             "year":"Release year",
                                             "num_parts":"Parts quanity"},
                                             hide_index=True)
