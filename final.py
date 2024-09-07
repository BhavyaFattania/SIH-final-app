import streamlit as st
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

# Load the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Initialize session state to store chat history
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("Visual Question Answering Chatbot")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Store the uploaded image in session state
    st.session_state.image = image
    
    # Chat interface
    user_input = st.text_input("You: ", key="input")

    if st.button("Send"):
        if user_input:
            # Process the image and question
            inputs = processor(st.session_state.image, user_input, return_tensors="pt")
            output = model.generate(**inputs)
            answer = processor.decode(output[0], skip_special_tokens=True)
            
            # Add user question and model answer to chat history
            st.session_state.history.append({"user": user_input, "bot": answer})
    
    # Display the chat history
    if st.session_state.history:
        for i, chat in enumerate(st.session_state.history):
            st.write(f"**You:** {chat['user']}")
            st.write(f"**Bot:** {chat['bot']}")
