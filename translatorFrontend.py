import streamlit as st
import transformers
from transformers import T5TokenizerFast, T5ForConditionalGeneration

st.title('Translator from Anglais to Francais:exclamation:')
st.write("Enter the text you'd like to translate:")

user_input = st.text_input("English Text:", "")

clicked = st.button("Translate")

st.write("French Translation:")

tokenizer = T5TokenizerFast.from_pretrained('t5-base')

model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)

if clicked and len(user_input) == 0:
    st.write("Come on now lets enter some text translate, i'll give you 3 seconds ...3...2...1...:gun:")
elif clicked and len(user_input) != 0:
    input_text = "translate English to French: " + user_input

    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids)

    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.text_area("French Text:", translation)
    
