from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from transformers import pipeline
import os

load_dotenv("../.env")

app = Flask(__name__)

# client = InferenceClient(
#     provider="fireworks-ai",
#     api_key=os.getenv("CHATBOT_API_KEY")
# )
client = InferenceClient(
            api_key=os.getenv("CHATBOT_API_KEY"),
        )

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct",
    trust_remote_code=True)
       

@app.route("/")
def home():
    return render_template("template.html")

@app.route("/chat", methods=["POST"])
def chat():

    try:
         
        user_message = request.json["message"]

        # response = client.chat.completions.create(
        #     model="meta-llama/Llama-3.1-8B-Instruct",
        #     messages=[
        #         {   
        #             "role": "user", 
        #             "content": user_message}
        #     ],            
        #     max_tokens=100
        # )

        # messages = [
        #         {"role": "user", "content": user_message},
        #     ]
        
        prompt = f"User: {user_message}\nAssistant:"
        result = pipe(
            prompt,
            max_new_tokens=200,
            clean_up_tokenization_spaces=False
        )

        # pipe(result)  

             
        # completion = client.chat.completions.create(
        #     model="deepseek-ai/DeepSeek-V4-Pro:novita",
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": user_message
        #         }
        #     ],
        #     max_tokens=100
        # )

        print(result)

        #bot_reply = response  # returns a string dir
       
        # response = client.chat.completions.create(
        #     model="gpt-4.1-mini",
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": user_message
        #         }
        #     ]
        # )
        generated_text = result[0]["generated_text"]
        bot_reply = generated_text.split("Assistant:")[-1].strip()
        # completion.choices[0].message.content
        return jsonify({
            "reply": bot_reply
        })

    except Exception as e:

        return jsonify({
            "reply": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)
# print("hello")