from flask import Flask, render_template, request, jsonify
import ollama

app = Flask(__name__)

context = "You are the manager of a 5-star restaurant - give precise and accurate responses. Here are some details about the restaurant - Opens at 5pm, Closes at 11pm and it is closed on Tuesdays. The restaurant specializes in European cuisines. Respond to guest inquiries accordingly. Here is the first query: "
messages = [{"role": "system", "content": context}]

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    messages.append({"role": "user", "content": user_message})
    
    response = ollama.chat(
        model='llama3.2',
        messages=messages
    )
    
    ai_response = response["message"]["content"]
    messages.append({"role": "assistant", "content": ai_response})
    
    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(debug=True)