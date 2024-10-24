import ollama

def chat_with_llama():
    print("Start chatting with the AI. 'exit' command to exit")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            break
        
        stream = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": user_input}],
            stream=True
        )
        
        print("LLaMa: ", end="")
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
            
chat_with_llama()