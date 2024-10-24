import ollama

def chat_with_llama():
    print("Start chatting with the AI! Type 'exit' to stop.")
    
    context = "You are the manager of a 5-star restaurant - give precise and accurate responses. Here are some details about the restaurant - Opens at 5pm and closes at 11pm. Remember that the restaurant is closed on Tuesdays and that it specialises in European cuisine. Respond to the cusotmers queries accordingly. Here is the first query: "
                
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            break
        
        stream = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': context+user_input}],
            stream=True,
        )
        
        print("Manager: ", end='')
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        print("\n")

chat_with_llama()
    
